//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_KNOWNBITSANALYSIS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

llvm::SmallVector<std::string> split (std::string s, std::string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  llvm::SmallVector<std::string> res;

  while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
    token = s.substr (pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back (token);
  }

  res.push_back (s.substr (pos_start));
  return res;
}

class KnownBits{
  llvm::APInt knownZeros, knownOnes;
public:
  KnownBits(size_t width=0){
    knownZeros=APInt(width,0);
    knownOnes=knownZeros;
  }

  KnownBits(APInt concreteVal){
    knownOnes=concreteVal;
    knownZeros=~knownOnes;
  }
  KnownBits(APInt zeros, APInt ones):knownZeros(zeros), knownOnes(ones){
    assert(zeros.getBitWidth() == ones.getBitWidth() && "parameter should have the same bit width");

  }

  unsigned getBitWidth()const{
    return knownZeros.getBitWidth();
  }

  llvm::APInt getKnownZeros()const{
    return knownZeros;
  }

  llvm::APInt getKnownOnes()const{
    return knownOnes;
  }



  std::string toString()const{
    llvm::SmallString<64>  ones,zeros;
    knownZeros.toString(zeros,2,false);
    knownOnes.toString(ones,2,false);
    return std::to_string(getBitWidth())+"|"+zeros.str().str()+"|"+ones.str().str();
  }

  static KnownBits fromString(std::string&& str){
    llvm::SmallVector<std::string> vec= split(str,"|");
    assert(vec.size()==3 && "expecting parsed string contains two delimiters");
    size_t width=(size_t)stoi(vec[0]);
    return KnownBits(APInt(width,vec[1],2),
                     APInt(width,vec[2],2));
  }

  bool hasConflict()const{
    return !(knownOnes&knownZeros).isZero();
  }
};

struct ConstantKnownBitsPattern
    : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = constantOp.getContext();

    auto value = llvm::dyn_cast<IntegerAttr>(constantOp.getValueAttr());
    if (!value)
        return success();
    if (!llvm::isa<IntegerType>(value.getType()))
        return success();

    if (constantOp->getAttr("analysis"))
        return success();

    IntegerType type=llvm::dyn_cast<IntegerType>(value.getType());
    APInt intVal=value.getValue();

    KnownBits bits(intVal);
    auto analysisAttr = StringAttr::get(ctx, bits.toString());

    rewriter.startRootUpdate(constantOp);
    constantOp->setAttr("analysis", analysisAttr);
    rewriter.finalizeRootUpdate(constantOp);
    return success();
  }
};

auto ANALYSIS_ATTR_NAME = "analysis";

StringAttr getAnalysis(Value val) {
    if (auto opRes = val.dyn_cast<OpResult>()) {
        auto owner = opRes.getOwner();
        if (auto analysis = owner->getAttr(ANALYSIS_ATTR_NAME)) {
            if (auto analysisStr = analysis.dyn_cast<StringAttr>())
            return analysisStr;
        }
    }

    assert(llvm::isa<IntegerType>(val.getType())&&"valueshould be an integer type");
    IntegerType type=llvm::dyn_cast<IntegerType>(val.getType());
    size_t width=type.getWidth();
    KnownBits bits(width);
    return StringAttr::get(val.getContext(), bits.toString());
}

StringAttr join(StringAttr lhs, StringAttr rhs) {
    std::string analysis;
    auto lhsVal = lhs.getValue();
    auto rhsVal = rhs.getValue();
    for (int i = 0; i < 32; i++) {
        char c = '?';
        if (lhsVal[i] == rhsVal[i]) {
            c = lhsVal[i];
        }
        analysis += c;
    }
    return StringAttr::get(lhs.getContext(), analysis);
}

struct OrKnownBitsPattern
    : public OpRewritePattern<arith::OrIOp> {
  using OpRewritePattern<arith::OrIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::OrIOp OrOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = OrOp.getContext();
    
    if (OrOp.getType() != IntegerType::get(ctx, 32))
        return success();

    auto analysisLhs = getAnalysis(OrOp.getLhs()).getValue();
    auto analysisRhs = getAnalysis(OrOp.getRhs()).getValue();

    KnownBits lBits=KnownBits::fromString(analysisLhs.str());
    KnownBits rBits=KnownBits::fromString(analysisRhs.str());
    assert(!lBits.hasConflict() &&"lhs contains conflict");
    assert(!rBits.hasConflict() && "rhs contains conflict");
    APInt ones, zeros;
    ones=lBits.getKnownOnes() | rBits.getKnownOnes();
    zeros=lBits.getKnownZeros() & rBits.getKnownZeros();

    KnownBits result(zeros,ones);
    assert(!result.hasConflict() && "result should not contain conflict");

    auto analysisAttr = StringAttr::get(ctx, result.toString());

    if (OrOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
        return success();

    rewriter.startRootUpdate(OrOp);
    OrOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(OrOp);
    return success();
  }
};
} // namespace


namespace {
struct KnownBitsAnalysisPass
    : public impl::KnownBitsAnalysisBase<
          KnownBitsAnalysisPass> {
  void runOnOperation() override;
};
} // namespace

void KnownBitsAnalysisPass::runOnOperation() {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConstantKnownBitsPattern, OrKnownBitsPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));

}

std::unique_ptr<Pass> mlir::createKnownBitsAnalysisPass() {
  return std::make_unique<KnownBitsAnalysisPass>();
}