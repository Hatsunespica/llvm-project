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

  /*
   * Return 1 if i-th bit is 1, 0 if it is 0. -1 for unknown
   */
  int getIthBit(size_t i)const{
    assert(i<getBitWidth() && "required position out of range");
    bool zero=knownZeros.isOneBitSet(i), one=knownOnes.isOneBitSet(i);
    if( zero && one){
      return 1;
    }else if(zero | one){
      return 0;
    }
    return -1;
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
    assert(verifyResult(lBits,rBits,result) && "result verification failed");

    auto analysisAttr = StringAttr::get(ctx, result.toString());

    if (OrOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
        return success();

    rewriter.startRootUpdate(OrOp);
    OrOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(OrOp);
    return success();
  }

  static bool verifyResult(KnownBits lBits, KnownBits rBits, KnownBits result){
    /*
     * True value table for OR
     *     0  1  X
     *  0  0  1  X
     *  1  1  1  1
     *  X  X  1  X
     */
    for(size_t i=0;i<lBits.getBitWidth();++i){
        int L=lBits.getIthBit(i),R=rBits.getIthBit(i),res=result.getIthBit(i);
        if(L==1||R==1){
            if(res!=1){
              return false;
            }
        }else if(L==0&&R==0){
            if(res!=0){
              return false;
            }
        }else{
            if(res!=-1){
              return false;
            }
        }
    }
    return true;
  }
};

struct AndKnownBitsPattern
    : public OpRewritePattern<arith::AndIOp> {
  using OpRewritePattern<arith::AndIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AndIOp AndOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = AndOp.getContext();

    auto analysisLhs = getAnalysis(AndOp.getLhs()).getValue();
    auto analysisRhs = getAnalysis(AndOp.getRhs()).getValue();

    KnownBits lBits=KnownBits::fromString(analysisLhs.str());
    KnownBits rBits=KnownBits::fromString(analysisRhs.str());
    assert(!lBits.hasConflict() &&"lhs contains conflict");
    assert(!rBits.hasConflict() && "rhs contains conflict");
    APInt ones, zeros;
    ones=lBits.getKnownOnes() & rBits.getKnownOnes();
    zeros=lBits.getKnownZeros() | rBits.getKnownZeros();

    KnownBits result(zeros,ones);
    assert(!result.hasConflict() && "result should not contain conflict");
    assert(verifyResult(lBits,rBits,result) && "result verification failed");

    auto analysisAttr = StringAttr::get(ctx, result.toString());

    if (AndOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
        return success();

    rewriter.startRootUpdate(AndOp);
    AndOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(AndOp);
    return success();
  }

  static bool verifyResult(KnownBits lBits, KnownBits rBits, KnownBits result){
    /*
     * True value table for AND
     *     0  1  X
     *  0  0  0  0
     *  1  0  1  X
     *  X  0  X  X
     */
    for(size_t i=0;i<lBits.getBitWidth();++i){
        int L=lBits.getIthBit(i),R=rBits.getIthBit(i),res=result.getIthBit(i);
        if(L==0||R==0){
            if(res!=0){
              return false;
            }
        }else if(L==1&&R==1){
            if(res!=1){
              return false;
            }
        }else{
            if(res!=-1){
              return false;
            }
        }
    }
    return true;
  }
};

struct XOrKnownBitsPattern
    : public OpRewritePattern<arith::XOrIOp> {
  using OpRewritePattern<arith::XOrIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::XOrIOp XOrOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = XOrOp.getContext();

    auto analysisLhs = getAnalysis(XOrOp.getLhs()).getValue();
    auto analysisRhs = getAnalysis(XOrOp.getRhs()).getValue();

    KnownBits lBits=KnownBits::fromString(analysisLhs.str());
    KnownBits rBits=KnownBits::fromString(analysisRhs.str());
    assert(!lBits.hasConflict() &&"lhs contains conflict");
    assert(!rBits.hasConflict() && "rhs contains conflict");
    APInt ones, zeros;
    ones = (lBits.getKnownZeros() & rBits.getKnownOnes()) |
           (lBits.getKnownOnes() & rBits.getKnownZeros());
    zeros = (lBits.getKnownZeros() & rBits.getKnownZeros()) |
            (lBits.getKnownOnes() & rBits.getKnownOnes());


    KnownBits result(zeros,ones);
    assert(!result.hasConflict() && "result should not contain conflict");
    assert(verifyResult(lBits,rBits,result) && "result verification failed");

    auto analysisAttr = StringAttr::get(ctx, result.toString());

    if (XOrOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
        return success();

    rewriter.startRootUpdate(XOrOp);
    XOrOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(XOrOp);
    return success();
  }

  static bool verifyResult(KnownBits lBits, KnownBits rBits, KnownBits result){
    /*
     * True value table for XOR
     *     0  1  X
     *  0  0  1  X
     *  1  1  0  X
     *  X  X  X  X
     */
    for(size_t i=0;i<lBits.getBitWidth();++i){
        int L=lBits.getIthBit(i),R=rBits.getIthBit(i),res=result.getIthBit(i);
        if(L==-1||R==-1){
            if(res!=-1){
              return false;
            }
        }else if(L==R){
            if(res!=0){
              return false;
            }
        }else{
            if(res!=1){
              return false;
            }
        }
    }
    return true;
  }
};

struct ExtSIKnownBitsPattern
    : public OpRewritePattern<arith::ExtSIOp> {
  using OpRewritePattern<arith::ExtSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtSIOp ExtSIOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = ExtSIOp.getContext();


    auto analysisIn = getAnalysis(ExtSIOp.getIn()).getValue();
    IntegerType newIntType=llvm::dyn_cast<IntegerType>(ExtSIOp.getOut().getType());


    KnownBits inBits=KnownBits::fromString(analysisIn.str());

    assert(!inBits.hasConflict() &&"lhs contains conflict");
    APInt zeros=inBits.getKnownZeros(),ones=inBits.getKnownOnes();
    //get highest bits

    APInt newZeros=APInt::getZero(newIntType.getWidth()-zeros.getBitWidth());
    APInt newOnes=newZeros;
    if(zeros.isNegative()){
        newZeros.flipAllBits();
    }
    newZeros=newZeros.concat(zeros);

    if(ones.isNegative()){
        newOnes.flipAllBits();
    }
    newOnes=newOnes.concat(ones);

    KnownBits result(newZeros,newOnes);
    assert(!result.hasConflict() && "result should not contain conflict");
    assert(verifyResult(inBits, result) && "result verification failed");

    auto analysisAttr = StringAttr::get(ctx, result.toString());

    if (ExtSIOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
        return success();

    rewriter.startRootUpdate(ExtSIOp);
    ExtSIOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(ExtSIOp);
    return success();
  }

  static bool verifyResult(KnownBits inBits, KnownBits result){
    /*
     * Verify sign bits in the result
     */
    bool inZero=inBits.getKnownZeros().isNegative();
    bool inOne=inBits.getKnownOnes().isNegative();
    for(size_t i=inBits.getBitWidth();i<result.getBitWidth();++i){
      if(result.getKnownOnes().isOneBitSet(i)!=inOne ||
            result.getKnownZeros().isOneBitSet(i)!=inZero){
            return false;
      }
    }
    return true;
  }
};

struct ExtUIKnownBitsPattern
    : public OpRewritePattern<arith::ExtUIOp> {
  using OpRewritePattern<arith::ExtUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtUIOp ExtUIOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = ExtUIOp.getContext();


    auto analysisIn = getAnalysis(ExtUIOp.getIn()).getValue();
    IntegerType newIntType=llvm::dyn_cast<IntegerType>(ExtUIOp.getOut().getType());


    KnownBits inBits=KnownBits::fromString(analysisIn.str());

    assert(!inBits.hasConflict() &&"lhs contains conflict");
    APInt zeros=inBits.getKnownZeros(),ones=inBits.getKnownOnes();
    //get highest bits

    APInt newZeros=APInt::getAllOnes(newIntType.getWidth()-zeros.getBitWidth());
    APInt newOnes=newZeros;
    newZeros=newZeros.concat(zeros);
    newOnes.flipAllBits();
    newOnes=newOnes.concat(ones);

    KnownBits result(newZeros,newOnes);
    assert(!result.hasConflict() && "result should not contain conflict");
    assert(verifyResult(inBits, result) && "result verification failed");

    auto analysisAttr = StringAttr::get(ctx, result.toString());

    if (ExtUIOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
      return success();

    rewriter.startRootUpdate(ExtUIOp);
    ExtUIOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(ExtUIOp);
    return success();
  }

  static bool verifyResult(KnownBits inBits, KnownBits result){
    /*
     * Verify sign bits in the result
     */
    for(size_t i=inBits.getBitWidth();i<result.getBitWidth();++i){
      if(!result.getKnownOnes().isOneBitSet(i) &&
          result.getKnownZeros().isOneBitSet(i)){
            return false;
      }
    }
    return true;
  }
};

struct TruncIKnownBitsPattern
    : public OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern<arith::TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp TruncIOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = TruncIOp.getContext();


    auto analysisIn = getAnalysis(TruncIOp.getIn()).getValue();
    IntegerType newIntType=llvm::dyn_cast<IntegerType>(TruncIOp.getOut().getType());


    KnownBits inBits=KnownBits::fromString(analysisIn.str());

    assert(!inBits.hasConflict() &&"lhs contains conflict");
    APInt zeros=inBits.getKnownZeros(),ones=inBits.getKnownOnes();
    //get highest bits

    APInt newZeros=zeros.trunc(newIntType.getWidth()),
          newOnes=ones.trunc(newIntType.getWidth());

    KnownBits result(newZeros,newOnes);
    assert(!result.hasConflict() && "result should not contain conflict");
    assert(verifyResult(inBits, result) && "result verification failed");

    auto analysisAttr = StringAttr::get(ctx, result.toString());

    if (TruncIOp->getAttr(ANALYSIS_ATTR_NAME) == analysisAttr)
      return success();

    rewriter.startRootUpdate(TruncIOp);
    TruncIOp->setAttr(ANALYSIS_ATTR_NAME, analysisAttr);
    rewriter.finalizeRootUpdate(TruncIOp);
    return success();
  }

  static bool verifyResult(KnownBits inBits, KnownBits result){
    /*
     * Verify very bits in the result;
     */
    for(size_t i=0;i<result.getBitWidth();++i){
      if(result.getKnownZeros().isOneBitSet(i)!=inBits.getKnownZeros().isOneBitSet(i)){
            return false;
      }
      if(result.getKnownOnes().isOneBitSet(i)!=inBits.getKnownOnes().isOneBitSet(i)){
            return false;
      }
    }
    return true;
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
    patterns.add<ConstantKnownBitsPattern, OrKnownBitsPattern,
                 AndKnownBitsPattern,XOrKnownBitsPattern,ExtSIKnownBitsPattern,
                 ExtUIKnownBitsPattern,TruncIKnownBitsPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));

}

std::unique_ptr<Pass> mlir::createKnownBitsAnalysisPass() {
  return std::make_unique<KnownBitsAnalysisPass>();
}