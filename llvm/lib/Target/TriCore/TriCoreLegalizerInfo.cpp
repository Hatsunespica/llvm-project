//===-- TriCoreLegalizerInfo.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for
/// TriCore.
/// \todo This should be generated by TableGen.
//===----------------------------------------------------------------------===//

#include "TriCoreLegalizerInfo.h"
#include "TriCoreSubtarget.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

static LegalityPredicate isMisalignedMemAccess() {
  return [=](const LegalityQuery &Query) -> bool {
    const unsigned MemSize = Query.MMODescrs[0].SizeInBits;
    const unsigned Align = Query.MMODescrs[0].AlignInBits;

    // Pointers require word alignment
    // TODO: remove this once we support half-word aligned pointers
    if (Query.Types[0].isPointer())
      return Align < 32;

    // Access is misaligned if the access has no natural alignment, up to
    // half-word alignment
    return Align < 16 && MemSize > Align;
  };
}

static LegalityPredicate isTruncStore() {
  return [=](const LegalityQuery &Query) {
    // Truncating store if type size is bigger than memory size
    return Query.Types[0].getSizeInBits() > Query.MMODescrs[0].SizeInBits;
  };
}

static LegalizeMutation narrowToAlignedMemAccess() {
  return [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
    const unsigned Align = Query.MMODescrs[0].AlignInBits;
    return std::make_pair(0, LLT::scalar(Align)); // <TyIdx, NewTy>
  };
}

static LegalizeMutation narrowToMemSize() {
  return [=](const LegalityQuery &Query) {
    // Use the memory size as size for the new type
    const unsigned MemSize = Query.MMODescrs[0].SizeInBits;
    return std::make_pair(0, LLT::scalar(MemSize));
  };
}

TriCoreLegalizerInfo::TriCoreLegalizerInfo(const TriCoreSubtarget &ST) {
  using namespace TargetOpcode;
  const LLT p0 = LLT::pointer(0, 32);
  const LLT s1 = LLT::scalar(1);
  const LLT s8 = LLT::scalar(8);
  const LLT s16 = LLT::scalar(16);
  const LLT s32 = LLT::scalar(32);
  const LLT s64 = LLT::scalar(64);

  // at least one G_IMPLICIT_DEF must be legal. we allow all types
  getActionDefinitionsBuilder(G_IMPLICIT_DEF)
      .legalFor({p0, s1, s8, s16, s32, s64})
      .clampScalar(0, s8, s64)
      .widenScalarToNextPow2(0);

  // G_PHI should be legal for all consumed types to avoid unnecessary
  // truncations and extensions
  getActionDefinitionsBuilder(G_PHI)
      .legalFor({p0, s8, s16, s32, s64})
      .clampScalar(0, s8, s64)
      .widenScalarToNextPow2(0);

  // Pointers

  // G_GLOBAL_VALUE and G_FRAME_INDEX are only valid for pointers
  getActionDefinitionsBuilder({G_FRAME_INDEX, G_GLOBAL_VALUE}).legalFor({p0});

  // G_INTTOPTR requires the scalar to have the same number of bits as the
  // pointer. It is not legal to narrow/widen the scalar as the extended/lost
  // bits change the address.
  getActionDefinitionsBuilder(G_INTTOPTR).legalFor({{p0, s32}});

  // G_PTRTOINT goes from a 32-bit pointer to a 32-bit scalar.
  getActionDefinitionsBuilder(G_PTRTOINT)
      .legalFor({{s32, p0}})
      .clampScalar(0, s32, s32);

  // Constants

  // G_CONSTANT is only legal for types that match our register size
  getActionDefinitionsBuilder(G_CONSTANT)
      .legalFor({p0, s32, s64})
      .clampScalar(0, s32, s64)
      .widenScalarToNextPow2(0);

  // Binary Ops

  // Simple binary operators are only legal for s32 types.
  getActionDefinitionsBuilder(
      {G_ADD, G_SUB, G_AND, G_OR, G_XOR, G_MUL, G_UMULH})
      .legalFor({s32})
      .clampScalar(0, s32, s32);

  // G_SDIV, G_UDIV is only legal for 32 bit types and has to be lowered for
  // 64-bit type
  getActionDefinitionsBuilder({G_SDIV, G_UDIV})
      .legalFor({s32})
      .clampScalar(0, s32, s64)
      .widenScalarToNextPow2(0)
      .libcallFor({s64});

  // G_SREM, G_UREM is only legal for 32 bit types and has to be lowered for
  // 64-bit type
  getActionDefinitionsBuilder({G_SREM, G_UREM})
      .legalFor({s32})
      .clampScalar(0, s32, s64)
      .widenScalarToNextPow2(0)
      .lowerFor({s64});

  // G_PTR_ADD must take a p0 and s32 operand
  getActionDefinitionsBuilder(G_PTR_ADD)
      .legalFor({{p0, s32}})
      .clampScalar(1, s32, s32);

  // Floating point Ops

  // G_FPEXT needs to convert from half to single, single to double and half to
  // double precision
  auto &FPExtActions = getActionDefinitionsBuilder(G_FPEXT)
                           .legalFor({{s32, s16}})
                           .lowerFor({{s64, s16}}); // FIXME: implement lower

  if (ST.hasTC18Ops())
    FPExtActions.legalFor({{s64, s32}});
  else
    FPExtActions.libcallFor({{s64, s32}});

  // G_FPTRUNC needs to convert from double to single, single to half and double
  // to single precision
  auto &FPTruncActions = getActionDefinitionsBuilder(G_FPTRUNC)
                             .legalFor({{s16, s32}})
                             .lowerFor({{s16, s64}}); // FIXME: implement lower

  if (ST.hasTC18Ops())
    FPTruncActions.legalFor({{s32, s64}});
  else
    FPTruncActions.libcallFor({{s32, s64}});

  // Overflow Ops

  // All variants of add/sub /w carry must produce an s32 result and an s1 carry
  getActionDefinitionsBuilder({G_UADDE, G_USUBE, G_UADDO, G_USUBO})
      .legalFor({{s32, s1}});

  // Shifts

  // G_SHL, G_LSHR and G_ASHR always produce the same type as their src type
  // (type idx 0, 32-bit). Additionally, only 32-bit shift amounts (type idx 1)
  // are allowed.
  getActionDefinitionsBuilder({G_SHL, G_LSHR, G_ASHR})
      .legalFor({{s32, s32}})
      .clampScalar(1, s32, s32)
      .clampScalar(0, s32, s32);

  // Comparisons & Select

  // G_ICMP is only legal for scalar 32-bit and pointer types. Result is s32.
  getActionDefinitionsBuilder(G_ICMP)
      .legalFor({{s32, s32}, {s32, p0}})
      .clampScalar(1, s32, s32)
      .clampScalar(0, s32, s32);

  // G_FCMP is always legal for s32, but depending on the subtarget feature
  // needs a libcall for s64
  auto &FPCMPActions = getActionDefinitionsBuilder(G_FCMP)
                           .legalFor({{s32, s32}})
                           .clampScalar(1, s32, s64)
                           .widenScalarToNextPow2(1)
                           .clampScalar(0, s32, s32);

  if (ST.hasTC18Ops()) {
    FPCMPActions.legalFor({s32, s64});
  } else {
    setFCmpLibcalls();
    FPCMPActions.customFor({s32, s64});
  }

  // G_SELECT is only valid for 32-bit and pointer types. Condition is s32.
  getActionDefinitionsBuilder(G_SELECT)
      .legalFor({{s32, s32}, {p0, s32}})
      .clampScalar(0, s32, s32)
      .clampScalar(1, s32, s32);

  // Extensions

  // G_{ANY,S,Z}EXT must be legal for all input types produced by at least one
  // legal instruction and all larger output types consumed by at least one
  // legal instruction
  getActionDefinitionsBuilder({G_ANYEXT, G_SEXT, G_ZEXT})
      .legalIf([=](const LegalityQuery &Query) {
        // Extensions are legal if the destination type fits in a register
        // and is a power of 2
        unsigned DstSize = Query.Types[0].getSizeInBits();
        return DstSize == 32 || DstSize == 64;
      })
      // Widen is currently not supported for G_*EXT. The artifact combiner of
      // the legalizer will create G_SEXT_INREG which can be widened.
      .clampScalar(0, s32, s64);

  // G_TRUNC is always legal as we can handle code-gen implications on the
  // extension side. Also this helps us to avoid certain code-duplications
  getActionDefinitionsBuilder(G_TRUNC).alwaysLegal();

  // G_SEXT_INREG is legal if it fits our registers
  getActionDefinitionsBuilder(G_SEXT_INREG)
      .legalForTypeWithAnyImm({s32, s64})
      .clampScalar(0, s32, s64)
      .widenScalarToNextPow2(0);

  // Floating Point Unary Ops

  // G_FPTOSI and G_FPTOUI are legal for s32 and s64 combinations, except for
  // {s64, s32}
  auto &FPConvActions = getActionDefinitionsBuilder({G_FPTOSI, G_FPTOUI})
                            .legalFor({{s32, s32}})
                            .clampScalar(0, s32, s64)
                            .widenScalarToNextPow2(0)
                            .clampScalar(1, s32, s64)
                            .widenScalarToNextPow2(1);

  if (ST.hasTC18Ops())
    FPConvActions.legalFor({{s32, s64}, {s64, s64}});
  else
    FPConvActions.libcallFor({{s32, s64}, {s64, s64}});

  // G_SITOFP and G_UITOFP are legal for s32 and s64 combinations, except for
  // {s32, s64}
  auto &IntToFPConvActions = getActionDefinitionsBuilder({G_SITOFP, G_UITOFP})
                                 .legalFor({{s32, s32}})
                                 .clampScalar(0, s32, s64)
                                 .widenScalarToNextPow2(0)
                                 .clampScalar(1, s32, s64)
                                 .widenScalarToNextPow2(1);

  if (ST.hasTC18Ops())
    IntToFPConvActions.legalFor({{s64, s32}, {s64, s64}});
  else
    IntToFPConvActions.libcallFor({{s64, s32}, {s64, s64}});

  // G_FABS legal for s32 and s64
  getActionDefinitionsBuilder(G_FABS)
      .legalFor({s32, s64})
      .clampScalar(0, s32, s64)
      .widenScalarToNextPow2(0);
      
  // Floating Point Binary ops.

  // Floating Point arithmetic instructions are legal for s32 for all target. 
  // Also legal for s64 if the target supports tc18 instructions, otherwise 
  // library call needed.
  auto &FPArithmActions = getActionDefinitionsBuilder({G_FADD, G_FSUB, G_FMUL, G_FDIV})
                              .legalFor({s32})
                              .clampScalar(0, s32, s64)
                              .widenScalarToNextPow2(0);

  if (ST.hasTC18Ops())
    FPArithmActions.legalFor({s64});
  else
    FPArithmActions.libcallFor({s64});

  // Load & Store

  // G_LOAD is legal for 32 and 64-bit scalar and pointer types.
  // Memory size must be a power of 2.
  getActionDefinitionsBuilder(G_LOAD)
      .legalForTypesWithMemDesc({
          // Load/store uses natural alignment up to half-word alignment
          // Pointers require word alignment
          // TODO: p0 load/store can use half-word alignment if they are put on
          //  the data regbank.
          {s32, p0, 8, 8},
          {s32, p0, 16, 16},
          {s32, p0, 32, 16},
          {s64, p0, 64, 16},
          {p0, p0, 32, 32},
      })
      // Unaligned loads must be broken up into aligned loads
      .narrowScalarIf(isMisalignedMemAccess(), narrowToAlignedMemAccess())
      // Non-power-of-2 loads need to be broken up
      .lowerIfMemSizeNotPow2()
      // Result must fit in a register
      .clampScalar(0, s32, s64)
      // Lower any extending loads left into G_ANYEXT and G_LOAD
      .lowerIf([=](const LegalityQuery &Query) {
        return Query.Types[0].getSizeInBits() != Query.MMODescrs[0].SizeInBits;
      })
      // Eliminate left-over non-pow-2 results
      .widenScalarToNextPow2(0);

  // G_STORE is legal for pointers and scalars if the store size is equal to the
  // scalar type size. Different to G_LOAD, we require explicit s8 and s16
  // value types, because this allows to match every possible store with
  // TableGen instead of having to fall back to C++ for truncating stores.
  getActionDefinitionsBuilder(G_STORE)
      .legalForTypesWithMemDesc({
          // Load/store uses natural alignment up to half-word alignment
          // Pointers require word alignment
          // TODO: p0 load/store can use half-word alignment if they are put on
          //  the data regbank.
          {s8, p0, 8, 8},
          {s16, p0, 16, 16},
          {s32, p0, 32, 16},
          {s64, p0, 64, 16},
          {p0, p0, 32, 32},
      })
      // Unaligned stores must be broken up into aligned stores
      .narrowScalarIf(isMisalignedMemAccess(), narrowToAlignedMemAccess())
      // Lower truncating stores into G_TRUNC and G_STORE
      .narrowScalarIf(isTruncStore(), narrowToMemSize())
      // Non-power-of-2 stores need to be broken up
      .lowerIfMemSizeNotPow2()
      // Extend / truncate the value to a power-of-2 between s8 and s64
      .clampScalar(0, s8, s64)
      .widenScalarToNextPow2(0);

  // G_SEXTLOAD and G_ZEXTLOAD are legal for a 32-bit result type
  getActionDefinitionsBuilder({G_SEXTLOAD, G_ZEXTLOAD})
      .legalForTypesWithMemDesc({
          {s32, p0, 8, 8},
          {s32, p0, 16, 16},
          {s32, p0, 32, 32},
      })
      .clampScalar(0, s32, s32)
      .lowerIfMemSizeNotPow2()
      // Lower anything left over to G_*EXT and G_LOAD
      .lower();

  // G_MERGE_VALUES and G_UNMERGE_VALUES are legalizer artifacts. Therefore
  // their types correspond to any legal types that are produced or consumed by
  // our instructions. This means that we require the smaller type to be a power
  // of 2 between s8 and s32 and the bigger type to be a power of 2 between s8
  // and s64. Furthermore the bigger type must be a multiple of the smaller type
  for (unsigned OpCode : {G_MERGE_VALUES, G_UNMERGE_VALUES}) {
    const unsigned BigTyIdx = OpCode == G_MERGE_VALUES ? 0 : 1;
    const unsigned SmallTyIdx = OpCode == G_MERGE_VALUES ? 1 : 0;

    getActionDefinitionsBuilder(OpCode)
        // Clamp SmallTy to s8-s32 and make it a power of 2
        .clampScalar(SmallTyIdx, s8, s32)
        .widenScalarToNextPow2(SmallTyIdx)
        // Clamp BigTy to s8-s64 and make it a power of 2
        .clampScalar(BigTyIdx, s8, s64)
        .widenScalarToNextPow2(BigTyIdx)
        // At this point we only have power of 2 types between s8 and s64
        .legalIf([=](const LegalityQuery &Query) {
          const LLT &BigTy = Query.Types[BigTyIdx];
          const LLT &SmallTy = Query.Types[SmallTyIdx];
          const unsigned BigSize = BigTy.getSizeInBits();
          const unsigned SmallSize = SmallTy.getSizeInBits();

          return BigSize % SmallSize == 0;
        });
  }

  // Branches

  // G_BRCOND is valid for s1 and s32 scalars.
  getActionDefinitionsBuilder(G_BRCOND).legalFor({s1, s32});

  // G_BRINDIRECT is valid for p0 types.
  getActionDefinitionsBuilder(G_BRINDIRECT).legalFor({p0});

  computeTables();
  verify(*ST.getInstrInfo());
}

void TriCoreLegalizerInfo::setFCmpLibcalls() {
  // FCMP_TRUE and FCMP_FALSE don't need libcalls, they should be
  // default-initialized.
  FCmp64Libcalls.resize(CmpInst::LAST_FCMP_PREDICATE + 1);
  FCmp64Libcalls[CmpInst::FCMP_OEQ] = {{RTLIB::OEQ_F64, CmpInst::ICMP_EQ}};
  FCmp64Libcalls[CmpInst::FCMP_OGE] = {{RTLIB::OGE_F64, CmpInst::ICMP_SGE}};
  FCmp64Libcalls[CmpInst::FCMP_OGT] = {{RTLIB::OGT_F64, CmpInst::ICMP_SGT}};
  FCmp64Libcalls[CmpInst::FCMP_OLE] = {{RTLIB::OLE_F64, CmpInst::ICMP_SLE}};
  FCmp64Libcalls[CmpInst::FCMP_OLT] = {{RTLIB::OLT_F64, CmpInst::ICMP_SLT}};
  FCmp64Libcalls[CmpInst::FCMP_ORD] = {{RTLIB::UO_F64, CmpInst::ICMP_EQ}};
  FCmp64Libcalls[CmpInst::FCMP_UGE] = {{RTLIB::OLT_F64, CmpInst::ICMP_SGE}};
  FCmp64Libcalls[CmpInst::FCMP_UGT] = {{RTLIB::OLE_F64, CmpInst::ICMP_SGT}};
  FCmp64Libcalls[CmpInst::FCMP_ULE] = {{RTLIB::OGT_F64, CmpInst::ICMP_SLE}};
  FCmp64Libcalls[CmpInst::FCMP_ULT] = {{RTLIB::OGE_F64, CmpInst::ICMP_SLT}};
  FCmp64Libcalls[CmpInst::FCMP_UNE] = {{RTLIB::UNE_F64, CmpInst::ICMP_NE}};
  FCmp64Libcalls[CmpInst::FCMP_UNO] = {{RTLIB::UO_F64, CmpInst::ICMP_NE}};
  FCmp64Libcalls[CmpInst::FCMP_ONE] = {{RTLIB::OGT_F64, CmpInst::ICMP_SGT},
                                       {RTLIB::OLT_F64, CmpInst::ICMP_SLT}};
  FCmp64Libcalls[CmpInst::FCMP_UEQ] = {{RTLIB::OEQ_F64, CmpInst::ICMP_EQ},
                                       {RTLIB::UO_F64, CmpInst::ICMP_NE}};
}

bool TriCoreLegalizerInfo::legalizeCustom(MachineInstr &MI,
                                          MachineRegisterInfo &MRI,
                                          MachineIRBuilder &MIRBuilder,
                                          GISelChangeObserver &Observer) const {
  switch (MI.getOpcode()) {
  default:
    // No idea what to do.
    return false;
  case TargetOpcode::G_FCMP:
    return legalizeFCmp(MI, MRI, MIRBuilder);
  }
  llvm_unreachable("expected switch to return");
}

bool TriCoreLegalizerInfo::legalizeIntrinsic(
    MachineInstr &MI, MachineIRBuilder &MIRBuilder,
    GISelChangeObserver &Observer) const {
  switch (MI.getIntrinsicID()) {
  case Intrinsic::memcpy:
  case Intrinsic::memset:
  case Intrinsic::memmove:
    if (createMemLibcall(MIRBuilder, *MIRBuilder.getMRI(), MI) ==
        LegalizerHelper::UnableToLegalize)
      return false;
    MI.eraseFromParent();
    return true;
  default:
    break;
  }
  return true;
}

bool TriCoreLegalizerInfo::legalizeFCmp(MachineInstr &MI,
                                        MachineRegisterInfo &MRI,
                                        MachineIRBuilder &MIRBuilder) const {
  Register Src1Reg = MI.getOperand(2).getReg();
  Register Src2Reg = MI.getOperand(3).getReg();

  assert(MRI.getType(Src1Reg) == MRI.getType(Src2Reg) &&
         MRI.getType(Src1Reg) == LLT::scalar(64) &&
         "Expected double float types");

  MIRBuilder.setInstr(MI);
  LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();

  auto OriginalResult = MI.getOperand(0).getReg();
  auto Predicate =
      static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());

  if (Predicate == FCmpInst::FCMP_TRUE || Predicate == FCmpInst::FCMP_FALSE) {
    // True and false predicates always return a constant
    MIRBuilder.buildConstant(OriginalResult,
                             Predicate == FCmpInst::FCMP_TRUE ? 1 : 0);
    MI.eraseFromParent();
    return true;
  }

  // Get the necessary libary calls for the given predicate. Double lib calls
  // always have the signature (double,double) -> int
  auto Libcalls = FCmp64Libcalls[Predicate];
  auto *RetTy = Type::getInt32Ty(Ctx);
  auto *ArgTy = Type::getDoubleTy(Ctx);

  SmallVector<Register, 2> Results;
  for (auto Libcall : Libcalls) {
    auto LibcallResult = MRI.createGenericVirtualRegister(LLT::scalar(32));
    auto Status =
        createLibcall(MIRBuilder, Libcall.LibcallID, {LibcallResult, RetTy},
                      {{Src1Reg, ArgTy}, {Src2Reg, ArgTy}});

    if (Status != LegalizerHelper::Legalized)
      return false;

    auto ProcessedResult =
        Libcalls.size() == 1
            ? OriginalResult
            : MRI.createGenericVirtualRegister(MRI.getType(OriginalResult));

    // We have a result, but we need to transform it into a proper 1-bit 0 or
    // 1, taking into account the different peculiarities of the values
    // returned by the comparison functions.
    CmpInst::Predicate ResultPred = Libcall.Predicate;
    if (ResultPred == CmpInst::BAD_ICMP_PREDICATE) {
      // We have a nice 0 or 1, and we just need to truncate it back to 1 bit
      // to keep the types consistent.
      MIRBuilder.buildTrunc(ProcessedResult, LibcallResult);
    } else {
      // We need to compare against 0.
      assert(CmpInst::isIntPredicate(ResultPred) && "Unsupported predicate");
      auto Zero = MRI.createGenericVirtualRegister(LLT::scalar(32));
      MIRBuilder.buildConstant(Zero, 0);
      MIRBuilder.buildICmp(ResultPred, ProcessedResult, LibcallResult, Zero);
    }
    Results.push_back(ProcessedResult);
  }

  if (Results.size() != 1) {
    assert(Results.size() == 2 && "Unexpected number of results");
    MIRBuilder.buildOr(OriginalResult, Results[0], Results[1]);
  }

  MI.eraseFromParent();
  return true;
}
