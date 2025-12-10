//===- AISOps.h -----------------------------------------------------------===//
//
// Part of the A-PXM project, under the Apache License v2.0.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef APXM_AIS_OPS_H
#define APXM_AIS_OPS_H

#include "apxm/AIS/IR/AISTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"
#include "apxm/AIS/Interfaces/RuntimeInterfaces.h.inc"

namespace mlir::ais {

struct BeliefResource : public SideEffects::Resource::Base<BeliefResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BeliefResource)
  StringRef getName() final { return "Beliefs"; }
};

struct GoalResource : public SideEffects::Resource::Base<GoalResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GoalResource)
  StringRef getName() final { return "Goals"; }
};

struct CapabilityResource
    : public SideEffects::Resource::Base<CapabilityResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CapabilityResource)
  StringRef getName() final { return "Capabilities"; }
};

struct EpisodicResource
    : public SideEffects::Resource::Base<EpisodicResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EpisodicResource)
  StringRef getName() final { return "Episodic"; }
};

} // namespace mlir::ais

#define GET_OP_CLASSES
#include "apxm/AIS/IR/AISOps.h.inc"
#undef GET_OP_CLASSES

#endif // APXM_AIS_OPS_H
