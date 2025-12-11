//===- Passes.h - AIS Pass Declarations -----------------------*- C++ -*-===//
//
// Part of the A-PXM project, under the Apache License v2.0.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// Consolidated pass declarations aligned with ais-compiler MVP design.
//
// Philosophy: Keep it simple
// - 3 domain-specific transform passes
// - 1 lowering pass to async dialect
// - Use MLIR native passes (canonicalizer, CSE, symbol-dce)
//
//===----------------------------------------------------------------------===//

#ifndef APXM_AIS_PASSES_H
#define APXM_AIS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::ais {

//===----------------------------------------------------------------------===//
// Pass Declarations (generated from Passes.td)
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "apxm/Dialect/AIS/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Create NormalizeAgentGraph pass - canonicalize AIS graph structure
std::unique_ptr<Pass> createNormalizeAgentGraphPass();

/// Create CapabilityScheduling pass - annotate with scheduling metadata
std::unique_ptr<Pass> createCapabilitySchedulingPass();

/// Create FuseReasoning pass - merge reasoning chains (highest ROI)
std::unique_ptr<Pass> createFuseReasoningPass();

/// Create AISToAsync pass - lower to async dialect for runtime interpretation
std::unique_ptr<Pass> createAISToAsyncPass();

/// Create AISToRust pass - emit Rust source code from AIS dialect
std::unique_ptr<Pass> createAISToRustPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all AIS passes with the global registry.
/// This enables passes to be used via textual pass pipeline syntax.
void registerAISPasses();

}  // namespace mlir::ais

#endif  // APXM_AIS_PASSES_H
