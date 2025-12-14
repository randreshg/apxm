/**
 * @file  Passes.h
 * @brief Public entry point for all AIS transformation passes.
 *
 * The file re-exports the auto-generated pass declarations (`GEN_PASS_DECL`)
 * and provides factory functions for each pass.  That keeps the registration
 * site (AisTransforms.cpp) and command-line driver code free of TableGen
 * details.
 *
 * Pass list (in canonical order):
 *   1. normalize         – canonicalise the graph
 *   2. scheduling        – annotate with tier/cost/parallel-safe flags
 *   3. fuse-reasoning    – batch LLM calls (highest ROI)
 *   4. lower-to-async    – expose parallelism to the runtime
 *   5. ais-emit-rust     – ahead-of-time Rust code generation
 */

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
