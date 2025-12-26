/*
 * @file AISToAsync.cpp
 * @brief Placeholder for AIS → Async lowering.
 *
 * The long-term plan is to convert AIS programs into the MLIR async dialect
 * so the runtime scheduler can directly interpret parallel regions. While
 * that lowering is being implemented, we keep this pass as a structural
 * verification step that leaves the IR unchanged. It still participates in
 * the pipeline (so callers can require the lowering stage) without
 * accidentally stripping the functions that artifact emission depends on.
 */

#include "ais/Dialect/AIS/Transforms/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::ais {
#define GEN_PASS_DEF_AISTOASYNCPASS
#include "ais/Dialect/AIS/Transforms/Passes.h.inc"

namespace {

struct AISToAsyncPass : impl::AISToAsyncPassBase<AISToAsyncPass> {
  void runOnOperation() override {
    // TODO: introduce real lowering to async dialect.
    // For now we simply leave the module untouched.
    (void)getOperation();
  }
};

} // namespace

std::unique_ptr<Pass> createAISToAsyncPass() {
  return std::make_unique<AISToAsyncPass>();
}

} // namespace mlir::ais
