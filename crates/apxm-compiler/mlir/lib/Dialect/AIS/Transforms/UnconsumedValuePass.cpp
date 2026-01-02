/**
 * @file  UnconsumedValuePass.cpp
 * @brief Dead Code Elimination (DCE) warning pass for AIS dialect.
 *
 * This pass scans all operations and emits warnings when operation results
 * are not consumed by any other operation. This helps developers identify:
 *
 * - Forgotten variable bindings (e.g., `rsn "query" -> unused_result`)
 * - Missing return statements in flows
 * - Logic errors where data flows are incomplete
 *
 * Operations with side effects (memory writes, invocations, communication)
 * are exempt from this warning as they have effects beyond their return value.
 */

#include "ais/Dialect/AIS/Transforms/Passes.h"

#include "ais/Dialect/AIS/IR/AISOps.h"
#include "ais/Dialect/AIS/Support/AISDebug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::ais {
#define GEN_PASS_DEF_UNCONSUMEDVALUEWARNING
#include "ais/Dialect/AIS/Transforms/Passes.h.inc"

namespace {

APXM_AIS_DEBUG_SETUP(dce_warning)

/// Checks if a value is used as an operand in any operation within the function.
/// This is a fallback check when use_empty() incorrectly returns true due to
/// MLIR builder issues with variadic operands.
static bool isUsedInAnyOperation(Value value, func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (operand == value) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

/// Returns true if the operation has side effects that make its execution
/// valuable even if the result is unused.
static bool hasSideEffects(Operation *op) {
  // Memory operations have side effects (write to memory)
  if (isa<UMemOp>(op))
    return true;

  // Invocation calls external capabilities (side effects)
  if (isa<InvOp>(op))
    return true;

  // Communication sends messages (side effects)
  if (isa<CommunicateOp>(op))
    return true;

  // Return transfers control flow (not a value producer warning candidate)
  if (isa<ReturnOp, func::ReturnOp>(op))
    return true;

  // Error operations are side-effectful (abort/signal)
  if (isa<ErrOp>(op))
    return true;

  // Fence operations are pure synchronization (no result to consume)
  if (isa<FenceOp>(op))
    return true;

  // Jump and branch operations are control flow
  if (isa<JumpOp, BranchOnValueOp, SwitchOp>(op))
    return true;

  // Loop operations are control flow structures
  if (isa<LoopStartOp, LoopEndOp>(op))
    return true;

  // TryCatch is control flow
  if (isa<TryCatchOp>(op))
    return true;

  return false;
}

/// Returns a human-readable name for the operation for warning messages.
static StringRef getOpDisplayName(Operation *op) {
  if (isa<AskOp>(op))
    return "ask";
  if (isa<ThinkOp>(op))
    return "think";
  if (isa<ReasonOp>(op))
    return "reason";
  if (isa<QMemOp>(op))
    return "memory query";
  if (isa<PlanOp>(op))
    return "plan";
  if (isa<ReflectOp>(op))
    return "reflection";
  if (isa<VerifyOp>(op))
    return "verification";
  if (isa<MergeOp>(op))
    return "merge";
  if (isa<WaitAllOp>(op))
    return "wait_all";
  if (isa<ExcOp>(op))
    return "exception";
  if (isa<ConstStrOp>(op))
    return "constant string";
  if (isa<FlowCallOp>(op))
    return "flow call";
  return op->getName().getStringRef();
}

struct UnconsumedValueWarningPass
    : impl::UnconsumedValueWarningBase<UnconsumedValueWarningPass> {
  void runOnOperation() override {
    APXM_AIS_DEBUG_HEADER(UnconsumedValueWarning);
    ModuleOp module = getOperation();

    struct Statistics {
      uint64_t scanned = 0;
      uint64_t warnings = 0;
    } stats;

    module.walk([&](func::FuncOp func) {
      // Skip external function declarations
      if (func.isExternal())
        return WalkResult::advance();

      func.walk([&](Operation *op) {
        stats.scanned++;

        // Skip operations with side effects
        if (hasSideEffects(op)) {
          APXM_AIS_DEBUG("  Skip (side-effect): " << op->getName());
          return WalkResult::advance();
        }

        // Check each result of the operation
        for (OpResult result : op->getResults()) {
          if (result.use_empty()) {
            // Fallback check: scan all operations for this value as an operand.
            // This handles cases where MLIR's use-def chain wasn't properly
            // updated (e.g., variadic operands in ops with regions).
            if (isUsedInAnyOperation(result, func)) {
              APXM_AIS_DEBUG("  Skip (found use via scan): " << op->getName());
              continue;
            }

            // Emit warning for unconsumed value
            op->emitWarning()
                << "result of " << getOpDisplayName(op)
                << " operation is not consumed; consider binding with '-> name' "
                   "or removing if unused";
            stats.warnings++;
            APXM_AIS_DEBUG("  Warning: unconsumed result from "
                           << op->getName());
          }
        }

        return WalkResult::advance();
      });

      return WalkResult::advance();
    });

    APXM_AIS_INFO("Scanned " << stats.scanned << " ops, emitted "
                             << stats.warnings << " warnings");
    APXM_AIS_DEBUG_FOOTER(UnconsumedValueWarning);
  }
};

} // namespace

std::unique_ptr<Pass> createUnconsumedValueWarningPass() {
  return std::make_unique<UnconsumedValueWarningPass>();
}

} // namespace mlir::ais
