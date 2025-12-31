/**
 * @file  Passes.td
 * @brief Transform and lowering passes that operate on the AIS dialect.
 *
 * Passes are split into two categories:
 *
 * 1. Domain-specific transforms
 *    - normalize, scheduling, fuse-reasoning – work purely on AIS ops
 * 2. Lowering passes
 *    - lower-to-async – convert to `async.execute` so the runtime can schedule
 *    - ais-emit-rust  – transpile to Rust source that uses apxm-runtime
 *
 * Each pass declares its constructor, dependent dialects, and tunable options
 * so that `mlir-opt --help` displays them.  Options are wired to the
 * generated pass class via `PassOptions` and can be set on the command line:
 *
 *   mlir-opt --pass-pipeline='builtin.module(ais-fuse-reasoning)' \
 *            --ais-parallel-threshold=5
 */

#ifndef APXM_AIS_PASSES
#define APXM_AIS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Domain-Specific Transform Passes
//===----------------------------------------------------------------------===//

def NormalizeAgentGraph : Pass<"normalize", "mlir::ModuleOp"> {
  let summary = "Normalize AIS graph structure";
  let description = [{
    Canonicalizes the AIS graph by:
    - Deduplicating reasoning contexts
    - Normalizing string attributes (lowercase capability/space names)
    - Establishing SSA ordering invariants for downstream passes

    This pass ensures the IR is in a canonical form that other passes
    can rely on, similar to MLIR's canonicalizer but domain-specific.
  }];
  let constructor = "mlir::ais::createNormalizeAgentGraphPass()";
}

def CapabilityScheduling : Pass<"scheduling", "mlir::ModuleOp"> {
  let summary = "Annotate operations with scheduling metadata";
  let description = [{
    Classifies capabilities into execution tiers and annotates operations
    with scheduling hints for the runtime:

    - Tier classification (io/compute/reasoning/memory/general)
    - Cost estimation based on context size
    - Parallel-safety markers for speculation

    These annotations guide the runtime dataflow scheduler to overlap work
    and optimize execution order.
  }];
  let constructor = "mlir::ais::createCapabilitySchedulingPass()";
  let options = [
    Option<"parallelThreshold", "parallel-threshold", "unsigned", /*default=*/"3",
           "Maximum context size considered safe for parallel execution">,
    Option<"baseCost", "base-cost", "unsigned", /*default=*/"2",
           "Base estimated cost for each operation">,
    Option<"contextWeight", "context-weight", "unsigned", /*default=*/"2",
           "Cost multiplier per context operand">
  ];
}

def FuseReasoning : Pass<"fuse-reasoning", "mlir::ModuleOp"> {
  let summary = "Fuse reasoning chains to reduce LLM calls";
  let description = [{
    Identifies producer-consumer ais.rsn chains and merges them into single
    batched operations. This is the highest-ROI optimization (100-400x) as it:

    - Reduces serialized LLM API calls (each ~500ms-2s)
    - Concatenates reasoning templates with separator
    - Combines contexts from both operations

    Only fuses when producer has single use.
  }];
  let constructor = "mlir::ais::createFuseReasoningPass()";
}

//===----------------------------------------------------------------------===//
// Analysis/Warning Passes
//===----------------------------------------------------------------------===//

def UnconsumedValueWarning : Pass<"unconsumed-value-warning", "mlir::ModuleOp"> {
  let summary = "Warn about unconsumed operation results";
  let description = [{
    Scans all operations and emits warnings when operation results are not
    consumed by any other operation. This helps developers identify:

    - Forgotten variable bindings (e.g., `rsn "query" -> unused_result`)
    - Missing return statements in flows
    - Logic errors where data flows are incomplete

    Operations with side effects (memory writes, invocations, communication)
    are exempt as they have effects beyond their return value.

    This pass is typically run after optimization passes but before artifact
    emission, as a diagnostic aid rather than a transformation.
  }];
  let constructor = "mlir::ais::createUnconsumedValueWarningPass()";
}

//===----------------------------------------------------------------------===//
// Conversion/Lowering Passes
//===----------------------------------------------------------------------===//

def AISToAsyncPass : Pass<"lower-to-async", "mlir::ModuleOp"> {
  let summary = "Lower AIS operations to async dialect";
  let description = [{
    Converts AIS operations (inv, rsn, qmem, umem, wait_all) to MLIR's
    async dialect by wrapping them in async.execute regions. This:

    - Makes parallelism explicit in the IR
    - Enables async-aware optimizations
    - Prepares for runtime interpretation

    Note: Runtime interprets async dialect directly without additional
    lowering layer (no AsyncToRuntime pass needed per ais-compiler design).
  }];
  let constructor = "mlir::ais::createAISToAsyncPass()";
  let dependentDialects = ["async::AsyncDialect", "func::FuncDialect"];
}

def AISToRustPass : Pass<"ais-emit-rust", "mlir::ModuleOp"> {
  let summary = "Emit Rust source code from AIS dialect";
  let description = [{
    Transpiles AIS operations to Rust source code that uses apxm-runtime.
    The generated code:

    - Uses apxm-runtime's public API (AgentBuilder, Executor, operations)
    - Is human-readable and debuggable
    - Compiles to native code with rustc
    - Achieves native performance (no interpretation overhead)

    This pass enables ahead-of-time compilation of agent programs.
  }];
  let constructor = "mlir::ais::createAISToRustPass()";
  let options = [
    Option<"output", "output", "std::string", /*default=*/"\"\"",
           "Output file path for generated Rust code">
  ];
}

#endif // APXM_AIS_PASSES
