//===----------------------------------------------------------------------===//
// This header provides compile-time constants for all runtime symbols,
// ensuring consistency between the MLIR compiler and Rust runtime.
//===----------------------------------------------------------------------===//

#ifndef APXM_AIS_GENERATED_RUNTIME_SYMBOLS_H
#define APXM_AIS_GENERATED_RUNTIME_SYMBOLS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace ais {
namespace runtime {
namespace symbols {

/// Query memory from Agent Abstract Machine (AAM) (id: 0)
constexpr llvm::StringLiteral QMEM = "ais_runtime_qmem";

/// Update memory in Agent Abstract Machine (AAM) (id: 1)
constexpr llvm::StringLiteral UMEM = "ais_runtime_umem";

/// Invoke external capability (tool/API) (id: 2)
constexpr llvm::StringLiteral INV = "ais_runtime_inv";

/// Invoke reasoning with LLM/SLM (id: 3)
constexpr llvm::StringLiteral RSN = "ais_runtime_rsn";

/// Execute sandboxed code (id: 4)
constexpr llvm::StringLiteral EXC = "ais_runtime_exc";

/// Decompose a goal into AIS subgraph (id: 5)
constexpr llvm::StringLiteral PLAN = "ais_runtime_plan";

/// Analyze episodic traces for improvement (id: 6)
constexpr llvm::StringLiteral REFLECT = "ais_runtime_reflect";

/// Fact-check claims against evidence (id: 7)
constexpr llvm::StringLiteral VERIFY = "ais_runtime_verify";

/// Unconditional jump to label (id: 8)
constexpr llvm::StringLiteral JUMP = "ais_runtime_jump";

/// Conditional branch on token value (id: 9)
constexpr llvm::StringLiteral BRANCH = "ais_runtime_branch";

/// Begin bounded loop (id: 10)
constexpr llvm::StringLiteral LOOP_START = "ais_runtime_loop_start";

/// End bounded loop (id: 11)
constexpr llvm::StringLiteral LOOP_END = "ais_runtime_loop_end";

/// Return from subgraph with value (id: 12)
constexpr llvm::StringLiteral RETURN_OP = "ais_runtime_return";

/// Merge multiple tokens into one (id: 13)
constexpr llvm::StringLiteral MERGE = "ais_runtime_merge";

/// Memory fence/barrier (id: 14)
constexpr llvm::StringLiteral FENCE = "ais_runtime_fence";

/// Wait for all tokens to be ready (id: 15)
constexpr llvm::StringLiteral WAIT_ALL = "ais_runtime_wait_all";

/// Set up try/catch exception handling (id: 16)
constexpr llvm::StringLiteral TRY_CATCH = "ais_runtime_try_catch";

/// Error recovery operation (id: 17)
constexpr llvm::StringLiteral ERR = "ais_runtime_err";

/// Inter-agent communication (id: 18)
constexpr llvm::StringLiteral COMMUNICATE = "ais_runtime_communicate";

} // namespace symbols

/// Operation categories for scheduling
namespace categories {
    constexpr llvm::StringLiteral MEMORY = "memory";
    constexpr llvm::StringLiteral CAPABILITY = "capability";
    constexpr llvm::StringLiteral REASONING = "reasoning";
    constexpr llvm::StringLiteral EXECUTION = "execution";
    constexpr llvm::StringLiteral CONTROL = "control";
    constexpr llvm::StringLiteral SYNC = "sync";
    constexpr llvm::StringLiteral ERROR = "error";
    constexpr llvm::StringLiteral MULTIAGENT = "multiagent";
} // namespace categories

/// Get symbol name for operation (use in AISOps.cpp)
inline llvm::StringRef getOperationSymbol(unsigned opId) {
    switch (opId) {
    case 0: return symbols::QMEM;
    case 1: return symbols::UMEM;
    case 2: return symbols::INV;
    case 3: return symbols::RSN;
    case 4: return symbols::EXC;
    case 5: return symbols::PLAN;
    case 6: return symbols::REFLECT;
    case 7: return symbols::VERIFY;
    case 8: return symbols::JUMP;
    case 9: return symbols::BRANCH;
    case 10: return symbols::LOOP_START;
    case 11: return symbols::LOOP_END;
    case 12: return symbols::RETURN_OP;
    case 13: return symbols::MERGE;
    case 14: return symbols::FENCE;
    case 15: return symbols::WAIT_ALL;
    case 16: return symbols::TRY_CATCH;
    case 17: return symbols::ERR;
    case 18: return symbols::COMMUNICATE;
    default: return "";
    }
}

/// Get estimated cost for scheduling
inline unsigned getOperationCost(unsigned opId) {
    switch (opId) {
    case 0: return 2;
    case 1: return 2;
    case 2: return 5;
    case 3: return 10;
    case 4: return 5;
    case 5: return 10;
    case 6: return 8;
    case 7: return 8;
    case 8: return 1;
    case 9: return 1;
    case 10: return 1;
    case 11: return 1;
    case 12: return 1;
    case 13: return 2;
    case 14: return 1;
    case 15: return 3;
    case 16: return 1;
    case 17: return 3;
    case 18: return 5;
    default: return 1;
    }
}

/// Total number of operations
constexpr unsigned OPERATION_COUNT = 19;

} // namespace runtime
} // namespace ais
} // namespace mlir

#endif // APXM_AIS_GENERATED_RUNTIME_SYMBOLS_H
