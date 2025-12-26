/**
 * @file Constants.h
 * @brief Central registry for string literals and numeric constants.
 *
 * This header keeps hard-coded tokens in one place so that:
 * - Types are caught at compile time
 * - Refactoring is trivial-change a constant here and every user picks it up
 */

#ifndef APXM_COMMON_CONSTANTS_H
#define APXM_COMMON_CONSTANTS_H

#include "llvm/ADT/StringRef.h"

namespace apxm {
namespace constants {

/// Memory space identifiers
namespace memory {
constexpr llvm::StringLiteral STM = "stm";
constexpr llvm::StringLiteral LTM = "ltm";
constexpr llvm::StringLiteral EPISODIC = "episodic";
constexpr llvm::StringLiteral DEFAULT_SPACE = "stm";
}  // namespace memory

/// Session and context identifiers
namespace session {
constexpr llvm::StringLiteral DEFAULT_SID = "default";
}  // namespace session

/// JSON and data format constants
namespace data {
constexpr llvm::StringLiteral EMPTY_JSON = "{}";
constexpr llvm::StringLiteral EMPTY_ARRAY = "[]";
}  // namespace data

/// Operation name aliases - used in parser and MLIR generation
namespace operations {
// Memory operations
constexpr llvm::StringLiteral QUERY_MEMORY = "query_memory";
constexpr llvm::StringLiteral QMEM = "qmem";
constexpr llvm::StringLiteral MEM = "mem";

// Invocation operations
constexpr llvm::StringLiteral INVOKE = "invoke";
constexpr llvm::StringLiteral INV = "inv";
constexpr llvm::StringLiteral LLM = "llm";
constexpr llvm::StringLiteral TOOL = "tool";

// Reasoning operations
constexpr llvm::StringLiteral REASON = "reason";
constexpr llvm::StringLiteral RSN = "rsn";
constexpr llvm::StringLiteral THINK = "think";

// Planning operations
constexpr llvm::StringLiteral PLAN = "plan";
constexpr llvm::StringLiteral PLN = "pln";

// Reflection operations
constexpr llvm::StringLiteral REFLECT = "reflect";
constexpr llvm::StringLiteral RFL = "rfl";

// Verification operations
constexpr llvm::StringLiteral VERIFY = "verify";
constexpr llvm::StringLiteral VRF = "vrf";

// Execution operations
constexpr llvm::StringLiteral EXEC = "exec";
constexpr llvm::StringLiteral EX = "ex";

// Communication operations
constexpr llvm::StringLiteral TALK = "talk";
constexpr llvm::StringLiteral TLK = "tlk";

// Synchronization operations
constexpr llvm::StringLiteral WAIT = "wait";
constexpr llvm::StringLiteral MERGE = "merge";
}  // namespace operations

/// Default file and input identifiers
namespace input {
constexpr llvm::StringLiteral DEFAULT_INPUT = "<input>";
constexpr llvm::StringLiteral STDIN = "<stdin>";
}  // namespace input

/// Version and metadata
namespace meta {
constexpr uint32_t BINARY_FORMAT_VERSION = 1;
constexpr uint64_t MIN_UNIQUE_ID = 1;
}  // namespace meta

}  // namespace constants
}  // namespace apxm

#endif  // APXM_COMMON_CONSTANTS_H
