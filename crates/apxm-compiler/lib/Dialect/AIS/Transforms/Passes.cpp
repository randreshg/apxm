/**
 * @file  Passes.cpp
 * @brief Registration hook for every AIS transform pass.
 *
 * A single call to `registerAISPasses()` inserts all passes declared in
 * Passes.td into the global MLIR registry, making them available to
 * `mlir-opt`, `mlir-translate`, or any tool that links against this
 * library.  The generated function is a one-liner; all per-pass logic
 * lives in the individual `.cpp` implementation files.
 */

#include "apxm/Dialect/AIS/Transforms/Passes.h"

namespace mlir::ais {
#define GEN_PASS_REGISTRATION
#include "apxm/Dialect/AIS/Transforms/Passes.h.inc"
} // namespace mlir::ais
