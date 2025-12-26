/*
 * @file RustEmitter.h
 * @brief Interface for Rust code emission from AIS MLIR
 *
 * This file exposes a way to emit Rust code from AIS MLIR.
 * We need this to have a transpiler that can lower to runtime.
 */

#ifndef APXM_TARGET_RUST_RUSTEMITTER_H
#define APXM_TARGET_RUST_RUSTEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "ais/Dialect/AIS/Conversion/Rust/AISToRust.h"

namespace mlir::ais {

/// Rust code emitter interface
class RustEmitter {
public:
  /// Create emitter with output stream and options
  explicit RustEmitter(llvm::raw_ostream &os, const RustCodegenOptions &options);

  ~RustEmitter();

  /// Emit complete module to output stream
  [[nodiscard]] LogicalResult emitModule(ModuleOp module);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::ais

#endif // APXM_TARGET_RUST_RUSTEMITTER_H
