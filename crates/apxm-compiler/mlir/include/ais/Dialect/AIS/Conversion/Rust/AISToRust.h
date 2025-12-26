/*
 * @file AISToRust.h
 * @brief Public API for AIS MLIR to Rust code generation
 *
 * We provide a set of functions to generate Rust source code from AIS MLIR modules.
 */

#ifndef APXM_TARGET_RUST_AISTORUST_H
#define APXM_TARGET_RUST_AISTORUST_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir::ais {

/// Configuration for Rust code generation
struct RustCodegenOptions {
  /// Emit human-readable comments in generated code
  bool emitComments = true;

  /// Emit debug symbols and assertions
  bool emitDebugSymbols = false;

  /// Optimize generated code (inline operations, constant folding)
  bool optimize = true;

  /// Emit main() function for standalone execution
  bool emitMainFunction = true;

  /// Optional module name for builder function and metadata
  std::string moduleName;

  /// Indentation spaces per level (default: 4)
  unsigned indentSize = 4;
};

/// Generate Rust source code from an AIS MLIR module
///
/// \param module The MLIR module to transpile
/// \param os Output stream for generated Rust code
/// \param options Code generation options
/// \return success() if generation succeeded, failure() otherwise
[[nodiscard]] LogicalResult emitRustSource(ModuleOp module, llvm::raw_ostream &os,
                                          const RustCodegenOptions &options = {});

/// Generate Rust source code and return as a string
///
/// \param module The MLIR module to transpile
/// \param options Code generation options
/// \return Generated Rust source code, or empty string on failure
[[nodiscard]] std::string generateRustSource(ModuleOp module,
                                           const RustCodegenOptions &options = {});

/// Create Rust emission pass instances
std::unique_ptr<mlir::Pass> createAISToRustPass();
std::unique_ptr<mlir::Pass> createAISToRustPass(const RustCodegenOptions &options);
void registerAISToRustPass();

} // namespace mlir::ais

#endif // APXM_TARGET_RUST_AISTORUST_H
