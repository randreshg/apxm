/*
 * @file AISToRust.cpp
 * @brief Implementation of public API for AIS MLIR to Rust code generation
 *
 *
 */

#include "ais/Dialect/AIS/Conversion/Rust/AISToRust.h"
#include "ais/Dialect/AIS/Conversion/Rust/RustEmitter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::ais;

LogicalResult mlir::ais::emitRustSource(ModuleOp module, llvm::raw_ostream &os,
                                       const RustCodegenOptions &options) {
  RustEmitter emitter(os, options);
  return emitter.emitModule(module);
}

std::string mlir::ais::generateRustSource(ModuleOp module,
                                         const RustCodegenOptions &options) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (failed(emitRustSource(module, os, options))) {
    return {};
  }
  os.flush();
  return buffer;
}

namespace {

/// Pass to emit Rust source code from AIS MLIR
class AISToRustPass : public PassWrapper<AISToRustPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AISToRustPass)

  explicit AISToRustPass(const RustCodegenOptions &opts = {})
      : options(opts) {}

  AISToRustPass(const AISToRustPass &other)
      : PassWrapper<AISToRustPass, OperationPass<ModuleOp>>(other), options(other.options) {}

  StringRef getArgument() const final {
    return "ais-emit-rust";
  }

  StringRef getDescription() const final {
    return "Emit Rust source code from AIS MLIR";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string contents;
    llvm::raw_string_ostream os(contents);

    if (failed(emitRustSource(module, os, options))) {
      module.emitError("failed to emit Rust source");
      signalPassFailure();
      return;
    }
    os.flush();

    if (outputPath.empty()) {
      llvm::outs() << contents << '\n';
      return;
    }

    std::error_code ec;
    llvm::raw_fd_ostream file(outputPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError("failed to open output file: ") << ec.message();
      signalPassFailure();
      return;
    }
    file << contents;
  }

  Option<std::string> outputPath{
      *this, "output", llvm::cl::desc("Output file path for generated Rust code")};

private:
  RustCodegenOptions options;
};

} // namespace

std::unique_ptr<Pass> mlir::ais::createAISToRustPass() {
  return std::make_unique<AISToRustPass>();
}

std::unique_ptr<Pass> mlir::ais::createAISToRustPass(const RustCodegenOptions &options) {
  return std::make_unique<AISToRustPass>(options);
}

void mlir::ais::registerAISToRustPass() {
  PassRegistration<AISToRustPass>();
}
