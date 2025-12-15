/**
 * @file  Module.cpp
 * @brief Parse and serialise AIS modules from/to textual MLIR.
 *
 * `apxm_module_parse*` turns text into an `ApxmModule`;
 * `apxm_module_to_string` does the reverse.
 * `apxm_module_verify` runs the standard MLIR verifier and pushes any
 * diagnostics into the thread-local error collector.
 *
 * The environment variable `APXM_PRINT_LOCATIONS=1` enables location
 * printing in the textual output for debugging.
 */


#include "apxm/CAPI/Module.h"
#include "apxm/CAPI/Error.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

extern "C" {

ApxmModule* apxm_module_parse(ApxmCompilerContext* ctx, const char* mlir_text) {
  if (!ctx || !mlir_text) {
    return nullptr;
  }

  auto source = llvm::MemoryBuffer::getMemBuffer(mlir_text);
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(source), llvm::SMLoc());

  mlir::ParserConfig config(ctx->mlir_context.get());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, config);
  return module ? new (std::nothrow) ApxmModule(ctx, module.release()) : nullptr;
}

ApxmModule* apxm_module_parse_file(ApxmCompilerContext* ctx, const char* file_path) {
  if (!ctx || !file_path) {
    return nullptr;
  }

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(file_path, ctx->mlir_context.get());
  return module ? new (std::nothrow) ApxmModule(ctx, module.release()) : nullptr;
}

bool apxm_module_verify(ApxmModule* module) {
  if (!module || !module->module) {
    return false;
  }
  return mlir::succeeded(mlir::verify(*module->module));
}

char* apxm_module_to_string(ApxmModule* module) {
  if (!module || !module->module) {
    return nullptr;
  }

  std::string str;
  llvm::raw_string_ostream os(str);

  static const char* debug_locations_env = std::getenv("APXM_PRINT_LOCATIONS");
  if (debug_locations_env && std::string(debug_locations_env) == "1") {
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(true);
    module->module->print(os, flags);
  } else {
    module->module->print(os);
  }

  os.flush();
  return strdup(str.c_str());
}

void apxm_module_destroy(ApxmModule* module) {
  delete module;
}

} // extern "C"
