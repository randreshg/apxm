/**
 * @file  Codegen.cpp
 * @brief Rust source-code emitter entry points for the C API.
 *
 * The returned C string is heap-allocated
 * and must be freed with `apxm_string_free`.
 */

#include "apxm/CAPI/CodeGen.h"
#include "apxm/CAPI/Module.h"
#if defined(APXM_HAS_RUST_EMITTER)
#include "apxm/Dialect/AIS/Conversion/Rust/AISToRust.h"
#include "apxm/Dialect/AIS/Conversion/Artifact/ArtifactEmitter.h"
#include "mlir/Support/LogicalResult.h"
#endif
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <cstdlib>

// Use the public `mlir::ais::RustCodegenOptions` declared in the
// AIS-to-Rust header. The local anonymous duplicate struct was removed
// to avoid type/ABI confusion and to reuse the canonical definition.

extern "C" {

char* apxm_codegen_emit_rust(ApxmModule* module) {
  ApxmCodegenOptions defaultOpts = {
    true,  // optimize
    true,  // emit_comments
    false, // emit_debug_symbols
    true,  // standalone
    nullptr // module_name
  };
  return apxm_codegen_emit_rust_with_options(module, &defaultOpts);
}

char* apxm_codegen_emit_rust_with_options(ApxmModule* module, const ApxmCodegenOptions* options) {
  if (!module || !module->module)
    return nullptr;

#if !defined(APXM_HAS_RUST_EMITTER)
  (void)options;
  return nullptr;
#else
  mlir::ais::RustCodegenOptions cppOptions;
  if (options) {
    cppOptions.optimize = options->optimize;
    cppOptions.emitComments = options->emit_comments;
    cppOptions.emitDebugSymbols = options->emit_debug_symbols;
    cppOptions.emitMainFunction = options->standalone;
    if (options->module_name && options->module_name[0]) {
      cppOptions.moduleName = options->module_name;
    }
  }

  std::string output;
  llvm::raw_string_ostream os(output);

  if (mlir::failed(mlir::ais::emitRustSource(*module->module, os, cppOptions))) {
    return nullptr;
  }

  os.flush();

  char* result = static_cast<char*>(malloc(output.size() + 1));
  if (!result) return nullptr;

  memcpy(result, output.c_str(), output.size());
  result[output.size()] = '\0';
  return result;
#endif
}

char* apxm_codegen_emit_artifact(ApxmModule* module, const ApxmArtifactOptions* options) {
  if (!module || !module->module)
    return nullptr;

#if !defined(APXM_HAS_RUST_EMITTER)
  (void)options;
  return nullptr;
#else
  mlir::ais::ArtifactEmitOptions cppOptions;
  if (options) {
    if (options->module_name && options->module_name[0])
      cppOptions.moduleName = options->module_name;
    cppOptions.emitDebugJson = options->emit_debug_json;
    if (options->target_version && options->target_version[0])
      cppOptions.targetVersion = options->target_version;
  }

  mlir::ais::ArtifactEmitter emitter(cppOptions);
  if (mlir::failed(emitter.emitModule(*module->module)))
    return nullptr;

  const auto &payload = emitter.getBuffer();
  uint64_t size = static_cast<uint64_t>(payload.size());
  size_t total = sizeof(uint64_t) + payload.size();
  char *result = static_cast<char *>(malloc(total));
  if (!result)
    return nullptr;

  std::memcpy(result, &size, sizeof(uint64_t));
  if (!payload.empty()) {
    std::memcpy(result + sizeof(uint64_t), payload.data(), payload.size());
  }
  return result;
#endif
}

void apxm_codegen_free(char *ptr) {
  free(ptr);
}

} // extern "C"
