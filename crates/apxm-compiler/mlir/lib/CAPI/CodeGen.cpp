/**
 * @file  Codegen.cpp
 * @brief Artifact emitter entry points for the C API.
 *
 * The returned buffer is heap-allocated and must be freed with `apxm_codegen_free`.
 */

#include "ais/CAPI/CodeGen.h"
#include "ais/CAPI/Module.h"
#include "ais/Dialect/AIS/Conversion/Artifact/ArtifactEmitter.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <cstdlib>

extern "C" {

char* apxm_codegen_emit_artifact(ApxmModule* module, const ApxmArtifactOptions* options) {
  if (!module || !module->module)
    return nullptr;

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
}

void apxm_codegen_free(char *ptr) {
  free(ptr);
}

} // extern "C"
