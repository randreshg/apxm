#ifndef APXM_CAPI_CODEGEN_H
#define APXM_CAPI_CODEGEN_H

#include "apxm/CAPI/Types.h"
#include "apxm/CAPI/Module.h"
#include "apxm/CAPI/Error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  bool optimize;
  bool emit_comments;
  bool emit_debug_symbols;
  bool standalone;
  const char *module_name;
} ApxmCodegenOptions;

typedef struct {
  const char *module_name;
  bool emit_debug_json;
  const char *target_version;
} ApxmArtifactOptions;

// Rust code generation interface
char *apxm_codegen_emit_rust(ApxmModule *module);
char *apxm_codegen_emit_rust_with_options(ApxmModule *module, const ApxmCodegenOptions *options);

// Artifact generation interface
char *apxm_codegen_emit_artifact(ApxmModule *module, const ApxmArtifactOptions *options);
void apxm_codegen_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif // APXM_CAPI_CODEGEN_H
