#ifndef APXM_CAPI_CODEGEN_H
#define APXM_CAPI_CODEGEN_H

#include "ais/CAPI/Types.h"
#include "ais/CAPI/Module.h"
#include "ais/CAPI/Error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *module_name;
  bool emit_debug_json;
  const char *target_version;
} ApxmArtifactOptions;

// Artifact generation interface
char *apxm_codegen_emit_artifact(ApxmModule *module, const ApxmArtifactOptions *options);
void apxm_codegen_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif // APXM_CAPI_CODEGEN_H
