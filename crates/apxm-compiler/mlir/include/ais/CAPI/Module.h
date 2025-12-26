#ifndef APXM_CAPI_MODULE_H
#define APXM_CAPI_MODULE_H

#include "ais/CAPI/Types.h"
#include "ais/CAPI/Error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiler context lifecycle
ApxmCompilerContext *apxm_compiler_context_create(void);
void apxm_compiler_context_destroy(ApxmCompilerContext *ctx);

// Module parsing and serialization
ApxmModule *apxm_module_parse(ApxmCompilerContext *ctx, const char *mlir_text);
ApxmModule *apxm_module_parse_file(ApxmCompilerContext *ctx, const char *file_path);
bool apxm_module_verify(ApxmModule *module);
char *apxm_module_to_string(ApxmModule *module);
void apxm_module_destroy(ApxmModule *module);

#ifdef __cplusplus
}
#endif

#endif // APXM_CAPI_MODULE_H
