#ifndef APXM_CAPI_DSL_H
#define APXM_CAPI_DSL_H

#include "ais/CAPI/Types.h"
#include "ais/CAPI/Module.h"
#include "ais/CAPI/Error.h"

#ifdef __cplusplus
extern "C" {
#endif

// DSL parsing interface
ApxmModule *apxm_parse_dsl(ApxmCompilerContext *ctx, const char *source, const char *filename);
ApxmModule *apxm_parse_dsl_file(ApxmCompilerContext *ctx, const char *path);

#ifdef __cplusplus
}
#endif

#endif // APXM_CAPI_DSL_H
