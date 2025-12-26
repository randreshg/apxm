#ifndef APXM_CAPI_TYPES_H
#define APXM_CAPI_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles for core compiler entities
typedef struct ApxmCompilerContext ApxmCompilerContext;
typedef struct ApxmModule ApxmModule;
typedef struct ApxmPassManager ApxmPassManager;

// Utility functions for memory management
void apxm_string_free(char *str);

#ifdef __cplusplus
}
#endif

/* Internal C++ definitions
 *
 * For internal C++ implementation files we provide the concrete definitions
 * of the opaque handles. These definitions are kept in a dedicated header
 * to avoid duplication and to keep the public header opaque for C users.
 *
 * The implementation target should define APXM_CAPI_INTERNAL (via the
 * target_compile_definitions in the CMakeLists) so that this include is
 * enabled only for the internal build.
 */
#if defined(__cplusplus) && defined(APXM_CAPI_INTERNAL)
#include "ais/CAPI/Internal.h"
#endif

#endif // APXM_CAPI_TYPES_H
