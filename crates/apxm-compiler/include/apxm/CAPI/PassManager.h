#ifndef APXM_CAPI_PASS_MANAGER_H
#define APXM_CAPI_PASS_MANAGER_H

#include "apxm/CAPI/Types.h"
#include "apxm/CAPI/Module.h"
#include "apxm/CAPI/PassRegistry.h"
#include "apxm/CAPI/Error.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pass manager lifecycle
ApxmPassManager *apxm_pass_manager_create(ApxmCompilerContext *ctx);
void apxm_pass_manager_destroy(ApxmPassManager *pm);
void apxm_pass_manager_clear(ApxmPassManager *pm);

// Pass execution and registration
bool apxm_pass_manager_run(ApxmPassManager *pm, ApxmModule *module);
bool apxm_pass_manager_add_pass_by_name(ApxmPassManager *pm, const char *pass_name);
bool apxm_pass_manager_has_pass(ApxmPassManager *pm, const char *pass_name);

// Specific pass additions (grouped by category)
// Transform passes
void apxm_pass_manager_add_normalize(ApxmPassManager *pm);
void apxm_pass_manager_add_fuse_reasoning(ApxmPassManager *pm);
void apxm_pass_manager_add_scheduling(ApxmPassManager *pm);

// Optimization passes
void apxm_pass_manager_add_canonicalizer(ApxmPassManager *pm);  // Includes DCE
void apxm_pass_manager_add_cse(ApxmPassManager *pm);
void apxm_pass_manager_add_symbol_dce(ApxmPassManager *pm);

// Lowering passes
void apxm_pass_manager_add_lower_to_async(ApxmPassManager *pm);
// NOTE: lower_to_runtime intentionally not provided.

#ifdef __cplusplus
}
#endif

#endif // APXM_CAPI_PASS_MANAGER_H
