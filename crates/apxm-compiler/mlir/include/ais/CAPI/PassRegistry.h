#ifndef APXM_CAPI_PASS_REGISTRY_H
#define APXM_CAPI_PASS_REGISTRY_H

#include "ais/CAPI/Types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  APXM_PASS_ANALYSIS = 0,
  APXM_PASS_TRANSFORM = 1,
  APXM_PASS_OPTIMIZATION = 2,
  APXM_PASS_LOWERING = 3
} ApxmPassCategory;

typedef struct {
  const char *name;
  const char *description;
  ApxmPassCategory category;
} ApxmPassInfo;

size_t apxm_pass_registry_get_count(void);
const ApxmPassInfo *apxm_pass_registry_get_pass(size_t index);
const ApxmPassInfo *apxm_pass_registry_find_pass(const char *name);

#ifdef __cplusplus
}
#endif

#endif // APXM_CAPI_PASS_REGISTRY_H
