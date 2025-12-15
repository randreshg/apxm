#ifndef APXM_CAPI_ERROR_H
#define APXM_CAPI_ERROR_H

#include "apxm/CAPI/Types.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// LLVM-style rich error diagnostics
typedef struct {
  uint32_t code;
  const char *message;
  const char *file_path;
  uint32_t file_line;
  uint32_t file_col;
  uint32_t file_line_end;
  uint32_t file_col_end;
  const char *snippet;
  uint32_t highlight_start;
  uint32_t highlight_end;
  const char *label;
  const char *help;
  size_t suggestions_count;
  size_t secondary_spans_count;
  size_t notes_count;
} ApxmError;

// Global error collector interface
void apxm_error_collector_clear(void);
size_t apxm_error_collector_count(void);
bool apxm_error_collector_has_errors(void);

// Error retrieval (caller owns returned memory)
size_t apxm_error_collector_get_all(ApxmError **out_errors, size_t max_count);
const ApxmError *apxm_error_collector_get_first(void);

// Push a new error into the collector
void apxm_error_collector_add(uint32_t code,
                              const char *message,
                              const char *file_path,
                              uint32_t file_line,
                              uint32_t file_col,
                              uint32_t file_line_end,
                              uint32_t file_col_end,
                              const char *snippet,
                              uint32_t highlight_start,
                              uint32_t highlight_end,
                              const char *label,
                              const char *help,
                              size_t suggestions_count,
                              size_t secondary_spans_count,
                              size_t notes_count);

// Memory management for errors
void apxm_error_free(ApxmError *err);
void apxm_error_array_free(ApxmError *errors, size_t count);

#ifdef __cplusplus
}
#endif

#endif // APXM_CAPI_ERROR_H
