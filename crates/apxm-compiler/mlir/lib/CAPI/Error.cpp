/**
 * @file  Error.cpp
 * @brief Thread-local error collector used by the C API.
 *
 * All parse, verify or codegen errors are funnelled through this collector
 * so that C callers can inspect them after every call.  The implementation
 * is *not* tied to MLIR's diagnostic engine; it is a stand-alone vector
 * protected by a mutex so that the library can be used from multi-threaded
 * hosts without data races.
 */

#include "ais/CAPI/Error.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <vector>
#include <mutex>

namespace {

struct ErrorDescriptor {
  uint32_t code;
  std::string message;
  std::string file_path;
  uint32_t file_line;
  uint32_t file_col;
  uint32_t file_line_end;
  uint32_t file_col_end;
  std::string snippet;
  uint32_t highlight_start;
  uint32_t highlight_end;
  std::string label;
  std::string help;
  size_t suggestions_count;
  size_t secondary_spans_count;
  size_t notes_count;
};

class ThreadLocalErrorCollector {
public:
  static ThreadLocalErrorCollector& instance() {
    static thread_local ThreadLocalErrorCollector collector;
    return collector;
  }

  void clear() {
    errors.clear();
  }

  void add(ErrorDescriptor desc) {
    errors.push_back(std::move(desc));
  }

  size_t count() const {
    return errors.size();
  }

  bool has_errors() const {
    return !errors.empty();
  }

  ErrorDescriptor* get(size_t index) {
    return index < errors.size() ? &errors[index] : nullptr;
  }

  void to_apxm_errors(llvm::SmallVectorImpl<ApxmError>& out) {
    out.clear();
    out.reserve(errors.size());

    for (auto& desc : errors) {
      ApxmError err;
      err.code = desc.code;
      err.message = desc.message.c_str();
      err.file_path = desc.file_path.c_str();
      err.file_line = desc.file_line;
      err.file_col = desc.file_col;
      err.file_line_end = desc.file_line_end;
      err.file_col_end = desc.file_col_end;
      err.snippet = desc.snippet.empty() ? nullptr : desc.snippet.c_str();
      err.highlight_start = desc.highlight_start;
      err.highlight_end = desc.highlight_end;
      err.label = desc.label.empty() ? nullptr : desc.label.c_str();
      err.help = desc.help.empty() ? nullptr : desc.help.c_str();
      err.suggestions_count = desc.suggestions_count;
      err.secondary_spans_count = desc.secondary_spans_count;
      err.notes_count = desc.notes_count;

      out.push_back(err);
    }
  }

private:
  ThreadLocalErrorCollector() = default;
  llvm::SmallVector<ErrorDescriptor, 4> errors;
};

} // anonymous namespace

extern "C" {

void apxm_error_collector_add(uint32_t code,
                              const char* message,
                              const char* file_path,
                              uint32_t file_line,
                              uint32_t file_col,
                              uint32_t file_line_end,
                              uint32_t file_col_end,
                              const char* snippet,
                              uint32_t highlight_start,
                              uint32_t highlight_end,
                              const char* label,
                              const char* help,
                              size_t suggestions_count,
                              size_t secondary_spans_count,
                              size_t notes_count) {
  ErrorDescriptor desc;
  desc.code = code;
  desc.message = message ? message : "";
  desc.file_path = file_path ? file_path : "";
  desc.file_line = file_line;
  desc.file_col = file_col;
  desc.file_line_end = file_line_end;
  desc.file_col_end = file_col_end;
  desc.snippet = snippet ? snippet : "";
  desc.highlight_start = highlight_start;
  desc.highlight_end = highlight_end;
  desc.label = label ? label : "";
  desc.help = help ? help : "";
  desc.suggestions_count = suggestions_count;
  desc.secondary_spans_count = secondary_spans_count;
  desc.notes_count = notes_count;

  ThreadLocalErrorCollector::instance().add(std::move(desc));
}

void apxm_error_collector_clear() {
  ThreadLocalErrorCollector::instance().clear();
}

size_t apxm_error_collector_count() {
  return ThreadLocalErrorCollector::instance().count();
}

bool apxm_error_collector_has_errors() {
  return ThreadLocalErrorCollector::instance().has_errors();
}

size_t apxm_error_collector_get_all(ApxmError** out_errors, size_t max_count) {
  if (!out_errors || max_count == 0)
    return 0;

  auto& collector = ThreadLocalErrorCollector::instance();
  size_t count = collector.count();
  if (count == 0)
    return 0;

  size_t to_copy = std::min(count, max_count);
  llvm::SmallVector<ApxmError, 8> apxm_errors;
  collector.to_apxm_errors(apxm_errors);

  for (size_t i = 0; i < to_copy; ++i) {
    ApxmError* err = static_cast<ApxmError*>(malloc(sizeof(ApxmError)));
    if (!err) {
      // Free any already allocated errors
      for (size_t j = 0; j < i; ++j) {
        free(out_errors[j]);
      }
      return 0;
    }
    *err = apxm_errors[i];
    out_errors[i] = err;
  }

  return to_copy;
}

const ApxmError* apxm_error_collector_get_first() {
  auto& collector = ThreadLocalErrorCollector::instance();
  if (!collector.has_errors())
    return nullptr;

  ErrorDescriptor* desc = collector.get(0);
  if (!desc)
    return nullptr;

  ApxmError* err = static_cast<ApxmError*>(malloc(sizeof(ApxmError)));
  if (!err)
    return nullptr;

  err->code = desc->code;
  err->message = desc->message.c_str();
  err->file_path = desc->file_path.c_str();
  err->file_line = desc->file_line;
  err->file_col = desc->file_col;
  err->file_line_end = desc->file_line_end;
  err->file_col_end = desc->file_col_end;
  err->snippet = desc->snippet.empty() ? nullptr : desc->snippet.c_str();
  err->highlight_start = desc->highlight_start;
  err->highlight_end = desc->highlight_end;
  err->label = desc->label.empty() ? nullptr : desc->label.c_str();
  err->help = desc->help.empty() ? nullptr : desc->help.c_str();
  err->suggestions_count = desc->suggestions_count;
  err->secondary_spans_count = desc->secondary_spans_count;
  err->notes_count = desc->notes_count;

  return err;
}

void apxm_error_free(ApxmError* err) {
  free(err);
}

void apxm_error_array_free(ApxmError* errors, size_t count) {
  if (!errors)
    return;

  for (size_t i = 0; i < count; ++i) {
    free(&errors[i]);
  }
  free(errors);
}

} // extern "C"
