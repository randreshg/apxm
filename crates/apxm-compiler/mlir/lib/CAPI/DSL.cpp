/**
 * @file  DSL.cpp
 * @brief Front-end that turns the high-level agent DSL into an AIS module.
 *
 * The lexer and parser are implemented in `apxm/Parser/`; this file only
 * wires them to the C API.  Diagnostics emitted by the parser are captured
 * and converted into the uniform error format so that hosts see the same
 * rich diagnostics for *both* DSL and MLIR-level problems.
 */

#include "ais/CAPI/DSL.h"
#include "ais/CAPI/Error.h"
#if defined(APXM_HAS_CPP_DSL)
#include "ais/Parser/Core/Parser.h"
#include "ais/Parser/Lexer/Lexer.h"
#include "ais/Parser/MLIR/MLIRGen.h"
#endif
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

#if defined(APXM_HAS_CPP_DSL)

namespace {

enum class DiagnosticSeverity : uint32_t {
  Error = 1,
  Warning = 2,
  Note = 3,
};

struct DiagnosticEntry {
  DiagnosticSeverity severity;
  std::string file;
  uint32_t line;
  uint32_t column;
  uint32_t column_end;
  std::string message;
  std::string category;
};

uint32_t safeUInt(int value) {
  return value > 0 ? static_cast<uint32_t>(value) : 0;
}

DiagnosticSeverity toSeverity(llvm::SourceMgr::DiagKind kind) {
  switch (kind) {
  case llvm::SourceMgr::DK_Error:
    return DiagnosticSeverity::Error;
  case llvm::SourceMgr::DK_Warning:
    return DiagnosticSeverity::Warning;
  case llvm::SourceMgr::DK_Note:
  case llvm::SourceMgr::DK_Remark:
    return DiagnosticSeverity::Note;
  }
  return DiagnosticSeverity::Error;
}

const char* toCategory(llvm::SourceMgr::DiagKind kind) {
  switch (kind) {
  case llvm::SourceMgr::DK_Error:
    return "error";
  case llvm::SourceMgr::DK_Warning:
    return "warning";
  case llvm::SourceMgr::DK_Note:
    return "note";
  case llvm::SourceMgr::DK_Remark:
    return "remark";
  }
  return "diagnostic";
}

struct DiagnosticCapture {
  std::vector<DiagnosticEntry> entries;

  static void handler(const llvm::SMDiagnostic& diag, void* context) {
    auto* capture = static_cast<DiagnosticCapture*>(context);

    DiagnosticEntry entry;
    entry.severity = toSeverity(diag.getKind());
    entry.file = diag.getFilename().empty() ? "<input>" : diag.getFilename().str();
    entry.line = safeUInt(diag.getLineNo());
    entry.column = safeUInt(diag.getColumnNo());
    entry.column_end = entry.column > 0 ? entry.column + 1 : 0;
    entry.message = diag.getMessage().str();
    entry.category = toCategory(diag.getKind());

    capture->entries.push_back(std::move(entry));
  }
};

ApxmModule* parse_internal(ApxmCompilerContext* ctx, llvm::SourceMgr& srcMgr, unsigned bufferId) {
  DiagnosticCapture capture;
  srcMgr.setDiagHandler(DiagnosticCapture::handler, &capture);

  apxm::parser::Lexer lexer(srcMgr, bufferId);
  apxm::parser::Parser parser(lexer);
  auto agent = parser.parseAgent();

  if (!agent || parser.hadError()) {
    // Convert diagnostics to errors
    for (auto& entry : capture.entries) {
      apxm_error_collector_add(static_cast<uint32_t>(entry.severity),
                               entry.message.c_str(),
                               entry.file.c_str(),
                               entry.line,
                               entry.column,
                               entry.line,
                               entry.column_end,
                               "",
                               0,
                               0,
                               entry.category.c_str(),
                               "",
                               0,
                               0,
                               0);
    }
    return nullptr;
  }

  apxm::parser::MLIRGen mlirGen(*ctx->mlir_context, srcMgr);
  auto module = mlirGen.generateModule(agent.get());
  return module ? new (std::nothrow) ApxmModule(ctx, module.release()) : nullptr;
}

} // anonymous namespace

#else

namespace {

ApxmModule* parse_internal(ApxmCompilerContext* ctx, llvm::SourceMgr& srcMgr, unsigned bufferId) {
  (void)ctx;
  (void)srcMgr;
  (void)bufferId;

  apxm_error_collector_add(1,
                           "C++ DSL parser is not available in this build",
                           "<input>",
                           0,
                           0,
                           0,
                           0,
                           "",
                           0,
                           0,
                           "not-implemented",
                           "Rebuild with the parser enabled",
                           0,
                           0,
                           0);
  return nullptr;
}

} // anonymous namespace

#endif

extern "C" {

ApxmModule* apxm_parse_dsl(ApxmCompilerContext* ctx, const char* source, const char* filename) {
  if (!ctx || !source) {
    return nullptr;
  }

  apxm_error_collector_clear();

  llvm::SourceMgr srcMgr;
  auto memBuf = llvm::MemoryBuffer::getMemBuffer(source, filename ? filename : "<input>");
  unsigned bufferId = srcMgr.AddNewSourceBuffer(std::move(memBuf), llvm::SMLoc());

  return parse_internal(ctx, srcMgr, bufferId);
}

ApxmModule* apxm_parse_dsl_file(ApxmCompilerContext* ctx, const char* path) {
  if (!ctx || !path) {
    return nullptr;
  }

  apxm_error_collector_clear();

  auto fileOrErr = llvm::MemoryBuffer::getFile(path);
  if (!fileOrErr) {
    apxm_error_collector_add(1,
                             "Failed to read file",
                             path,
                             0,
                             0,
                             0,
                             0,
                             "",
                             0,
                             0,
                             "io-error",
                             "Check if file exists and has read permissions",
                             0,
                             0,
                             0);
    return nullptr;
  }

  llvm::SourceMgr srcMgr;
  unsigned bufferId = srcMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  return parse_internal(ctx, srcMgr, bufferId);
}

} // extern "C"
