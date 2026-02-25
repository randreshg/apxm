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
#include "ais/Parser/Graph/GraphGen.h"
#include "ais/Parser/Lexer/Lexer.h"
#include "ais/Parser/MLIR/MLIRGen.h"
#endif
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
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

void emitCapturedDiagnostics(const DiagnosticCapture& capture) {
  for (const auto& entry : capture.entries) {
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
}

void emitGraphLoweringError(const apxm::parser::GraphGenError& error, llvm::SourceMgr& srcMgr) {
  uint32_t line = 0;
  uint32_t column = 0;
  std::string fileName = "<input>";
  if (error.location.isValid()) {
    unsigned bufferId = srcMgr.FindBufferContainingLoc(error.location.getStart());
    if (bufferId != 0) {
      auto lineCol = srcMgr.getLineAndColumn(error.location.getStart(), bufferId);
      line = safeUInt(static_cast<int>(lineCol.first));
      column = safeUInt(static_cast<int>(lineCol.second));
      fileName = srcMgr.getMemoryBuffer(bufferId)->getBufferIdentifier().str();
    }
  }

  apxm_error_collector_add(static_cast<uint32_t>(DiagnosticSeverity::Error),
                           error.message.c_str(),
                           fileName.c_str(),
                           line,
                           column,
                           line,
                           column > 0 ? column + 1 : 0,
                           "",
                           0,
                           0,
                           "graph-lowering",
                           "",
                           0,
                           0,
                           0);
}

bool parseAgents(llvm::SourceMgr& srcMgr, unsigned bufferId, DiagnosticCapture& capture,
                 std::vector<std::unique_ptr<apxm::parser::AgentDecl>>& agents) {
  srcMgr.setDiagHandler(DiagnosticCapture::handler, &capture);

  apxm::parser::Lexer lexer(srcMgr, bufferId);
  apxm::parser::Parser parser(lexer);
  agents = parser.parseAgents();
  return !agents.empty() && !parser.hadError();
}

ApxmModule* parse_internal(ApxmCompilerContext* ctx, llvm::SourceMgr& srcMgr, unsigned bufferId) {
  DiagnosticCapture capture;
  std::vector<std::unique_ptr<apxm::parser::AgentDecl>> agents;
  if (!parseAgents(srcMgr, bufferId, capture, agents)) {
    emitCapturedDiagnostics(capture);
    return nullptr;
  }

  apxm::parser::MLIRGen mlirGen(*ctx->mlir_context, srcMgr);
  auto module = mlirGen.generateModuleFromAgents(agents);
  return module ? new (std::nothrow) ApxmModule(ctx, module.release()) : nullptr;
}

char* parse_graph_json_internal(ApxmCompilerContext* ctx, llvm::SourceMgr& srcMgr,
                                unsigned bufferId) {
  (void)ctx;
  DiagnosticCapture capture;
  std::vector<std::unique_ptr<apxm::parser::AgentDecl>> agents;
  if (!parseAgents(srcMgr, bufferId, capture, agents)) {
    emitCapturedDiagnostics(capture);
    return nullptr;
  }

  apxm::parser::GraphGen graphGen(srcMgr);
  auto graphJson = graphGen.generateEntryGraphJson(agents);
  if (!graphJson) {
    if (graphGen.getLastError()) {
      emitGraphLoweringError(*graphGen.getLastError(), srcMgr);
    } else {
      apxm_error_collector_add(1,
                               "Graph lowering failed",
                               "<input>",
                               0,
                               0,
                               0,
                               0,
                               "",
                               0,
                               0,
                               "graph-lowering",
                               "",
                               0,
                               0,
                               0);
    }
    return nullptr;
  }

  return strdup(graphJson->c_str());
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

char* parse_graph_json_internal(ApxmCompilerContext* ctx, llvm::SourceMgr& srcMgr,
                                unsigned bufferId) {
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

char* apxm_parse_dsl_to_graph_json(ApxmCompilerContext* ctx, const char* source,
                                   const char* filename) {
  if (!ctx || !source) {
    return nullptr;
  }

  apxm_error_collector_clear();

  llvm::SourceMgr srcMgr;
  auto memBuf = llvm::MemoryBuffer::getMemBuffer(source, filename ? filename : "<input>");
  unsigned bufferId = srcMgr.AddNewSourceBuffer(std::move(memBuf), llvm::SMLoc());
  return parse_graph_json_internal(ctx, srcMgr, bufferId);
}

char* apxm_parse_dsl_file_to_graph_json(ApxmCompilerContext* ctx, const char* path) {
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
  return parse_graph_json_internal(ctx, srcMgr, bufferId);
}

} // extern "C"
