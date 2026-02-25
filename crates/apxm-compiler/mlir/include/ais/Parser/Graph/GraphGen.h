/*
 * @file GraphGen.h
 * @brief Lower AIS DSL AST into canonical ApxmGraph JSON.
 */

#ifndef APXM_PARSER_GRAPH_GRAPHGEN_H
#define APXM_PARSER_GRAPH_GRAPHGEN_H

#include "ais/Parser/AST/Declaration.h"
#include "ais/Parser/Utils/Location.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>
#include <string>
#include <vector>

namespace apxm::parser {

struct GraphGenError {
  Location location;
  std::string message;
};

/// Graph canonicalization pass for DSL frontend.
///
/// This pass walks DSL AST declarations/statements and emits canonical
/// ApxmGraph JSON. It currently targets flow-style DAG programs and reports
/// diagnostics for unsupported structured-control constructs.
class GraphGen {
public:
  explicit GraphGen(llvm::SourceMgr &sourceMgr);

  /// Lower input agents into a single canonical graph JSON document.
  /// Returns nullopt when lowering fails; inspect getLastError().
  std::optional<std::string>
  generateEntryGraphJson(const std::vector<std::unique_ptr<AgentDecl>> &agents);

  const std::optional<GraphGenError> &getLastError() const noexcept {
    return lastError;
  }

private:
  llvm::SourceMgr &srcMgr;
  std::optional<GraphGenError> lastError;
};

} // namespace apxm::parser

#endif // APXM_PARSER_GRAPH_GRAPHGEN_H
