/*
 * @file Parser.h
 * @brief Recursive descent parser for APXM DSL - Facade pattern
 * Delegates to specialized parsers: ExpressionParser, StatementParser, DeclarationParser
 * Uses LLVM SourceMgr for diagnostic emission and error recovery
 */
#ifndef APXM_PARSER_PARSER_H
#define APXM_PARSER_PARSER_H

#include "ais/Parser/AST/AST.h"
#include "ais/Parser/Lexer/Lexer.h"
#include "ais/Parser/Parsers/DeclarationParser.h"
#include "ais/Parser/Parsers/ExpressionParser.h"
#include "ais/Parser/Parsers/StatementParser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace apxm {
namespace parser {

/// Recursive descent parser for APXM DSL - Facade pattern
class Parser {
public:
  Parser(Lexer &lexer);

  /// Parse a complete agent definition
  std::unique_ptr<AgentDecl> parseAgent();

  /// Parse all agent definitions in the file
  std::vector<std::unique_ptr<AgentDecl>> parseAgents();

  /// Check if any errors were emitted during parsing
  bool hadError() const {
    return hadErrorFlag;
  }

private:
  Lexer &lexer;
  bool hadErrorFlag = false;

  // Specialized parsers
  ExpressionParser exprParser;
  StatementParser stmtParser;
  DeclarationParser declParser;
};

}  // namespace parser
}  // namespace apxm

#endif  // APXM_PARSER_PARSER_H
