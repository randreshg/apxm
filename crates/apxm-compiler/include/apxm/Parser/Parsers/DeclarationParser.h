/*
 * @file DeclarationParser.h
 * @brief Parser for agent declarations and BDI components
 *
 * Declarations are used to define the structure and behavior of agents within the system.
 */

#ifndef APXM_PARSER_PARSERS_DECLARATIONPARSER_H
#define APXM_PARSER_PARSERS_DECLARATIONPARSER_H

#include "apxm/Parser/AST/AST.h"
#include "apxm/Parser/Parsers/ParserBase.h"
#include <memory>
#include <vector>

namespace apxm::parser {

class StatementParser;
class ExpressionParser;

/// Parser for agent declarations and BDI components
class DeclarationParser final : public ParserBase {
public:
  DeclarationParser(Lexer &lexer, StatementParser &stmtParser,
                   ExpressionParser &exprParser, bool &hadErrorFlag) noexcept
      : ParserBase(lexer, hadErrorFlag), stmtParser(stmtParser), exprParser(exprParser) {}

  [[nodiscard]] std::unique_ptr<AgentDecl> parseAgent();

private:
  StatementParser &stmtParser;
  ExpressionParser &exprParser;

  // Declaration parsing methods
  [[nodiscard]] std::unique_ptr<MemoryDecl> parseMemoryDecl();
  [[nodiscard]] std::unique_ptr<CapabilityDecl> parseCapabilityDecl();
  [[nodiscard]] std::unique_ptr<FlowDecl> parseFlowDecl();
  [[nodiscard]] std::unique_ptr<BeliefDecl> parseBeliefDecl();
  [[nodiscard]] std::unique_ptr<GoalDecl> parseGoalDecl();
  [[nodiscard]] std::unique_ptr<OnEventDecl> parseOnEventDecl();

  // Helper methods
  bool parseNamedTypeList(TokenKind endToken, llvm::StringRef elementDesc,
                         llvm::SmallVectorImpl<std::pair<std::string, std::string>> &storage,
                         bool allowTypeKeywords = false);
  bool parsePatternFieldList(llvm::SmallVectorImpl<std::pair<std::string, std::string>> &storage);
};

} // namespace apxm::parser

#endif // APXM_PARSER_PARSERS_DECLARATIONPARSER_H
