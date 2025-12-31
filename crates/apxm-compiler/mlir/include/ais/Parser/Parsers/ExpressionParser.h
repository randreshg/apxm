/*
 * @file ExpressionParser.h
 * @brief Parser for expressions with operator precedence handling
 *
 * Expressions are parsed using operator precedence parsing.
 */

#ifndef APXM_PARSER_PARSERS_EXPRESSIONPARSER_H
#define APXM_PARSER_PARSERS_EXPRESSIONPARSER_H

#include "ais/Parser/AST/AST.h"
#include "ais/Parser/Parsers/ParserBase.h"
#include <memory>
#include <vector>

namespace apxm::parser {

/// Parser for expressions with operator precedence handling
class ExpressionParser final : public ParserBase {
public:
  ExpressionParser(Lexer &lexer, bool &hadErrorFlag) noexcept
      : ParserBase(lexer, hadErrorFlag) {}

  [[nodiscard]] std::unique_ptr<Expr> parseExpression();

private:
  // Operator precedence parsing (lowest to highest)
  [[nodiscard]] std::unique_ptr<Expr> parseAssignmentExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseLogicalOrExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseLogicalAndExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseEqualityExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseComparisonExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseAdditiveExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseMultiplicativeExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseUnaryExpr();
  [[nodiscard]] std::unique_ptr<Expr> parsePrimaryExpr();
  [[nodiscard]] std::unique_ptr<Expr> parsePostfixExpr(std::unique_ptr<Expr> base);
  [[nodiscard]] std::unique_ptr<Expr> parseArrayExpr();
  [[nodiscard]] std::unique_ptr<Expr> parseStringWithInterpolation(Location loc, llvm::StringRef unescaped);
  [[nodiscard]] std::unique_ptr<Expr> parseMemberAccess(std::unique_ptr<Expr> base);
  [[nodiscard]] std::unique_ptr<Expr> parseCallExpr(llvm::StringRef callee, Location loc);
  [[nodiscard]] std::unique_ptr<Expr> parsePipelineExpr();

  // Helper methods
  bool parseExpressionList(TokenKind endToken,
                          llvm::SmallVectorImpl<std::unique_ptr<Expr>> &storage);

  // Binary operator parsing template
  template <typename MatchFn, typename OpFn>
  [[nodiscard]] std::unique_ptr<Expr>
  parseBinaryExpr(std::unique_ptr<Expr> (ExpressionParser::*parseNext)(),
                 MatchFn matchOp,
                 OpFn getOpType);

  [[nodiscard]] std::unique_ptr<Expr> desugarPipeline(Location pipeLoc,
                                                    std::unique_ptr<Expr> value,
                                                    std::unique_ptr<Expr> stage);

  // Friend declarations for cross-parser access
  friend class StatementParser;
  friend class DeclarationParser;
};

} // namespace apxm::parser

#endif // APXM_PARSER_PARSERS_EXPRESSIONPARSER_H
