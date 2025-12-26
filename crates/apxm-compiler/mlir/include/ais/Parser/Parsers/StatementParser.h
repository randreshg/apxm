/*
 * @file StatementParser.h
 * @brief Parser for statements in the APXM DSL.
 *
 * Statements are the basic building blocks of the APXM DSL.
 * They represent actions that can be performed in the program.
 *
 * Statements can be classified into several categories:
 *
 * 1. Declaration Statements: These statements declare variables, functions, or other entities in the program.
 * 2. Expression Statements: These statements evaluate expressions and discard the result.
 * 3. Control Flow Statements: These statements control the flow of execution in the program.
 * 4. Error Handling Statements: These statements handle errors that may occur during program execution.
 *
 * Each statement is represented by a subclass of the Stmt class, which is defined in the AST module.
 */

#ifndef APXM_PARSER_PARSERS_STATEMENTPARSER_H
#define APXM_PARSER_PARSERS_STATEMENTPARSER_H

#include "ais/Parser/AST/AST.h"
#include "ais/Parser/Parsers/ParserBase.h"
#include <memory>
#include <optional>
#include <vector>

namespace apxm::parser {

class ExpressionParser;

/// Parser for statements
class StatementParser final : public ParserBase {
public:
  StatementParser(Lexer &lexer, ExpressionParser &exprParser, bool &hadErrorFlag) noexcept
      : ParserBase(lexer, hadErrorFlag), exprParser(exprParser) {}

  [[nodiscard]] std::unique_ptr<Stmt> parseStatement();

private:
  ExpressionParser &exprParser;

  // Statement parsing methods
  [[nodiscard]] std::unique_ptr<Stmt> parseLetStmt();
  [[nodiscard]] std::unique_ptr<Stmt> parseReturnStmt();
  [[nodiscard]] std::unique_ptr<Stmt> parseIfStmt();
  [[nodiscard]] std::unique_ptr<Stmt> parseParallelStmt();
  [[nodiscard]] std::unique_ptr<Stmt> parseLoopStmt();
  [[nodiscard]] std::unique_ptr<Stmt> parseTryCatchStmt();

  // Helper methods
  bool parseStatementBlock(llvm::SmallVectorImpl<std::unique_ptr<Stmt>> &storage);

  // Friend declarations for cross-parser access
  friend class DeclarationParser;
};

} // namespace apxm::parser

#endif // APXM_PARSER_PARSERS_STATEMENTPARSER_H
