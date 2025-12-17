/*
 * @file StatementParser.cpp
 * @brief Parse statements for the APXM DSL.
 */

#include "apxm/Parser/Parsers/StatementParser.h"
#include "apxm/Parser/Parsers/ExpressionParser.h"
#include "apxm/Parser/Utils/Container.h"
#include <memory>

using namespace apxm::parser;

std::unique_ptr<Stmt> StatementParser::parseStatement() {
  if (peek(TokenKind::kw_let)) {
    return parseLetStmt();
  } else if (peek(TokenKind::kw_return)) {
    return parseReturnStmt();
  } else if (peek(TokenKind::kw_if)) {
    return parseIfStmt();
  } else if (peek(TokenKind::kw_parallel)) {
    return parseParallelStmt();
  } else if (peek(TokenKind::kw_loop)) {
    return parseLoopStmt();
  } else if (peek(TokenKind::kw_try)) {
    return parseTryCatchStmt();
  } else {
    // Expression statement
    auto expr = exprParser.parseExpression();
    if (!expr) {
      synchronize();
      return nullptr;
    }

    Location loc = expr->getLocation();
    if (!expect(TokenKind::semicolon)) {
      synchronize();
      return nullptr;
    }

    return std::make_unique<ExprStmt>(loc, std::move(expr));
  }
}

std::unique_ptr<Stmt> StatementParser::parseLetStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_let)) return nullptr;

  Token nameTok = peek();
  if (!expect(TokenKind::identifier)) return nullptr;
  llvm::StringRef varName = nameTok.spelling;

  std::optional<llvm::StringRef> typeAnnotation;
  if (consume(TokenKind::colon)) {
    if (!peek(TokenKind::identifier) && !peek(TokenKind::kw_string) &&
        !peek(TokenKind::kw_number) && !peek(TokenKind::kw_bool) &&
        !peek(TokenKind::kw_json) && !peek(TokenKind::kw_void)) {
      emitError(getCurrentLocation(), "Expected type annotation");
      return nullptr;
    }
    typeAnnotation = peek().spelling;
    advance();
  }

  if (!expect(TokenKind::equal)) return nullptr;

  auto initExpr = exprParser.parseExpression();
  if (!initExpr) {
    synchronize();
    return nullptr;
  }

  if (!expect(TokenKind::semicolon)) return nullptr;

  return std::make_unique<LetStmt>(loc, varName, typeAnnotation, std::move(initExpr));
}

std::unique_ptr<Stmt> StatementParser::parseReturnStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_return)) return nullptr;

  std::unique_ptr<Expr> returnExpr;
  if (!peek(TokenKind::semicolon)) {
    returnExpr = exprParser.parseExpression();
    if (!returnExpr) {
      synchronize();
      return nullptr;
    }
  }

  if (!expect(TokenKind::semicolon)) return nullptr;

  return std::make_unique<ReturnStmt>(loc, std::move(returnExpr));
}

std::unique_ptr<Stmt> StatementParser::parseIfStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_if)) return nullptr;

  if (!expect(TokenKind::l_paren)) return nullptr;

  auto condition = exprParser.parseExpression();
  if (!condition) {
    synchronize();
    return nullptr;
  }

  if (!expect(TokenKind::r_paren)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 4> thenStmts;
  if (!parseStatementBlock(thenStmts)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 4> elseStmts;
  if (consume(TokenKind::kw_else)) {
    if (!parseStatementBlock(elseStmts)) return nullptr;
  }

  return std::make_unique<IfStmt>(loc, std::move(condition), thenStmts, elseStmts);
}

std::unique_ptr<Stmt> StatementParser::parseParallelStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_parallel)) return nullptr;

  if (!expect(TokenKind::l_brace)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 4> body;
  if (!parseStatementBlock(body)) return nullptr;

  if (!expect(TokenKind::r_brace)) return nullptr;

  return std::make_unique<ParallelStmt>(loc, body);
}

std::unique_ptr<Stmt> StatementParser::parseLoopStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_loop)) return nullptr;

  if (!expect(TokenKind::l_paren)) return nullptr;

  Token iterTok = peek();
  if (!expect(TokenKind::identifier)) return nullptr;
  llvm::StringRef varName = iterTok.spelling;

  if (!expect(TokenKind::kw_in)) return nullptr;

  auto collection = exprParser.parseExpression();
  if (!collection) {
    synchronize();
    return nullptr;
  }

  if (!expect(TokenKind::r_paren)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 4> body;
  if (!parseStatementBlock(body)) return nullptr;

  return std::make_unique<LoopStmt>(loc, varName, std::move(collection), body);
}

std::unique_ptr<Stmt> StatementParser::parseTryCatchStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_try)) return nullptr;

  if (!expect(TokenKind::l_brace)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 4> tryBody;
  if (!parseStatementBlock(tryBody)) return nullptr;

  if (!expect(TokenKind::kw_catch)) return nullptr;

  if (!expect(TokenKind::l_brace)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 4> catchBody;
  if (!parseStatementBlock(catchBody)) return nullptr;

  if (!expect(TokenKind::r_brace)) return nullptr;

  return std::make_unique<TryCatchStmt>(loc, tryBody, catchBody);
}

bool StatementParser::parseStatementBlock(llvm::SmallVectorImpl<std::unique_ptr<Stmt>> &storage) {
  if (peek(TokenKind::l_brace)) {
    advance(); // consume '{'
  } else {
    // Single statement block
    auto stmt = parseStatement();
    if (!stmt) return false;
    storage.push_back(std::move(stmt));
    return true;
  }

  while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
    auto stmt = parseStatement();
    if (!stmt) {
      synchronize();
      continue;
    }
    storage.push_back(std::move(stmt));
  }

  return expect(TokenKind::r_brace);
}
