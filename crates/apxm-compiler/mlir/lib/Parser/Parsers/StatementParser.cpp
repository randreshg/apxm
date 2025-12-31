/*
 * @file StatementParser.cpp
 * @brief Parse statements for the APXM DSL.
 */

#include "ais/Parser/Parsers/StatementParser.h"
#include "ais/Parser/Parsers/ExpressionParser.h"
#include "ais/Parser/Utils/Container.h"
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
  } else if (peek(TokenKind::kw_switch)) {
    return parseSwitchStmt();
  } else {
    // Expression statement (possibly with binding)
    auto expr = exprParser.parseExpression();
    if (!expr) {
      synchronize();
      return nullptr;
    }

    Location loc = expr->getLocation();

    // Check for binding syntax: `expr -> varname`
    // This is syntactic sugar for: `let varname = expr;`
    if (consume(TokenKind::arrow)) {
      Token varTok = peek();
      if (!expectIdentifier()) {
        synchronize();
        return nullptr;
      }
      llvm::StringRef varName = varTok.spelling;

      // Semicolon is optional for binding statements in flow bodies
      consume(TokenKind::semicolon);

      // Convert to a LetStmt
      return std::make_unique<LetStmt>(loc, varName, std::nullopt, std::move(expr));
    }

    // Semicolon is optional for expression statements in flow bodies
    consume(TokenKind::semicolon);

    return std::make_unique<ExprStmt>(loc, std::move(expr));
  }
}

std::unique_ptr<Stmt> StatementParser::parseLetStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_let)) return nullptr;

  Token nameTok = peek();
  if (!expectIdentifier()) return nullptr;
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

  // Semicolon is optional for let statements in flow bodies
  consume(TokenKind::semicolon);

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

  // Semicolon is optional for return statements in flow bodies
  consume(TokenKind::semicolon);

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
  if (!expectIdentifier()) return nullptr;
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

std::unique_ptr<Stmt> StatementParser::parseSwitchStmt() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_switch)) return nullptr;

  // Parse discriminant expression
  auto discriminant = exprParser.parseExpression();
  if (!discriminant) {
    synchronize();
    return nullptr;
  }

  if (!expect(TokenKind::l_brace)) return nullptr;

  llvm::SmallVector<SwitchCase, 4> cases;
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> defaultBody;

  // Parse cases and default
  while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
    if (peek(TokenKind::kw_case)) {
      advance(); // consume 'case'

      // Parse case label (string literal)
      Token labelTok = peek();
      if (!expect(TokenKind::string_literal)) {
        synchronize();
        continue;
      }
      // Remove quotes from the string literal
      llvm::StringRef label = labelTok.spelling;
      if (label.size() >= 2 && label.front() == '"' && label.back() == '"') {
        label = label.drop_front().drop_back();
      }

      if (!expect(TokenKind::fat_arrow)) {
        synchronize();
        continue;
      }

      // Parse case body (single statement)
      llvm::SmallVector<std::unique_ptr<Stmt>, 4> caseBody;
      auto stmt = parseStatement();
      if (!stmt) {
        synchronize();
        continue;
      }
      caseBody.push_back(std::move(stmt));

      cases.push_back(SwitchCase(label, caseBody));

    } else if (peek(TokenKind::kw_default)) {
      advance(); // consume 'default'

      if (!expect(TokenKind::fat_arrow)) {
        synchronize();
        continue;
      }

      // Parse default body (single statement)
      auto stmt = parseStatement();
      if (!stmt) {
        synchronize();
        continue;
      }
      defaultBody.push_back(std::move(stmt));

    } else {
      emitError(getCurrentLocation(), "Expected 'case' or 'default' in switch statement");
      synchronize();
      break;
    }
  }

  if (!expect(TokenKind::r_brace)) return nullptr;

  // Parse optional result binding: } -> identifier
  std::string resultBinding;
  if (peek(TokenKind::arrow)) {
    advance();  // consume '->'
    Token identTok = peek();
    if (!expectIdentifier()) return nullptr;
    resultBinding = identTok.spelling.str();
  }

  return std::make_unique<SwitchStmt>(loc, std::move(discriminant), cases, defaultBody, resultBinding);
}
