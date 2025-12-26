/*
 * @file ExpressionParser.cpp
 * @brief Implementation of the expression parser.
 *
 * Expressions are parsed using a recursive descent approach.
 */

#include "ais/Parser/Parsers/ExpressionParser.h"
#include "ais/Parser/Utils/Container.h"
#include "ais/Parser/Utils/Token.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <type_traits>
#include <utility>

using namespace apxm::parser;

namespace {

bool isOperationIdentifier(TokenKind kind) noexcept {
  switch (kind) {
  case TokenKind::identifier:
  case TokenKind::kw_llm:
  case TokenKind::kw_tool:
  case TokenKind::kw_mem:
  case TokenKind::kw_think:
  case TokenKind::kw_plan:
  case TokenKind::kw_reflect:
  case TokenKind::kw_verify:
  case TokenKind::kw_exec:
  case TokenKind::kw_talk:
  case TokenKind::kw_wait:
  case TokenKind::kw_merge:
    return true;
  default:
    return false;
  }
}

} // namespace

std::unique_ptr<Expr> ExpressionParser::parseExpression() {
  return parsePipelineExpr();
}

std::unique_ptr<Expr> ExpressionParser::parseAssignmentExpr() {
  auto expr = parseLogicalOrExpr();
  if (!expr) return nullptr;

  if (peek(TokenKind::equal)) {
    Location loc = getCurrentLocation();
    advance(); // consume '='

    auto rhs = parseAssignmentExpr();
    if (!rhs) {
      synchronize();
      return expr;
    }

    return std::make_unique<AssignmentExpr>(loc, std::move(expr), std::move(rhs));
  }

  return expr;
}

std::unique_ptr<Expr> ExpressionParser::parseLogicalOrExpr() {
  return parseBinaryExpr(&ExpressionParser::parseLogicalAndExpr,
      [](TokenKind kind) { return kind == TokenKind::pipe_pipe; },
      [](TokenKind) { return BinaryExpr::Operator::Or; });
}

std::unique_ptr<Expr> ExpressionParser::parseLogicalAndExpr() {
  return parseBinaryExpr(&ExpressionParser::parseEqualityExpr,
      [](TokenKind kind) { return kind == TokenKind::amp_amp; },
      [](TokenKind) { return BinaryExpr::Operator::And; });
}

std::unique_ptr<Expr> ExpressionParser::parseEqualityExpr() {
  return parseBinaryExpr(&ExpressionParser::parseComparisonExpr,
      [](TokenKind kind) { return kind == TokenKind::equal_equal || kind == TokenKind::bang_equal; },
      [](TokenKind kind) {
        return kind == TokenKind::equal_equal ? BinaryExpr::Operator::Equal
                                              : BinaryExpr::Operator::NotEqual;
      });
}

std::unique_ptr<Expr> ExpressionParser::parseComparisonExpr() {
  return parseBinaryExpr(&ExpressionParser::parseAdditiveExpr,
      [](TokenKind kind) {
        return kind == TokenKind::less || kind == TokenKind::less_equal ||
               kind == TokenKind::greater || kind == TokenKind::greater_equal;
      },
      [](TokenKind kind) {
        switch (kind) {
        case TokenKind::less: return BinaryExpr::Operator::Less;
        case TokenKind::less_equal: return BinaryExpr::Operator::LessEqual;
        case TokenKind::greater: return BinaryExpr::Operator::Greater;
        case TokenKind::greater_equal: return BinaryExpr::Operator::GreaterEqual;
        default: return BinaryExpr::Operator::Less; // unreachable
        }
      });
}

std::unique_ptr<Expr> ExpressionParser::parseAdditiveExpr() {
  return parseBinaryExpr(&ExpressionParser::parseMultiplicativeExpr,
      [](TokenKind kind) { return kind == TokenKind::plus || kind == TokenKind::minus; },
      [](TokenKind kind) {
        return kind == TokenKind::plus ? BinaryExpr::Operator::Add
                                       : BinaryExpr::Operator::Sub;
      });
}

std::unique_ptr<Expr> ExpressionParser::parseMultiplicativeExpr() {
  return parseBinaryExpr(&ExpressionParser::parseUnaryExpr,
      [](TokenKind kind) { return kind == TokenKind::star || kind == TokenKind::slash || kind == TokenKind::percent; },
      [](TokenKind kind) {
        switch (kind) {
        case TokenKind::star: return BinaryExpr::Operator::Mul;
        case TokenKind::slash: return BinaryExpr::Operator::Div;
        case TokenKind::percent: return BinaryExpr::Operator::Mod;
        default: return BinaryExpr::Operator::Mul; // unreachable
        }
      });
}

std::unique_ptr<Expr> ExpressionParser::parseUnaryExpr() {
  if (peek(TokenKind::minus) || peek(TokenKind::bang)) {
    Location loc = getCurrentLocation();
    TokenKind opKind = peek().kind;
    advance(); // consume operator

    auto operand = parseUnaryExpr();
    if (!operand) {
      synchronize();
      return nullptr;
    }

    UnaryExpr::Operator op = opKind == TokenKind::minus
                           ? UnaryExpr::Operator::Negate
                           : UnaryExpr::Operator::Not;

    return std::make_unique<UnaryExpr>(loc, op, std::move(operand));
  }

  return parsePrimaryExpr();
}

std::unique_ptr<Expr> ExpressionParser::parsePrimaryExpr() {
  Location loc = getCurrentLocation();
  std::unique_ptr<Expr> base;

  if (isOperationIdentifier(peek().kind)) {
    llvm::StringRef name = peek().spelling;
    advance();
    base = std::make_unique<VarExpr>(loc, name);
  } else if (peek(TokenKind::identifier)) {
    llvm::StringRef name = peek().spelling;
    advance();
    base = std::make_unique<VarExpr>(loc, name);
  } else if (peek(TokenKind::string_literal)) {
    auto value = getStringValue(peek());
    if (!value) {
      emitError(loc, "Invalid string literal");
      return nullptr;
    }
    llvm::StringRef strValue = *value;
    advance(); // consume string literal
    base = std::make_unique<StringLiteralExpr>(loc, strValue);
  } else if (peek(TokenKind::number_literal)) {
    auto value = getNumericValue(peek());
    if (!value) {
      emitError(loc, "Invalid number literal");
      return nullptr;
    }
    double numValue = *value;
    advance(); // consume number literal
    base = std::make_unique<NumberLiteralExpr>(loc, numValue);
  } else if (isBooleanLiteral(peek().kind)) {
    bool boolValue = getBooleanValue(peek().kind);
    advance(); // consume boolean literal
    base = std::make_unique<BooleanLiteralExpr>(loc, boolValue);
  } else if (peek(TokenKind::kw_null)) {
    advance(); // consume null
    base = std::make_unique<NullLiteralExpr>(loc);
  } else if (peek(TokenKind::l_paren)) {
    advance(); // consume '('
    auto expr = parseExpression();
    if (!expr) {
      synchronize();
      return nullptr;
    }
    if (!expect(TokenKind::r_paren)) {
      synchronize();
      return expr;
    }
    base = std::move(expr);
  } else if (peek(TokenKind::l_square)) {
    base = parseArrayExpr();
  } else {
    emitError(loc, "Expected expression");
    synchronize();
    return nullptr;
  }

  if (!base)
    return nullptr;

  return parsePostfixExpr(std::move(base));
}

std::unique_ptr<Expr> ExpressionParser::parseArrayExpr() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::l_square)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Expr>, 4> elements;
  if (!parseExpressionList(TokenKind::r_square, elements)) {
    return nullptr;
  }

  return std::make_unique<ArrayExpr>(loc, elements);
}

std::unique_ptr<Expr> ExpressionParser::parsePostfixExpr(std::unique_ptr<Expr> base) {
  while (true) {
    if (peek(TokenKind::l_paren)) {
      auto *var = llvm::dyn_cast<VarExpr>(base.get());
      if (!var) {
        emitError(getCurrentLocation(), "Call target must be an identifier");
        synchronize();
        return nullptr;
      }
      auto callLoc = var->getLocation();
      auto callee = var->getName().str();
      base = parseCallExpr(callee, callLoc);
      if (!base) return nullptr;
      continue;
    }

    if (peek(TokenKind::l_square)) {
      Location loc = getCurrentLocation();
      advance(); // consume '['

      auto index = parseExpression();
      if (!index) {
        synchronize();
        return nullptr;
      }

      if (!expect(TokenKind::r_square)) {
        synchronize();
        return nullptr;
      }

      base = std::make_unique<SubscriptExpr>(loc, std::move(base), std::move(index));
      continue;
    }

    if (peek(TokenKind::dot)) {
      advance(); // consume '.'

      Location loc = getCurrentLocation();
      Token memberTok = peek();
      if (!expect(TokenKind::identifier)) {
        synchronize();
        return nullptr;
      }

      llvm::StringRef member = memberTok.spelling;
      base = std::make_unique<MemberAccessExpr>(loc, std::move(base), member);
      continue;
    }

    break;
  }

  return base;
}

std::unique_ptr<Expr> ExpressionParser::parseCallExpr(llvm::StringRef callee, Location loc) {
  if (!expect(TokenKind::l_paren)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Expr>, 4> args;
  if (!parseExpressionList(TokenKind::r_paren, args)) {
    return nullptr;
  }

  return std::make_unique<CallExpr>(loc, callee, args);
}

bool ExpressionParser::parseExpressionList(TokenKind endToken,
                                         llvm::SmallVectorImpl<std::unique_ptr<Expr>> &storage) {
  while (!peek(endToken) && !peek(TokenKind::eof)) {
    auto expr = parseExpression();
    if (!expr) {
      synchronize();
      return false;
    }
    storage.push_back(std::move(expr));

    if (!peek(endToken) && !consume(TokenKind::comma)) {
      emitError(getCurrentLocation(), "Expected ',' or '" +
               std::to_string(static_cast<char>(endToken)) + "'");
      return false;
    }
  }

  return expect(endToken);
}

std::unique_ptr<Expr> ExpressionParser::parsePipelineExpr() {
  auto expr = parseAssignmentExpr();
  if (!expr) return nullptr;

  while (peek(TokenKind::pipe_greater)) {
    Location pipeLoc = getCurrentLocation();
    advance(); // consume '|>'

    auto stage = parsePrimaryExpr();
    if (!stage) {
      synchronize();
      return expr;
    }

    expr = desugarPipeline(pipeLoc, std::move(expr), std::move(stage));
  }

  return expr;
}

std::unique_ptr<Expr> ExpressionParser::desugarPipeline(Location pipeLoc,
                                                     std::unique_ptr<Expr> value,
                                                     std::unique_ptr<Expr> stage) {
  // Pipeline: value |> stage
  // Desugars to: stage(value)

  if (llvm::isa<CallExpr>(stage.get())) {
    // If stage is already a call, insert value as first argument
    std::unique_ptr<CallExpr> stageCall(static_cast<CallExpr *>(stage.release()));
    auto existingArgs = stageCall->takeArgs();

    llvm::SmallVector<std::unique_ptr<Expr>, 4> newArgs;
    newArgs.push_back(std::move(value));
    for (auto &arg : existingArgs) {
      newArgs.push_back(std::move(arg));
    }
    return std::make_unique<CallExpr>(pipeLoc, stageCall->getCallee(), newArgs);
  } else if (auto *var = llvm::dyn_cast<VarExpr>(stage.get())) {
    // If stage is a variable, call it with value
    llvm::SmallVector<std::unique_ptr<Expr>, 1> args;
    args.push_back(std::move(value));
    return std::make_unique<CallExpr>(pipeLoc, var->getName(), args);
  } else {
    emitError(pipeLoc, "Pipeline stage must be a function or variable");
    return nullptr;
  }
}

template <typename MatchFn, typename OpFn>
std::unique_ptr<Expr>
ExpressionParser::parseBinaryExpr(std::unique_ptr<Expr> (ExpressionParser::*parseNext)(),
                               MatchFn matchOp,
                               OpFn getOpType) {
  using OpType = std::invoke_result_t<OpFn, TokenKind>;
  auto left = (this->*parseNext)();
  if (!left) return nullptr;

  while (matchOp(peek().kind)) {
    TokenKind opKind = peek().kind;
    Location loc = getCurrentLocation();
    advance(); // consume operator

    auto right = (this->*parseNext)();
    if (!right) {
      synchronize();
      return left;
    }

    if constexpr (std::is_same_v<OpType, BinaryExpr::Operator>) {
      auto op = getOpType(opKind);
      left = std::make_unique<BinaryExpr>(loc, op, std::move(left), std::move(right));
    } else {
      // Handle other binary operators if needed
      left = std::make_unique<BinaryExpr>(loc, getOpType(opKind), std::move(left), std::move(right));
    }
  }

  return left;
}
