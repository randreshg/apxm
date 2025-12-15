/*
 * @file TokenUtils.h
 * @brief Token Manipulation Utilities
 *
 * This file contains the following utilities for token handling and validation:
 * - isValidIdentifier: Check if token is a valid identifier
 * - getStringValue: Extract string value from string literal token
 * - getNumericValue: Extract numeric value from number literal token
 * - isBooleanLiteral: Check if token represents a boolean literal
 * - getBooleanValue: Get boolean value from boolean literal token
 */

#ifndef APXM_PARSER_UTILS_TOKEN_H
#define APXM_PARSER_UTILS_TOKEN_H

#include "apxm/Parser/Lexer/Lexer.h"
#include "llvm/ADT/StringSwitch.h"
#include <optional>
#include <string>
#include <cstdlib>

namespace apxm::parser {

/// Check if token is a valid identifier
[[nodiscard]] bool isValidIdentifier(const Token &tok) noexcept;

/// Extract string value from string literal token
[[nodiscard]] std::optional<std::string> getStringValue(const Token &tok) noexcept;

/// Extract numeric value from number literal token
[[nodiscard]] std::optional<double> getNumericValue(const Token &tok) noexcept;

/// Check if token represents a boolean literal
[[nodiscard]] bool isBooleanLiteral(TokenKind kind) noexcept;

/// Get boolean value from boolean literal token
[[nodiscard]] bool getBooleanValue(TokenKind kind) noexcept;

} // namespace apxm::parser

inline bool apxm::parser::isValidIdentifier(const Token &tok) noexcept {
  return tok.kind == TokenKind::identifier;
}

inline std::optional<std::string>
apxm::parser::getStringValue(const Token &tok) noexcept {
  if (tok.kind != TokenKind::string_literal)
    return std::nullopt;

  llvm::StringRef spelling = tok.spelling;
  if (spelling.size() >= 2 && spelling.front() == '"' && spelling.back() == '"') {
    spelling = spelling.drop_front().drop_back();
  }

  std::string result;
  result.reserve(spelling.size());
  for (size_t i = 0; i < spelling.size(); ++i) {
    char c = spelling[i];
    if (c == '\\' && i + 1 < spelling.size()) {
      char next = spelling[++i];
      switch (next) {
      case 'n': result.push_back('\n'); break;
      case 'r': result.push_back('\r'); break;
      case 't': result.push_back('\t'); break;
      case '\\': result.push_back('\\'); break;
      case '"': result.push_back('"'); break;
      default:
        result.push_back(next);
        break;
      }
    } else {
      result.push_back(c);
    }
  }
  return result;
}

inline std::optional<double>
apxm::parser::getNumericValue(const Token &tok) noexcept {
  if (tok.kind != TokenKind::number_literal)
    return std::nullopt;

  std::string spelling = tok.spelling.str();
  char *end = nullptr;
  double value = std::strtod(spelling.c_str(), &end);
  if (!end || end != spelling.c_str() + spelling.size())
    return std::nullopt;
  return value;
}

inline bool apxm::parser::isBooleanLiteral(TokenKind kind) noexcept {
  return kind == TokenKind::kw_true || kind == TokenKind::kw_false;
}

inline bool apxm::parser::getBooleanValue(TokenKind kind) noexcept {
  return kind == TokenKind::kw_true;
}

#endif // APXM_PARSER_UTILS_TOKEN_H
