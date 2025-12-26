/*
 * @file ParserBase.h
 * @brief Base class for all parsers providing common token manipulation utilities
 *
 * This file contains the base class for all parsers in the APXM DSL.
 * It provides common token manipulation utilities such as peeking, consuming,
 * and advancing tokens, along with error handling functionality.
 */

#ifndef APXM_PARSER_PARSERS_PARSERBASE_H
#define APXM_PARSER_PARSERS_PARSERBASE_H

#include "ais/Common/ErrorDescriptor.h"
#include "ais/Parser/Lexer/Lexer.h"
#include "ais/Parser/Utils/Location.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <string>

namespace apxm {
enum class ErrorCode : uint32_t;
struct ErrorDescriptor;
}

namespace apxm::parser {

/// Convert token kind to string representation for diagnostics
[[nodiscard]] inline llvm::StringRef tokenKindToString(TokenKind kind) noexcept {
  switch (kind) {
  case TokenKind::kw_agent: return "agent";
  case TokenKind::kw_beliefs: return "beliefs";
  case TokenKind::kw_goals: return "goals";
  case TokenKind::kw_plans: return "plans";
  case TokenKind::kw_on: return "on";
  case TokenKind::kw_when: return "when";
  case TokenKind::kw_with: return "with";
  case TokenKind::kw_from: return "from";
  case TokenKind::kw_context: return "context";
  case TokenKind::kw_memory: return "memory";
  case TokenKind::kw_capability: return "capability";
  case TokenKind::kw_flow: return "flow";
  case TokenKind::kw_let: return "let";
  case TokenKind::kw_if: return "if";
  case TokenKind::kw_else: return "else";
  case TokenKind::kw_return: return "return";
  case TokenKind::kw_parallel: return "parallel";
  case TokenKind::kw_loop: return "loop";
  case TokenKind::kw_in: return "in";
  case TokenKind::kw_try: return "try";
  case TokenKind::kw_catch: return "catch";
  case TokenKind::kw_llm: return "llm";
  case TokenKind::kw_tool: return "tool";
  case TokenKind::kw_mem: return "mem";
  case TokenKind::kw_think: return "think";
  case TokenKind::kw_plan: return "plan";
  case TokenKind::kw_reflect: return "reflect";
  case TokenKind::kw_verify: return "verify";
  case TokenKind::kw_exec: return "exec";
  case TokenKind::kw_talk: return "talk";
  case TokenKind::kw_wait: return "wait";
  case TokenKind::kw_merge: return "merge";
  case TokenKind::kw_true: return "true";
  case TokenKind::kw_false: return "false";
  case TokenKind::kw_null: return "null";
  case TokenKind::kw_STM: return "STM";
  case TokenKind::kw_LTM: return "LTM";
  case TokenKind::kw_Episodic: return "Episodic";
  case TokenKind::kw_string: return "string";
  case TokenKind::kw_number: return "number";
  case TokenKind::kw_bool: return "bool";
  case TokenKind::kw_token: return "token";
  case TokenKind::kw_json: return "json";
  case TokenKind::kw_agent_type: return "agent";
  case TokenKind::kw_capability_type: return "capability";
  case TokenKind::kw_goal_type: return "goal";
  case TokenKind::kw_handle: return "handle";
  case TokenKind::kw_result: return "result";
  case TokenKind::kw_response: return "response";
  case TokenKind::kw_context_type: return "context";
  case TokenKind::kw_void: return "void";
  case TokenKind::identifier: return "identifier";
  case TokenKind::string_literal: return "string literal";
  case TokenKind::number_literal: return "number literal";
  case TokenKind::l_brace: return "{";
  case TokenKind::r_brace: return "}";
  case TokenKind::l_paren: return "(";
  case TokenKind::r_paren: return ")";
  case TokenKind::l_square: return "[";
  case TokenKind::r_square: return "]";
  case TokenKind::comma: return ",";
  case TokenKind::colon: return ":";
  case TokenKind::semicolon: return ";";
  case TokenKind::arrow: return "->";
  case TokenKind::fat_arrow: return "=>";
  case TokenKind::equal: return "=";
  case TokenKind::dot: return ".";
  case TokenKind::plus: return "+";
  case TokenKind::minus: return "-";
  case TokenKind::star: return "*";
  case TokenKind::slash: return "/";
  case TokenKind::percent: return "%";
  case TokenKind::bang: return "!";
  case TokenKind::amp_amp: return "&&";
  case TokenKind::pipe_pipe: return "||";
  case TokenKind::pipe_greater: return "|>";
  case TokenKind::equal_equal: return "==";
  case TokenKind::bang_equal: return "!=";
  case TokenKind::less: return "<";
  case TokenKind::less_equal: return "<=";
  case TokenKind::greater: return ">";
  case TokenKind::greater_equal: return ">=";
  case TokenKind::eof: return "end of file";
  case TokenKind::unknown: return "unknown";
  }
  return "invalid token";
}

/// Create error descriptor from location and message
ErrorDescriptor createErrorDescriptor(const Location &loc, ErrorCode code, llvm::StringRef message,
                                      class llvm::SourceMgr &srcMgr);

/// Emit error to lexer and error collector
void emitError(const Location &loc, ErrorCode code, const llvm::Twine &message, Lexer &lexer);

/// Base class for all parsers providing common token manipulation utilities
class ParserBase {
protected:
  Lexer &lexer;
  bool &hadError;

  ParserBase(Lexer &lexer, bool &hadError) noexcept
      : lexer(lexer), hadError(hadError) {}

  /// Get current token location
  [[nodiscard]] Location getCurrentLocation() const noexcept {
    return Location(peek().location);
  }

  /// Peek at current token without consuming
  [[nodiscard]] const Token &peek() const noexcept {
    return lexer.peek();
  }

  /// Check if current token matches expected kind
  [[nodiscard]] bool peek(TokenKind kind) const noexcept {
    return peek().is(kind);
  }

  /// Consume token if it matches expected kind
  [[nodiscard]] bool consume(TokenKind kind) noexcept {
    if (!peek(kind))
      return false;
    (void)lexer.lex();
    return true;
  }

  /// Expect and consume token, report error if not found
  [[nodiscard]] bool expect(TokenKind kind) {
    if (consume(kind))
      return true;

    emitError(getCurrentLocation(), "Expected " + std::string(tokenKindToString(kind)));
    return false;
  }

  /// Advance to next token and return previous token
  Token advance() noexcept {
    return lexer.lex();
  }

  /// Synchronize parser after error by skipping to next statement boundary
  void synchronize() noexcept;

  /// Emit error with location and message
  void emitError(Location loc, llvm::StringRef message);
};

} // namespace apxm::parser

#endif // APXM_PARSER_PARSERS_PARSERBASE_H
