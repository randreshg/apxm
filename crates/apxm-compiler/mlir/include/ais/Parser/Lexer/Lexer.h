/*
 * @file Lexer.h
 * @brief Gets the tokens from the APXM DSL source code and provides a stream of tokens for parsing.
 *
 * This file defines the Lexer class, which is responsible for tokenizing the APXM DSL source code.
 * It provides a stream of tokens that can be used by the parser to construct an abstract syntax tree.
 */
#ifndef APXM_PARSER_LEXER_H
#define APXM_PARSER_LEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

namespace apxm::parser {

class Lexer;
struct Token;

/// Token types for the APXM DSL with LLVM-style RTTI support
enum class TokenKind {
  // BDI keywords (alphabetical)
  kw_agent,
  kw_beliefs,
  kw_context,
  kw_from,
  kw_goals,
  kw_on,
  kw_plans,
  kw_when,
  kw_with,

  // Structure keywords (alphabetical)
  kw_capability,
  kw_flow,
  kw_memory,

  // Control flow keywords (alphabetical)
  kw_case,
  kw_catch,
  kw_default,
  kw_else,
  kw_if,
  kw_in,
  kw_let,
  kw_loop,
  kw_parallel,
  kw_return,
  kw_switch,
  kw_try,

  // Short operations (alphabetical)
  kw_ask,      // Simple Q&A (no extended thinking) - LOW latency
  kw_exec,     // Execute code
  kw_llm,      // LLM call
  kw_merge,    // Merge results
  kw_mem,      // Memory access
  kw_plan,     // Planning
  kw_print,    // Print output
  kw_reason,   // Structured reasoning with beliefs/goals - MEDIUM latency
  kw_reflect,  // Reflection
  kw_talk,     // Communicate
  kw_think,    // Extended thinking with budget - HIGH latency
  kw_tool,     // Tool invocation
  kw_verify,   // Verification
  kw_wait,     // Wait for completion

  // Literals and built-ins
  kw_bool,
  kw_false,
  kw_json,
  kw_null,
  kw_number,
  kw_string,
  kw_true,

  // Memory tiers
  kw_Episodic,
  kw_LTM,
  kw_STM,

  // Types (alphabetical)
  kw_agent_type,
  kw_capability_type,
  kw_context_type,
  kw_goal_type,
  kw_handle,
  kw_response,
  kw_result,
  kw_token,
  kw_void,

  // Annotations
  at_sign,      // @
  kw_entry,     // entry (annotation keyword)

  // Punctuation
  arrow,        // ->
  bang,         // !
  bang_equal,   // !=
  colon,        // :
  comma,        // ,
  dot,          // .
  equal,        // =
  equal_equal,  // ==
  fat_arrow,    // =>
  greater,      // >
  greater_equal,// >=
  l_brace,      // {
  l_paren,      // (
  l_square,     // [
  less,         // <
  less_equal,   // <=
  minus,        // -
  percent,      // %
  pipe_greater, // |>
  pipe_pipe,    // ||
  plus,         // +
  r_brace,      // }
  r_paren,      // )
  r_square,     // ]
  semicolon,    // ;
  slash,        // /
  star,         // *
  amp_amp,      // &&

  // Literals
  identifier,
  number_literal,
  string_literal,

  // Special
  eof,
  unknown
};

/// Token structure with location tracking
struct Token {
  TokenKind kind;
  llvm::StringRef spelling;
  llvm::SMLoc location;

  Token(TokenKind kind, llvm::StringRef spelling, llvm::SMLoc location)
      : kind(kind), spelling(spelling), location(location) {}

  /// Check if token matches a specific kind
  bool is(TokenKind k) const { return kind == k; }

  /// Check if token matches any of the provided kinds
  template <typename... Kinds>
  bool isOneOf(TokenKind k, Kinds... kinds) const {
    if (kind == k)
      return true;
    return isOneOf(kinds...);
  }

  /// Base case for isOneOf recursion
  bool isOneOf() const { return false; }
};

/// Lexer for APXM DSL with strict error handling
class Lexer {
public:
  /// Create lexer for a buffer in the source manager
  Lexer(llvm::SourceMgr &srcMgr, unsigned bufferID);

  /// Get next token, consuming it from the stream
  Token lex();

  /// Peek at next token without consuming it
  [[nodiscard]] const Token &peek() const { return curTok; }

  /// Get source manager for diagnostics
  llvm::SourceMgr &getSourceMgr() { return srcMgr; }

  /// Check if lexer has encountered errors
  bool hasErrors() const { return hasError; }

private:
  llvm::SourceMgr &srcMgr;
  const char *curPtr;
  const char *bufferEnd;
  Token curTok;
  bool hasError = false;

  /// Main lexing implementation
  Token lexImpl();

  /// Lex identifier or keyword
  Token lexIdentifierOrKeyword();

  /// Lex string literal with proper escaping
  Token lexStringLiteral();

  /// Lex numeric literal with validation
  Token lexNumberLiteral();

  /// Skip whitespace and comments
  void skipWhitespaceAndComments();

  /// Lex comment (single-line)
  void lexComment();

  /// Get current character safely
  [[nodiscard]] char peekChar(int offset = 0) const;

  /// Consume next character
  char consumeChar();

  /// Get location at current pointer
  [[nodiscard]] llvm::SMLoc getCurrentLocation() const;

  /// Get string ref between two pointers
  [[nodiscard]] llvm::StringRef getStringRef(const char *start, const char *end) const;
};

} // namespace apxm::parser

#endif // APXM_PARSER_LEXER_H
