/*
 * @file Lexer.cpp
 * @brief APXM DSL Lexer Implementation
 *
 * This file implements the lexer for the APXM DSL, providing a high-performance
 * lexer with comprehensive error handling.
 */

#include "ais/Parser/Lexer/Lexer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Unicode.h"
#include <cctype>
#include <cerrno>
#include <cstdlib>

using namespace apxm::parser;

Lexer::Lexer(llvm::SourceMgr &srcMgr, unsigned bufferID)
    : srcMgr(srcMgr), curTok(TokenKind::unknown, "", llvm::SMLoc()) {
  auto *buffer = srcMgr.getMemoryBuffer(bufferID);
  curPtr = buffer->getBufferStart();
  bufferEnd = buffer->getBufferEnd();
  curTok = lexImpl();
}

Token Lexer::lex() {
  Token tok = curTok;
  curTok = lexImpl();
  return tok;
}

Token Lexer::lexImpl() {
  skipWhitespaceAndComments();

  if (curPtr >= bufferEnd)
    return Token(TokenKind::eof, "", getCurrentLocation());

  const char *tokStart = curPtr;
  char c = *curPtr;

  // Handle identifiers starting with alphabetic characters
  if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
    return lexIdentifierOrKeyword();

  // String literals
  if (c == '"')
    return lexStringLiteral();

  // Numeric literals
  if (std::isdigit(c) || (c == '-' && std::isdigit(peekChar(1))))
    return lexNumberLiteral();

  // Punctuation and operators
  consumeChar();
  switch (c) {
  case '{':
    return Token(TokenKind::l_brace, "{", getCurrentLocation());
  case '}':
    return Token(TokenKind::r_brace, "}", getCurrentLocation());
  case '(':
    return Token(TokenKind::l_paren, "(", getCurrentLocation());
  case ')':
    return Token(TokenKind::r_paren, ")", getCurrentLocation());
  case '[':
    return Token(TokenKind::l_square, "[", getCurrentLocation());
  case ']':
    return Token(TokenKind::r_square, "]", getCurrentLocation());
  case ',':
    return Token(TokenKind::comma, ",", getCurrentLocation());
  case ':':
    return Token(TokenKind::colon, ":", getCurrentLocation());
  case ';':
    return Token(TokenKind::semicolon, ";", getCurrentLocation());
  case '+':
    return Token(TokenKind::plus, "+", getCurrentLocation());
  case '*':
    return Token(TokenKind::star, "*", getCurrentLocation());
  case '%':
    return Token(TokenKind::percent, "%", getCurrentLocation());
  case '.':
    return Token(TokenKind::dot, ".", getCurrentLocation());

  case '-':
    if (peekChar() == '>') {
      consumeChar();
      return Token(TokenKind::arrow, "->", getCurrentLocation());
    }
    return Token(TokenKind::minus, "-", getCurrentLocation());

  case '/':
    // Check for comment start
    if (peekChar() == '/') {
      lexComment();
      return lexImpl(); // Recurse after comment
    }
    return Token(TokenKind::slash, "/", getCurrentLocation());

  case '=':
    if (peekChar() == '=') {
      consumeChar();
      return Token(TokenKind::equal_equal, "==", getCurrentLocation());
    }
    if (peekChar() == '>') {
      consumeChar();
      return Token(TokenKind::fat_arrow, "=>", getCurrentLocation());
    }
    return Token(TokenKind::equal, "=", getCurrentLocation());

  case '!':
    if (peekChar() == '=') {
      consumeChar();
      return Token(TokenKind::bang_equal, "!=", getCurrentLocation());
    }
    return Token(TokenKind::bang, "!", getCurrentLocation());

  case '&':
    if (peekChar() == '&') {
      consumeChar();
      return Token(TokenKind::amp_amp, "&&", getCurrentLocation());
    }
    break;

  case '|':
    if (peekChar() == '|') {
      consumeChar();
      return Token(TokenKind::pipe_pipe, "||", getCurrentLocation());
    }
    if (peekChar() == '>') {
      consumeChar();
      return Token(TokenKind::pipe_greater, "|>", getCurrentLocation());
    }
    break;

  case '<':
    if (peekChar() == '=') {
      consumeChar();
      return Token(TokenKind::less_equal, "<=", getCurrentLocation());
    }
    return Token(TokenKind::less, "<", getCurrentLocation());

  case '>':
    if (peekChar() == '=') {
      consumeChar();
      return Token(TokenKind::greater_equal, ">=", getCurrentLocation());
    }
    return Token(TokenKind::greater, ">", getCurrentLocation());

  case '@':
    return Token(TokenKind::at_sign, "@", getCurrentLocation());
  }

  // Report unknown character
  hasError = true;
  srcMgr.PrintMessage(getCurrentLocation(), llvm::SourceMgr::DK_Error,
                      "Unexpected character: " + std::string(1, c));
  return Token(TokenKind::unknown, llvm::StringRef(tokStart, 1), getCurrentLocation());
}

Token Lexer::lexIdentifierOrKeyword() {
  const char *tokStart = curPtr;

  // First character already checked - continue with identifier body
  while (curPtr < bufferEnd) {
    char c = *curPtr;
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      consumeChar();
    } else {
      break;
    }
  }

  llvm::StringRef spelling = getStringRef(tokStart, curPtr);

  // Fast keyword lookup using StringSwitch
  TokenKind kind = llvm::StringSwitch<TokenKind>(spelling)
      // BDI keywords
      .Case("agent", TokenKind::kw_agent)
      .Case("beliefs", TokenKind::kw_beliefs)
      .Case("context", TokenKind::kw_context)
      .Case("from", TokenKind::kw_from)
      .Case("goals", TokenKind::kw_goals)
      .Case("on", TokenKind::kw_on)
      .Case("plans", TokenKind::kw_plans)
      .Case("when", TokenKind::kw_when)
      .Case("with", TokenKind::kw_with)

      // Structure keywords
      .Case("capability", TokenKind::kw_capability)
      .Case("flow", TokenKind::kw_flow)
      .Case("memory", TokenKind::kw_memory)

      // Control flow
      .Case("case", TokenKind::kw_case)
      .Case("catch", TokenKind::kw_catch)
      .Case("default", TokenKind::kw_default)
      .Case("else", TokenKind::kw_else)
      .Case("if", TokenKind::kw_if)
      .Case("in", TokenKind::kw_in)
      .Case("let", TokenKind::kw_let)
      .Case("loop", TokenKind::kw_loop)
      .Case("parallel", TokenKind::kw_parallel)
      .Case("return", TokenKind::kw_return)
      .Case("switch", TokenKind::kw_switch)
      .Case("try", TokenKind::kw_try)

      // Operations
      .Case("ask", TokenKind::kw_ask)
      .Case("exec", TokenKind::kw_exec)
      .Case("llm", TokenKind::kw_llm)
      .Case("merge", TokenKind::kw_merge)
      .Case("mem", TokenKind::kw_mem)
      .Case("plan", TokenKind::kw_plan)
      .Case("print", TokenKind::kw_print)
      .Case("reason", TokenKind::kw_reason)
      .Case("reflect", TokenKind::kw_reflect)
      .Case("talk", TokenKind::kw_talk)
      .Case("think", TokenKind::kw_think)
      .Case("tool", TokenKind::kw_tool)
      .Case("verify", TokenKind::kw_verify)
      .Case("wait", TokenKind::kw_wait)

      // Literals and built-ins
      .Case("bool", TokenKind::kw_bool)
      .Case("false", TokenKind::kw_false)
      .Case("json", TokenKind::kw_json)
      .Case("null", TokenKind::kw_null)
      .Case("number", TokenKind::kw_number)
      .Case("string", TokenKind::kw_string)
      .Case("true", TokenKind::kw_true)

      // Memory tiers
      .Case("Episodic", TokenKind::kw_Episodic)
      .Case("LTM", TokenKind::kw_LTM)
      .Case("STM", TokenKind::kw_STM)

      // Types (note: agent/capability/context already defined above as keywords)
      .Case("goal", TokenKind::kw_goal_type)
      .Case("handle", TokenKind::kw_handle)
      .Case("response", TokenKind::kw_response)
      .Case("token", TokenKind::kw_token)
      .Case("void", TokenKind::kw_void)

      // Annotations
      .Case("entry", TokenKind::kw_entry)

      .Default(TokenKind::identifier);

  return Token(kind, spelling, llvm::SMLoc::getFromPointer(tokStart));
}

Token Lexer::lexStringLiteral() {
  const char *tokStart = curPtr;
  consumeChar(); // consume opening "

  while (curPtr < bufferEnd) {
    char c = *curPtr;

    // Handle closing quote
    if (c == '"') {
      consumeChar(); // consume closing "
      break;
    }

    // Handle escape sequences
    if (c == '\\') {
      consumeChar(); // consume backslash

      if (curPtr >= bufferEnd) {
        hasError = true;
        srcMgr.PrintMessage(getCurrentLocation(), llvm::SourceMgr::DK_Error,
                           "Unterminated escape sequence in string literal");
        break;
      }

      // Consume escaped character
      consumeChar();
      continue;
    }

    // Handle newlines in strings
    if (c == '\n' || c == '\r') {
      hasError = true;
      srcMgr.PrintMessage(llvm::SMLoc::getFromPointer(curPtr), llvm::SourceMgr::DK_Error,
                         "Newline in string literal");
    }

    consumeChar();
  }

  // Check for unterminated string
  if (curPtr >= bufferEnd) {
    hasError = true;
    srcMgr.PrintMessage(llvm::SMLoc::getFromPointer(tokStart), llvm::SourceMgr::DK_Error,
                       "Unterminated string literal");
  }

  return Token(TokenKind::string_literal, getStringRef(tokStart, curPtr),
              llvm::SMLoc::getFromPointer(tokStart));
}

Token Lexer::lexNumberLiteral() {
  const char *tokStart = curPtr;
  bool hasDecimalPoint = false;

  // Handle optional minus sign
  if (*curPtr == '-') {
    consumeChar();
  }

  // Parse digits and decimal point
  while (curPtr < bufferEnd) {
    char c = *curPtr;

    if (std::isdigit(c)) {
      consumeChar();
      continue;
    }

    if (c == '.' && !hasDecimalPoint) {
      hasDecimalPoint = true;
      consumeChar();
      continue;
    }

    break;
  }

  // Validate number format
  llvm::StringRef numStr = getStringRef(tokStart, curPtr);
  char *endPtr;
  errno = 0;
  strtod(numStr.data(), &endPtr);

  if (errno == ERANGE) {
    hasError = true;
    srcMgr.PrintMessage(llvm::SMLoc::getFromPointer(tokStart), llvm::SourceMgr::DK_Error,
                       "Number literal out of range");
  }

  return Token(TokenKind::number_literal, numStr, llvm::SMLoc::getFromPointer(tokStart));
}

void Lexer::skipWhitespaceAndComments() {
  while (curPtr < bufferEnd) {
    char c = *curPtr;

    // Handle whitespace
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      consumeChar();
      continue;
    }

    // Handle comments
    if (c == '/' && peekChar(1) == '/') {
      lexComment();
      continue;
    }

    break;
  }
}

void Lexer::lexComment() {
  // Skip "//"
  consumeChar();
  consumeChar();

  // Skip until end of line
  while (curPtr < bufferEnd && *curPtr != '\n' && *curPtr != '\r') {
    consumeChar();
  }
}

char Lexer::peekChar(int offset) const {
  const char *ptr = curPtr + offset;
  return (ptr < bufferEnd && ptr >= curPtr) ? *ptr : '\0';
}

char Lexer::consumeChar() {
  return (curPtr < bufferEnd) ? *curPtr++ : '\0';
}

llvm::SMLoc Lexer::getCurrentLocation() const {
  return llvm::SMLoc::getFromPointer(curPtr);
}

llvm::StringRef Lexer::getStringRef(const char *start, const char *end) const {
  return llvm::StringRef(start, static_cast<size_t>(end - start));
}
