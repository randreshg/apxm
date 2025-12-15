/*
 * @file ParserBase.cpp
 * @brief Base Parser Implementation
 *
 * ParserBase is the base class for all parsers in the APXM compiler.
 * It provides common functionality such as error handling and token synchronization.
 */

#include "apxm/Parser/Parsers/ParserBase.h"
#include "llvm/Support/raw_ostream.h"

using namespace apxm::parser;

void ParserBase::synchronize() noexcept {
  // Skip tokens until we find a statement boundary
  while (true) {
    if (lexer.peek().is(TokenKind::eof)) {
      return;
    }

    switch (lexer.peek().kind) {
    case TokenKind::kw_let:
    case TokenKind::kw_return:
    case TokenKind::kw_if:
    case TokenKind::kw_parallel:
    case TokenKind::kw_loop:
    case TokenKind::kw_try:
    case TokenKind::kw_catch:
    case TokenKind::r_brace: // End of block
      return;
    default:
      lexer.lex(); // Skip token
    }
  }
}

void ParserBase::emitError(Location loc, llvm::StringRef message) {
  apxm::parser::emitError(loc, ErrorCode::SyntaxError, message, lexer);
  hadError = true;
}
