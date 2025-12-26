/*
 * @file Error.cpp
 * @brief Implementation of error handling utilities.
 *
 * Errors are represented as ErrorDescriptor objects, which contain information
 * about the error's location, code, message, and source snippet.
 */

#include "ais/Parser/Parsers/ParserBase.h"
#include "ais/Parser/Lexer/Lexer.h"
#include "ais/CAPI/Error.h"
#include "llvm/ADT/SmallString.h"
#include <cstring>

namespace {

void pushErrorDescriptor(const apxm::ErrorDescriptor &desc) {
  const auto &span = desc.primary_span;
  apxm_error_collector_add(static_cast<uint32_t>(desc.code),
                           desc.message.c_str(),
                           span.file.c_str(),
                           span.line_start,
                           span.col_start,
                           span.line_end,
                           span.col_end,
                           span.snippet.c_str(),
                           span.highlight_start,
                           span.highlight_end,
                           span.label.c_str(),
                           desc.help.c_str(),
                           desc.suggestions.size(),
                           desc.secondary_spans.size(),
                           desc.notes.size());
}

} // namespace

namespace apxm::parser {

    ErrorDescriptor createErrorDescriptor(const Location &loc, ErrorCode code, llvm::StringRef message,
                                          llvm::SourceMgr &srcMgr) {
      ErrorDescriptor desc;
      desc.code = code;
      desc.message = message;

      // Extract file, line, column from Location
      auto [line, col] = srcMgr.getLineAndColumn(loc.getStart());
      unsigned bufferID = srcMgr.FindBufferContainingLoc(loc.getStart());
      auto buffer = srcMgr.getMemoryBuffer(bufferID);
      llvm::StringRef filename = buffer->getBufferIdentifier();

      desc.primary_span.file = filename;
      desc.primary_span.line_start = line;
      desc.primary_span.col_start = col;
      desc.primary_span.line_end = line;
      desc.primary_span.col_end = col + 1;

      // Extract source snippet
      if (line > 0 && line <= buffer->getBuffer().count('\n') + 1) {
        llvm::SmallString<256> lineText;
        llvm::StringRef bufferStr = buffer->getBuffer();
        const char *lineStart = bufferStr.data();
        const char *lineEnd = bufferStr.data() + bufferStr.size();

        // Find the line
        for (unsigned i = 1; i < line && lineStart < lineEnd; ++i) {
          lineStart = (const char *)memchr(lineStart, '\n', lineEnd - lineStart);
          if (lineStart) {
            lineStart++;  // Skip newline
          } else {
            break;
          }
        }

        // Find end of line
        const char *lineEndPtr = (const char *)memchr(lineStart, '\n', lineEnd - lineStart);
        if (!lineEndPtr) {
          lineEndPtr = lineEnd;
        }

        if (lineStart < lineEndPtr) {
          llvm::StringRef lineStr(lineStart, lineEndPtr - lineStart);
          desc.primary_span.snippet = lineStr.str();
          desc.primary_span.highlight_start = col - 1;
          desc.primary_span.highlight_end = col;
        }
      }

      // Add suggestions based on error code
      switch (code) {
      case ErrorCode::ExpectedExpression: {
        Suggestion sugg;
        sugg.message = "Add an expression here";
        sugg.replacement.span = desc.primary_span;
        sugg.replacement.code = "expression";
        sugg.confidence = SuggestionConfidence::High;
        sugg.applicability = SuggestionApplicability::HasPlaceholders;
        desc.suggestions.push_back(sugg);
        break;
      }
      case ErrorCode::ExpectedIdentifier: {
        Suggestion sugg;
        sugg.message = "Add a variable or function name";
        sugg.replacement.span = desc.primary_span;
        sugg.replacement.code = "name";
        sugg.confidence = SuggestionConfidence::High;
        sugg.applicability = SuggestionApplicability::HasPlaceholders;
        desc.suggestions.push_back(sugg);
        break;
      }
      case ErrorCode::UnexpectedToken: {
        Suggestion sugg;
        sugg.message = "Check the syntax around this token";
        sugg.confidence = SuggestionConfidence::Medium;
        sugg.applicability = SuggestionApplicability::Unspecified;
        desc.suggestions.push_back(sugg);
        break;
      }
      default:
        break;
      }

      return desc;
    }

    void emitError(const Location &loc, ErrorCode code, const llvm::Twine &message, Lexer &lexer) {
      auto &srcMgr = lexer.getSourceMgr();

      // Convert Twine to string
      llvm::SmallString<256> msgStr;
      message.toVector(msgStr);

      // Create error descriptor
      ErrorDescriptor desc = createErrorDescriptor(loc, code, llvm::StringRef(msgStr), srcMgr);

      // Add to ErrorCollector
      pushErrorDescriptor(desc);
    }

}; // namespace apxm::parser
