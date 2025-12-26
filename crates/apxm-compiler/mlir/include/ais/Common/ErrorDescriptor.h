/*
 * @file ErrorDescriptor.h
 * @brief Detailed Error description for error codes and messages.
 *
 * Each error code is associated with a specific error message that provides
 * information about the error and how to resolve it.
 *
 * The error codes are defined in the ErrorCode enum class and must match exactly
 * with the Rust ErrorCode enum.
 */

#ifndef APXM_COMMON_ERROR_DESCRIPTOR_H
#define APXM_COMMON_ERROR_DESCRIPTOR_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace apxm {

/// Error code (matches Rust ErrorCode enum)
/// Must match exactly with Rust ErrorCode
enum class ErrorCode : uint32_t {
  // Parser Errors (E001-E099)
  UnexpectedToken = 1,
  ExpectedExpression = 2,
  ExpectedIdentifier = 3,
  InvalidNumberLiteral = 4,
  InvalidStringLiteral = 5,
  UnknownKeyword = 6,
  MissingClosingBrace = 7,
  MissingClosingParen = 8,
  MissingClosingBracket = 9,
  DuplicateDeclaration = 10,
  ExpectedTypeAnnotation = 11,
  InvalidMemoryTier = 12,
  ExpectedEventType = 13,
  ExpectedCapabilityName = 14,
  ExpectedFlowName = 15,
  ExpectedBeliefName = 16,
  ExpectedGoalName = 17,
  ExpectedAgentName = 18,
  ExpectedMemoryName = 19,
  InvalidOperator = 20,
  ExpectedFunctionName = 21,
  ExpectedMemberName = 22,
  ExpectedArrayIndex = 23,
  ExpectedParameter = 24,
  ExpectedReturnValue = 25,
  ExpectedCondition = 26,
  ExpectedLoopVariable = 27,
  ExpectedCollection = 28,
  ExpectedCodeString = 29,
  ExpectedTraceId = 30,
  ExpectedGoalString = 31,
  ExpectedTemplateString = 32,
  ExpectedRecipient = 33,
  SyntaxError = 34,

  // Type Errors (E101-E199)
  TypeMismatch = 101,
  UndefinedVariable = 102,
  InvalidTypeAnnotation = 103,
  TypeInferenceFailed = 104,
  TypeNotFound = 105,
  InvalidTypeConversion = 106,
  TypeAnnotationRequired = 107,

  // MLIR/Verification Errors (E201-E299)
  MLIRVerificationFailed = 201,
  InvalidOperation = 202,
  DagCycleDetected = 203,
  MissingRequiredOperand = 204,
  InvalidOperandType = 205,
  OperationNotFound = 206,
  InvalidOperationResult = 207,

  // Optimization Errors (E301-E399)
  PassExecutionFailed = 301,
  OptimizationConflict = 302,
  PassNotFound = 303,
  PassDependencyFailed = 304,

  // Runtime Errors (E401-E499)
  SchedulerError = 401,
  OperationExecutionFailed = 402,
  Timeout = 403,
  CapabilityNotFound = 404,
  MemoryAccessError = 405,
  LLMBackendError = 406,

  // Generic Errors (E900-E999)
  InternalError = 900,
  NotImplemented = 901,
  InvalidConfiguration = 902,
  IoError = 903,
};

/// Source span (structured)
struct SourceSpan {
  std::string file;
  uint32_t line_start;
  uint32_t col_start;
  uint32_t line_end;
  uint32_t col_end;
  std::string snippet;
  uint32_t highlight_start;
  uint32_t highlight_end;
  std::string label;
};

/// Replacement part for multi-part suggestions
struct ReplacementPart {
  SourceSpan span;
  std::string code;
  std::string label;
};

/// Code replacement suggestion
struct CodeReplacement {
  SourceSpan span;
  std::string code;
  std::string label;
  std::vector<ReplacementPart> parts;
};

/// Suggestion confidence level
enum class SuggestionConfidence : uint8_t {
  Low = 0,
  Medium = 1,
  High = 2,
};

/// Suggestion applicability
enum class SuggestionApplicability : uint8_t {
  Unspecified = 0,
  MaybeIncorrect = 1,
  HasPlaceholders = 2,
  MachineApplicable = 3,
};

/// Suggestion (structured)
struct Suggestion {
  std::string message;
  CodeReplacement replacement;
  std::string help;
  SuggestionConfidence confidence;
  SuggestionApplicability applicability;
};

/// Error descriptor (structured metadata)
struct ErrorDescriptor {
  ErrorCode code;
  std::string message;
  SourceSpan primary_span;
  std::vector<SourceSpan> secondary_spans;
  std::vector<Suggestion> suggestions;
  std::string help;
  std::vector<std::string> notes;

  /// Clear all data
  void clear() {
    secondary_spans.clear();
    suggestions.clear();
    notes.clear();
  }
};

}  // namespace apxm

#endif  // APXM_COMMON_ERROR_DESCRIPTOR_H
