//! Error codes for all APXM components.
//!
//! Error codes are organized by component and provide a stable identifier
//! for each error type. This enables:
//! - Documentation lookup
//! - Error categorization
//! - Automated fixes
//! - Error tracking/metrics
//!
//! Format: E<component><number>
//! - E001-E099: Parser errors
//! - E101-E199: Type errors
//! - E201-E299: MLIR/Verification errors
//! - E301-E399: Optimization errors
//! - E401-E499: Runtime errors
//! - E900-E999: Generic errors

use serde::{Deserialize, Serialize};
use std::fmt;

/// Error code prefix indicates component
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum ErrorCode {
    // ========================================================================
    // Parser Errors (E001-E099)
    // ========================================================================
    /// E001: Unexpected token
    /// Example: `expected ';' but found '}'`
    UnexpectedToken = 1,

    /// E002: Expected expression
    /// Example: `expected expression after '='`
    ExpectedExpression = 2,

    /// E003: Expected identifier
    /// Example: `expected variable name in let binding`
    ExpectedIdentifier = 3,

    /// E004: Invalid number literal
    /// Example: `invalid number: '123abc'`
    InvalidNumberLiteral = 4,

    /// E005: Invalid string literal
    /// Example: `unterminated string literal`
    InvalidStringLiteral = 5,

    /// E006: Unknown keyword
    /// Example: `unknown keyword 'unknown_keyword'`
    UnknownKeyword = 6,

    /// E007: Missing closing brace
    /// Example: `expected '}' to close '{' at line 5`
    MissingClosingBrace = 7,

    /// E008: Missing closing parenthesis
    /// Example: `expected ')' to close '(' at line 10`
    MissingClosingParen = 8,

    /// E009: Missing closing bracket
    /// Example: `expected ']' to close '[' at line 15`
    MissingClosingBracket = 9,

    /// E010: Duplicate declaration
    /// Example: `'variable_name' declared twice`
    DuplicateDeclaration = 10,

    /// E011: Expected type annotation
    /// Example: `expected type after ':' in let binding`
    ExpectedTypeAnnotation = 11,

    /// E012: Invalid memory tier
    /// Example: `'invalid_tier' is not a valid memory tier (use STM, LTM, or Episodic)`
    InvalidMemoryTier = 12,

    /// E013: Expected event type
    /// Example: `expected event type in 'on' handler`
    ExpectedEventType = 13,

    /// E014: Expected capability name
    /// Example: `expected capability name in capability declaration`
    ExpectedCapabilityName = 14,

    /// E015: Expected flow name
    /// Example: `expected flow name in flow declaration`
    ExpectedFlowName = 15,

    /// E016: Expected belief name
    /// Example: `expected belief name in beliefs block`
    ExpectedBeliefName = 16,

    /// E017: Expected goal name
    /// Example: `expected goal name in goals list`
    ExpectedGoalName = 17,

    /// E018: Expected agent name
    /// Example: `expected agent name after 'agent' keyword`
    ExpectedAgentName = 18,

    /// E019: Expected memory name
    /// Example: `expected memory store name in memory declaration`
    ExpectedMemoryName = 19,

    /// E020: Invalid operator
    /// Example: `'++' is not a valid operator`
    InvalidOperator = 20,

    /// E021: Expected function name
    /// Example: `expected function name before '('`
    ExpectedFunctionName = 21,

    /// E022: Expected member name
    /// Example: `expected member name after '.'`
    ExpectedMemberName = 22,

    /// E023: Expected array index
    /// Example: `expected index expression in '[]'`
    ExpectedArrayIndex = 23,

    /// E024: Expected parameter
    /// Example: `expected parameter in function call`
    ExpectedParameter = 24,

    /// E025: Expected return value
    /// Example: `expected return value after 'return' keyword`
    ExpectedReturnValue = 25,

    /// E026: Expected condition
    /// Example: `expected condition in 'if' statement`
    ExpectedCondition = 26,

    /// E027: Expected loop variable
    /// Example: `expected loop variable in 'loop' statement`
    ExpectedLoopVariable = 27,

    /// E028: Expected collection
    /// Example: `expected collection in 'loop' statement`
    ExpectedCollection = 28,

    /// E029: Expected code string
    /// Example: `expected code string in exec() call`
    ExpectedCodeString = 29,

    /// E030: Expected trace ID
    /// Example: `expected trace ID in reflect() call`
    ExpectedTraceId = 30,

    /// E031: Expected goal string
    /// Example: `expected goal string in plan() call`
    ExpectedGoalString = 31,

    /// E032: Expected template string
    /// Example: `expected template string in verify() call`
    ExpectedTemplateString = 32,

    /// E033: Expected recipient
    /// Example: `expected recipient in talk() call`
    ExpectedRecipient = 33,

    // ========================================================================
    // Type Errors (E101-E199)
    // ========================================================================
    /// E101: Type mismatch
    /// Example: `expected 'string' but found 'number'`
    TypeMismatch = 101,

    /// E102: Undefined variable
    /// Example: `variable 'x' is not defined`
    UndefinedVariable = 102,

    /// E103: Invalid type annotation
    /// Example: `'unknown_type' is not a valid type`
    InvalidTypeAnnotation = 103,

    /// E104: Type inference failed
    /// Example: `cannot infer type for expression`
    TypeInferenceFailed = 104,

    /// E105: Type not found
    /// Example: `type 'CustomType' is not defined`
    TypeNotFound = 105,

    /// E106: Invalid type conversion
    /// Example: `cannot convert 'number' to 'string'`
    InvalidTypeConversion = 106,

    /// E107: Type annotation required
    /// Example: `type annotation required for this expression`
    TypeAnnotationRequired = 107,

    // ========================================================================
    // MLIR/Verification Errors (E201-E299)
    // ========================================================================
    /// E201: MLIR verification failed
    /// Example: `operation 'ais.inv' has invalid operands`
    MLIRVerificationFailed = 201,

    /// E202: Invalid operation
    /// Example: `operation 'ais.unknown' is not defined`
    InvalidOperation = 202,

    /// E203: DAG cycle detected
    /// Example: `circular dependency detected in execution graph`
    DagCycleDetected = 203,

    /// E204: Missing required operand
    /// Example: `operation 'ais.rsn' requires at least one context operand`
    MissingRequiredOperand = 204,

    /// E205: Invalid operand type
    /// Example: `operand type 'number' is not valid for operation 'ais.inv'`
    InvalidOperandType = 205,

    /// E206: Operation not found
    /// Example: `operation 'custom_op' is not registered`
    OperationNotFound = 206,

    /// E207: Invalid operation result
    /// Example: `operation result type does not match expected type`
    InvalidOperationResult = 207,

    // ========================================================================
    // Optimization Errors (E301-E399)
    // ========================================================================
    /// E301: Pass execution failed
    /// Example: `pass 'normalize' failed: internal error`
    PassExecutionFailed = 301,

    /// E302: Optimization conflict
    /// Example: `cannot apply 'fuse-reasoning' and 'scheduling' together`
    OptimizationConflict = 302,

    /// E303: Pass not found
    /// Example: `pass 'unknown_pass' is not registered`
    PassNotFound = 303,

    /// E304: Pass dependency failed
    /// Example: `required pass 'dependency-analysis' failed before 'normalize'`
    PassDependencyFailed = 304,

    // ========================================================================
    // Runtime Errors (E401-E499)
    // ========================================================================
    /// E401: Scheduler error
    /// Example: `scheduler failed to execute operation`
    SchedulerError = 401,

    /// E402: Operation execution failed
    /// Example: `operation 'ais.inv' failed: capability not found`
    OperationExecutionFailed = 402,

    /// E403: Timeout
    /// Example: `operation timed out after 30s`
    Timeout = 403,

    /// E404: Capability not found
    /// Example: `capability 'search_web' is not registered`
    CapabilityNotFound = 404,

    /// E405: Memory access error
    /// Example: `cannot access memory store 'invalid_store'`
    MemoryAccessError = 405,

    /// E406: LLM backend error
    /// Example: `LLM backend 'openai' returned an error`
    LLMBackendError = 406,

    // ========================================================================
    // Generic Errors (E900-E999)
    // ========================================================================
    /// E900: Internal error
    /// Example: `internal compiler error: please report this bug`
    InternalError = 900,

    /// E901: Not implemented
    /// Example: `feature 'X' is not yet implemented`
    NotImplemented = 901,

    /// E902: Invalid configuration
    /// Example: `invalid compiler configuration: 'option' is not valid`
    InvalidConfiguration = 902,

    /// E903: IO error
    /// Example: `failed to read file: permission denied`
    IoError = 903,
}

impl ErrorCode {
    /// Get error code as string (e.g., "E001")
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::UnexpectedToken => "E001",
            ErrorCode::ExpectedExpression => "E002",
            ErrorCode::ExpectedIdentifier => "E003",
            ErrorCode::InvalidNumberLiteral => "E004",
            ErrorCode::InvalidStringLiteral => "E005",
            ErrorCode::UnknownKeyword => "E006",
            ErrorCode::MissingClosingBrace => "E007",
            ErrorCode::MissingClosingParen => "E008",
            ErrorCode::MissingClosingBracket => "E009",
            ErrorCode::DuplicateDeclaration => "E010",
            ErrorCode::ExpectedTypeAnnotation => "E011",
            ErrorCode::InvalidMemoryTier => "E012",
            ErrorCode::ExpectedEventType => "E013",
            ErrorCode::ExpectedCapabilityName => "E014",
            ErrorCode::ExpectedFlowName => "E015",
            ErrorCode::ExpectedBeliefName => "E016",
            ErrorCode::ExpectedGoalName => "E017",
            ErrorCode::ExpectedAgentName => "E018",
            ErrorCode::ExpectedMemoryName => "E019",
            ErrorCode::InvalidOperator => "E020",
            ErrorCode::ExpectedFunctionName => "E021",
            ErrorCode::ExpectedMemberName => "E022",
            ErrorCode::ExpectedArrayIndex => "E023",
            ErrorCode::ExpectedParameter => "E024",
            ErrorCode::ExpectedReturnValue => "E025",
            ErrorCode::ExpectedCondition => "E026",
            ErrorCode::ExpectedLoopVariable => "E027",
            ErrorCode::ExpectedCollection => "E028",
            ErrorCode::ExpectedCodeString => "E029",
            ErrorCode::ExpectedTraceId => "E030",
            ErrorCode::ExpectedGoalString => "E031",
            ErrorCode::ExpectedTemplateString => "E032",
            ErrorCode::ExpectedRecipient => "E033",
            ErrorCode::TypeMismatch => "E101",
            ErrorCode::UndefinedVariable => "E102",
            ErrorCode::InvalidTypeAnnotation => "E103",
            ErrorCode::TypeInferenceFailed => "E104",
            ErrorCode::TypeNotFound => "E105",
            ErrorCode::InvalidTypeConversion => "E106",
            ErrorCode::TypeAnnotationRequired => "E107",
            ErrorCode::MLIRVerificationFailed => "E201",
            ErrorCode::InvalidOperation => "E202",
            ErrorCode::DagCycleDetected => "E203",
            ErrorCode::MissingRequiredOperand => "E204",
            ErrorCode::InvalidOperandType => "E205",
            ErrorCode::OperationNotFound => "E206",
            ErrorCode::InvalidOperationResult => "E207",
            ErrorCode::PassExecutionFailed => "E301",
            ErrorCode::OptimizationConflict => "E302",
            ErrorCode::PassNotFound => "E303",
            ErrorCode::PassDependencyFailed => "E304",
            ErrorCode::SchedulerError => "E401",
            ErrorCode::OperationExecutionFailed => "E402",
            ErrorCode::Timeout => "E403",
            ErrorCode::CapabilityNotFound => "E404",
            ErrorCode::MemoryAccessError => "E405",
            ErrorCode::LLMBackendError => "E406",
            ErrorCode::InternalError => "E900",
            ErrorCode::NotImplemented => "E901",
            ErrorCode::InvalidConfiguration => "E902",
            ErrorCode::IoError => "E903",
        }
    }

    /// Get component name
    pub fn component(&self) -> &'static str {
        let code = *self as u32;
        if code < 100 {
            "parser"
        } else if code < 200 {
            "type"
        } else if code < 300 {
            "mlir"
        } else if code < 400 {
            "optimization"
        } else if code < 500 {
            "runtime"
        } else {
            "generic"
        }
    }

    /// Get documentation URL
    pub fn documentation_url(&self) -> String {
        format!("https://apxm.dev/errors/{}", self.as_str())
    }

    /// Convert from u32 (for FFI)
    pub fn from_u32(code: u32) -> Option<Self> {
        match code {
            1 => Some(ErrorCode::UnexpectedToken),
            2 => Some(ErrorCode::ExpectedExpression),
            3 => Some(ErrorCode::ExpectedIdentifier),
            4 => Some(ErrorCode::InvalidNumberLiteral),
            5 => Some(ErrorCode::InvalidStringLiteral),
            6 => Some(ErrorCode::UnknownKeyword),
            7 => Some(ErrorCode::MissingClosingBrace),
            8 => Some(ErrorCode::MissingClosingParen),
            9 => Some(ErrorCode::MissingClosingBracket),
            10 => Some(ErrorCode::DuplicateDeclaration),
            11 => Some(ErrorCode::ExpectedTypeAnnotation),
            12 => Some(ErrorCode::InvalidMemoryTier),
            13 => Some(ErrorCode::ExpectedEventType),
            14 => Some(ErrorCode::ExpectedCapabilityName),
            15 => Some(ErrorCode::ExpectedFlowName),
            16 => Some(ErrorCode::ExpectedBeliefName),
            17 => Some(ErrorCode::ExpectedGoalName),
            18 => Some(ErrorCode::ExpectedAgentName),
            19 => Some(ErrorCode::ExpectedMemoryName),
            20 => Some(ErrorCode::InvalidOperator),
            21 => Some(ErrorCode::ExpectedFunctionName),
            22 => Some(ErrorCode::ExpectedMemberName),
            23 => Some(ErrorCode::ExpectedArrayIndex),
            24 => Some(ErrorCode::ExpectedParameter),
            25 => Some(ErrorCode::ExpectedReturnValue),
            26 => Some(ErrorCode::ExpectedCondition),
            27 => Some(ErrorCode::ExpectedLoopVariable),
            28 => Some(ErrorCode::ExpectedCollection),
            29 => Some(ErrorCode::ExpectedCodeString),
            30 => Some(ErrorCode::ExpectedTraceId),
            31 => Some(ErrorCode::ExpectedGoalString),
            32 => Some(ErrorCode::ExpectedTemplateString),
            33 => Some(ErrorCode::ExpectedRecipient),
            101 => Some(ErrorCode::TypeMismatch),
            102 => Some(ErrorCode::UndefinedVariable),
            103 => Some(ErrorCode::InvalidTypeAnnotation),
            104 => Some(ErrorCode::TypeInferenceFailed),
            105 => Some(ErrorCode::TypeNotFound),
            106 => Some(ErrorCode::InvalidTypeConversion),
            107 => Some(ErrorCode::TypeAnnotationRequired),
            201 => Some(ErrorCode::MLIRVerificationFailed),
            202 => Some(ErrorCode::InvalidOperation),
            203 => Some(ErrorCode::DagCycleDetected),
            204 => Some(ErrorCode::MissingRequiredOperand),
            205 => Some(ErrorCode::InvalidOperandType),
            206 => Some(ErrorCode::OperationNotFound),
            207 => Some(ErrorCode::InvalidOperationResult),
            301 => Some(ErrorCode::PassExecutionFailed),
            302 => Some(ErrorCode::OptimizationConflict),
            303 => Some(ErrorCode::PassNotFound),
            304 => Some(ErrorCode::PassDependencyFailed),
            401 => Some(ErrorCode::SchedulerError),
            402 => Some(ErrorCode::OperationExecutionFailed),
            403 => Some(ErrorCode::Timeout),
            404 => Some(ErrorCode::CapabilityNotFound),
            405 => Some(ErrorCode::MemoryAccessError),
            406 => Some(ErrorCode::LLMBackendError),
            900 => Some(ErrorCode::InternalError),
            901 => Some(ErrorCode::NotImplemented),
            902 => Some(ErrorCode::InvalidConfiguration),
            903 => Some(ErrorCode::IoError),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_display() {
        assert_eq!(ErrorCode::UnexpectedToken.as_str(), "E001");
        assert_eq!(ErrorCode::TypeMismatch.as_str(), "E101");
        assert_eq!(ErrorCode::InternalError.as_str(), "E900");
    }

    #[test]
    fn test_error_code_component() {
        assert_eq!(ErrorCode::UnexpectedToken.component(), "parser");
        assert_eq!(ErrorCode::TypeMismatch.component(), "type");
        assert_eq!(ErrorCode::MLIRVerificationFailed.component(), "mlir");
        assert_eq!(ErrorCode::PassExecutionFailed.component(), "optimization");
        assert_eq!(ErrorCode::SchedulerError.component(), "runtime");
        assert_eq!(ErrorCode::InternalError.component(), "generic");
    }

    #[test]
    fn test_error_code_from_u32() {
        assert_eq!(ErrorCode::from_u32(1), Some(ErrorCode::UnexpectedToken));
        assert_eq!(ErrorCode::from_u32(101), Some(ErrorCode::TypeMismatch));
        assert_eq!(ErrorCode::from_u32(999), None);
    }

    #[test]
    fn test_error_code_documentation_url() {
        let url = ErrorCode::UnexpectedToken.documentation_url();
        assert!(url.contains("E001"));
        assert!(url.starts_with("https://"));
    }
}
