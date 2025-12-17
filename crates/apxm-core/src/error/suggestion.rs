//! Suggestions for fixing errors.
//!
//! Provides actionable suggestions like Rust compiler:
//! - Code replacements
//! - Help text
//! - Multiple options

use crate::error::span::Span;
use serde::{Deserialize, Serialize};

/// A suggestion for fixing an error
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Suggestion {
    /// Message explaining the suggestion
    pub message: String,
    /// Code replacement (if applicable)
    pub replacement: Option<CodeReplacement>,
    /// Help text (educational content)
    pub help: Option<String>,
    /// Confidence level
    pub confidence: SuggestionConfidence,
    /// Applicable only if this condition is met
    pub applicability: SuggestionApplicability,
}

/// Confidence level for suggestions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionConfidence {
    /// Very likely correct
    High,
    /// Probably correct
    Medium,
    /// Might be correct
    Low,
}

/// When the suggestion is applicable
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionApplicability {
    /// Always applicable
    Unspecified,
    /// Only if the user has permission/access
    MaybeIncorrect,
    /// Has placeholders that need to be filled
    HasPlaceholders,
    /// Machine applicable (can be auto-fixed)
    MachineApplicable,
}

/// Code replacement suggestion
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeReplacement {
    /// Span to replace
    pub span: Span,
    /// Replacement code
    pub code: String,
    /// Label for the replacement
    pub label: Option<String>,
    /// Whether this is a multi-part replacement
    pub parts: Vec<ReplacementPart>,
}

/// Part of a multi-part replacement
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplacementPart {
    /// Span for this part
    pub span: Span,
    /// Code to insert at this span
    pub code: String,
    /// Label for this part
    pub label: Option<String>,
}

impl Suggestion {
    /// Create a simple suggestion with just a message
    pub fn new(message: String) -> Self {
        Suggestion {
            message,
            replacement: None,
            help: None,
            confidence: SuggestionConfidence::Medium,
            applicability: SuggestionApplicability::Unspecified,
        }
    }

    /// Create a suggestion with code replacement
    pub fn with_replacement(message: String, span: Span, code: String) -> Self {
        Suggestion {
            message,
            replacement: Some(CodeReplacement {
                span,
                code,
                label: None,
                parts: Vec::new(),
            }),
            help: None,
            confidence: SuggestionConfidence::High,
            applicability: SuggestionApplicability::MachineApplicable,
        }
    }

    /// Add help text
    pub fn with_help(mut self, help: String) -> Self {
        self.help = Some(help);
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: SuggestionConfidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set applicability
    pub fn with_applicability(mut self, applicability: SuggestionApplicability) -> Self {
        self.applicability = applicability;
        self
    }

    /// Add a label to the replacement
    pub fn with_replacement_label(mut self, label: String) -> Self {
        if let Some(ref mut repl) = self.replacement {
            repl.label = Some(label);
        }
        self
    }

    /// Create a multi-part replacement
    pub fn with_multi_part_replacement(mut self, parts: Vec<ReplacementPart>) -> Self {
        if let Some(ref mut repl) = self.replacement {
            repl.parts = parts;
        }
        self
    }
}

impl CodeReplacement {
    /// Create a simple replacement
    pub fn new(span: Span, code: String) -> Self {
        CodeReplacement {
            span,
            code,
            label: None,
            parts: Vec::new(),
        }
    }

    /// Add a label
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Add replacement parts
    pub fn with_parts(mut self, parts: Vec<ReplacementPart>) -> Self {
        self.parts = parts;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggestion_new() {
        let sugg = Suggestion::new("Try adding a semicolon".to_string());
        assert_eq!(sugg.message, "Try adding a semicolon");
        assert!(sugg.replacement.is_none());
    }

    #[test]
    fn test_suggestion_with_replacement() {
        let span = Span::new("test.apxm".to_string(), 10, 5, 3);
        let sugg = Suggestion::with_replacement(
            "Add semicolon".to_string(),
            span.clone(),
            ";".to_string(),
        );
        assert!(sugg.replacement.is_some());
        assert_eq!(
            sugg.replacement.as_ref().map(|r| r.code.as_str()),
            Some(";")
        );
        assert_eq!(sugg.confidence, SuggestionConfidence::High);
    }

    #[test]
    fn test_suggestion_with_help() {
        let sugg = Suggestion::new("Fix this".to_string())
            .with_help("This is educational help text".to_string());
        assert_eq!(sugg.help, Some("This is educational help text".to_string()));
    }
}
