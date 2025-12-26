//! Error type with full context.
//!
//! This is the core error type that provides:
//! - Error codes
//! - Source spans with snippets
//! - Actionable suggestions
//! - Educational help text
//! - Error context (operation_id, trace_id, etc.)

use crate::error::codes::ErrorCode;
use crate::error::common::ErrorContext;
use crate::error::span::Span;
use crate::error::suggestion::{ReplacementPart, Suggestion};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fmt;

/// error with full context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Error {
    /// Error code.
    pub code: ErrorCode,
    /// Primary error message.
    pub message: String,
    /// Primary span (where the error occurred).
    pub primary_span: Span,
    /// Secondary spans (related locations).
    pub secondary_spans: Vec<Span>,
    /// Suggestions for fixing the error.
    pub suggestions: Vec<Suggestion>,
    /// Help text (educational content).
    pub help: Option<String>,
    /// Error context (operation_id, trace_id, etc.).
    pub context: Option<ErrorContext>,
    /// Notes (additional information).
    pub notes: Vec<String>,
    /// Related errors (error chain).
    pub related: Vec<Error>,
}

impl Error {
    /// Creates a new error with the given code, message, and primary span.
    pub fn new<S>(code: ErrorCode, message: S, primary_span: Span) -> Self
    where
        S: Into<String>,
    {
        Error {
            code,
            message: message.into(),
            primary_span,
            secondary_spans: Vec::new(),
            suggestions: Vec::new(),
            help: None,
            context: None,
            notes: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Creates a new error with a default `<unknown>` span.
    pub fn new_generic<S>(code: ErrorCode, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::new(code, message, Span::new("<unknown>".to_string(), 0, 0, 0))
    }

    /// Creates a new error associated with a file but no specific location.
    pub fn new_global<S, F>(code: ErrorCode, message: S, file: F) -> Self
    where
        S: Into<String>,
        F: Into<String>,
    {
        Self::new(code, message, Span::new(file.into(), 0, 0, 0))
    }

    /// Adds a suggestion to this error.
    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Adds help text to this error.
    pub fn with_help<S>(mut self, help: S) -> Self
    where
        S: Into<String>,
    {
        self.help = Some(help.into());
        self
    }

    /// Adds error context (operation_id, trace_id, etc.) to this error.
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Adds a note to this error.
    pub fn with_note<S>(mut self, note: S) -> Self
    where
        S: Into<String>,
    {
        self.notes.push(note.into());
        self
    }

    /// Adds a secondary span to this error.
    pub fn with_secondary_span(mut self, span: Span) -> Self {
        self.secondary_spans.push(span);
        self
    }

    /// Adds a related error to this error.
    pub fn with_related(mut self, error: Error) -> Self {
        self.related.push(error);
        self
    }

    /// Returns a rustc-style pretty-printed representation of this error.
    ///
    /// If `source` is provided, spans will be rendered with line snippets and
    /// caret highlights. Otherwise, only span locations are shown.
    pub fn pretty_print(&self, source: Option<&str>) -> String {
        let mut output = String::with_capacity(1024);

        // Pre-split source once for all span formatting to keep this DRY and efficient.
        let lines: Option<Vec<&str>> = source.map(|src| src.lines().collect());
        let lines = lines.as_deref();

        // Error header.
        output.push_str(&format!(
            "{}: {}\n",
            format!("error[{}]", self.code.as_str()).red().bold(),
            self.message.bold()
        ));

        // Primary span.
        Self::push_span(&mut output, &self.primary_span, lines, true);

        // Secondary spans.
        for span in &self.secondary_spans {
            Self::push_span(&mut output, span, lines, false);
        }

        // Suggestions.
        for (i, suggestion) in self.suggestions.iter().enumerate() {
            output.push_str(&format!(
                "\n  {}: {}\n",
                format!("help[{}]", i + 1).cyan().bold(),
                suggestion.message
            ));

            if let Some(replacement) = &suggestion.replacement {
                if !replacement.parts.is_empty() {
                    // Multi-part replacement: only emitted if we know the source.
                    if let Some(lines) = lines {
                        for part in &replacement.parts {
                            output.push_str(&Self::format_replacement_part(part, lines));
                        }
                    }
                } else {
                    // Simple replacement: can be shown without source.
                    output.push_str(&format!("     {}\n", "|".blue().bold()));
                    output.push_str(&format!(
                        "     {} {}\n",
                        "|".blue().bold(),
                        replacement.code.green()
                    ));
                }
            }

            if let Some(help) = &suggestion.help {
                output.push_str(&format!("     {}\n", "|".blue().bold()));
                output.push_str(&format!("     {} {}\n", "|".blue().bold(), help));
            }
        }

        // Help text.
        if let Some(help) = &self.help {
            output.push_str(&format!("\n  {}: {}\n", "help".cyan().bold(), help));
        }

        // Notes.
        for note in &self.notes {
            output.push_str(&format!("  {}: {}\n", "note".cyan(), note));
        }

        // Related errors (only headers, to avoid deeply nested pretty-printing).
        for related in &self.related {
            output.push_str(&format!(
                "\n  {}: {}\n",
                format!("related error[{}]", related.code.as_str()).red(),
                related.message
            ));
        }

        // Documentation link.
        output.push_str(&format!(
            "\n  For more information, see: {}\n",
            self.code.documentation_url()
        ));

        output
    }

    /// Returns a short, log-friendly error message.
    pub fn short_message(&self) -> String {
        format!("[{}] {}", self.code.as_str(), self.message)
    }

    /// Pushes either a span with snippet or a simple location line, depending on
    /// whether source lines are available.
    fn push_span(output: &mut String, span: &Span, lines: Option<&[&str]>, is_primary: bool) {
        match lines {
            Some(lines) => {
                output.push_str(&Self::format_span_with_snippet(span, lines, is_primary))
            }
            None => output.push_str(&format!("  {} {}\n", "-->".blue().bold(), span)),
        }
    }

    /// Formats a span together with a source snippet and caret highlights.
    fn format_span_with_snippet(span: &Span, lines: &[&str], is_primary: bool) -> String {
        if span.line_start == 0 || span.line_start > lines.len() {
            return format!("  {} {}\n", "-->".blue().bold(), span);
        }

        let line_idx = span.line_start - 1;
        let line = lines[line_idx];
        let mut output = String::new();

        // File location.
        output.push_str(&format!(
            "  {} {}:{}:{}\n",
            "-->".blue().bold(),
            span.file,
            span.line_start,
            span.col_start
        ));
        output.push_str(&format!("   {}\n", "|".blue().bold()));

        // Line number and content.
        let line_num_str = span.line_start.to_string();
        output.push_str(&format!(
            "{} {} {}\n",
            format!("{:>4}", line_num_str).blue().bold(),
            "|".blue().bold(),
            line
        ));
        output.push_str(&format!("   {} ", "|".blue().bold()));

        // Highlight the error.
        let highlight_start = span.col_start.saturating_sub(1);
        let highlight_end = span.col_end.min(line.len());
        let highlight_len = highlight_end.saturating_sub(highlight_start);

        for _ in 0..highlight_start {
            output.push(' ');
        }

        if highlight_len > 0 {
            for _ in 0..highlight_len {
                output.push_str(&"^".red().bold().to_string());
            }
        } else {
            // Always show at least one caret.
            output.push_str(&"^".red().bold().to_string());
        }

        // Label.
        if let Some(label) = &span.label {
            output.push(' ');
            output.push_str(&label.red().bold().to_string());
        } else if is_primary {
            output.push_str(&" expected here".red().bold().to_string());
        }

        output.push('\n');
        output
    }

    /// Formats a replacement part (for multi-part suggestions) using the source.
    fn format_replacement_part(part: &ReplacementPart, lines: &[&str]) -> String {
        if part.span.line_start == 0 || part.span.line_start > lines.len() {
            return String::new();
        }

        let line_idx = part.span.line_start - 1;
        let line = lines[line_idx];
        let mut output = String::new();

        let line_num_str = part.span.line_start.to_string();
        output.push_str(&format!(
            "{} {} {}\n",
            format!("{:>4}", line_num_str).blue().bold(),
            "|".blue().bold(),
            line
        ));
        output.push_str(&format!("   {} ", "|".blue().bold()));

        // Spaces before the replacement.
        let highlight_start = part.span.col_start.saturating_sub(1);
        for _ in 0..highlight_start {
            output.push(' ');
        }

        // Replacement code.
        output.push_str(&part.code.green().to_string());

        // Optional label.
        if let Some(label) = &part.label {
            output.push_str("  // ");
            output.push_str(label);
        }

        output.push('\n');
        output
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Expose the first related error as the source, if any.
        self.related.first().map(|e| e as &dyn std::error::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::span::Span;

    #[test]
    fn test_error_new() {
        let span = Span::new("test.apxm".to_string(), 10, 5, 3);
        let error = Error::new(ErrorCode::ExpectedExpression, "expected expression", span);
        assert_eq!(error.code, ErrorCode::ExpectedExpression);
        assert_eq!(error.message, "expected expression");
    }

    #[test]
    fn test_error_with_suggestion() {
        let span = Span::new("test.apxm".to_string(), 10, 5, 3);
        let error = Error::new(
            ErrorCode::ExpectedExpression,
            "expected expression",
            span.clone(),
        )
        .with_suggestion(Suggestion::with_replacement(
            "Add expression".to_string(),
            span,
            "42".to_string(),
        ));
        assert_eq!(error.suggestions.len(), 1);
    }

    #[test]
    fn test_error_display() {
        let span = Span::new("test.apxm".to_string(), 10, 5, 3);
        let error = Error::new(ErrorCode::ExpectedExpression, "expected expression", span);
        let display = format!("{error}");
        assert!(display.contains("E002"));
        assert!(display.contains("expected expression"));
    }

    #[test]
    fn test_error_pretty_print() {
        let span = Span::new("test.apxm".to_string(), 1, 1, 1);
        let error = Error::new(ErrorCode::ExpectedExpression, "expected expression", span);
        let source = "let x = ";
        let output = error.pretty_print(Some(source));
        assert!(output.contains("error[E002]"));
        assert!(output.contains("expected expression"));
    }
}
