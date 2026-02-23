//! Source code spans for error reporting.
//!
//! Inspired by Rust compiler's Span system, providing rich source location
//! information including source snippets, highlighting, and multi-span support.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A source code span (location + length)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// File path
    pub file: String,
    /// Starting line (1-indexed)
    pub line_start: usize,
    /// Starting column (1-indexed)
    pub col_start: usize,
    /// Ending line (1-indexed)
    pub line_end: usize,
    /// Ending column (1-indexed)
    pub col_end: usize,
    /// Source code snippet (for display)
    pub snippet: Option<String>,
    /// Highlighted portion (indices within snippet)
    pub highlight: Option<(usize, usize)>,
    /// Label for this span (e.g., "expected here", "found here")
    pub label: Option<String>,
}

impl Span {
    /// Create a new span from location and length
    pub fn new(file: String, line: usize, col: usize, length: usize) -> Self {
        Span {
            file,
            line_start: line,
            col_start: col,
            line_end: line,
            col_end: col + length,
            snippet: None,
            highlight: None,
            label: None,
        }
    }

    /// Create a span with snippet and highlighting
    pub fn with_snippet(mut self, snippet: String, highlight: (usize, usize)) -> Self {
        self.snippet = Some(snippet);
        self.highlight = Some(highlight);
        self
    }

    /// Add a label to this span
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Create a multi-line span
    pub fn multi_line(
        file: String,
        line_start: usize,
        col_start: usize,
        line_end: usize,
        col_end: usize,
    ) -> Self {
        Span {
            file,
            line_start,
            col_start,
            line_end,
            col_end,
            snippet: None,
            highlight: None,
            label: None,
        }
    }

    /// Check if span is single-line
    pub fn is_single_line(&self) -> bool {
        self.line_start == self.line_end
    }

    /// Get the length of the span in characters
    pub fn length(&self) -> usize {
        if self.is_single_line() {
            self.col_end.saturating_sub(self.col_start)
        } else {
            // Multi-line: approximate length
            (self.line_end - self.line_start) * 80 + self.col_end
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_single_line() {
            write!(f, "{}:{}:{}", self.file, self.line_start, self.col_start)
        } else {
            write!(
                f,
                "{}:{}:{} to {}:{}:{}",
                self.file, self.line_start, self.col_start, self.file, self.line_end, self.col_end
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_multi_line() {
        let span = Span::multi_line("test.apxm".to_string(), 10, 5, 12, 10);
        assert_eq!(span.line_start, 10);
        assert_eq!(span.line_end, 12);
        assert!(!span.is_single_line());
    }

    #[test]
    fn test_span_display() {
        let span = Span::new("test.apxm".to_string(), 10, 5, 3);
        assert_eq!(span.to_string(), "test.apxm:10:5");

        let multi = Span::multi_line("test.apxm".to_string(), 10, 5, 12, 10);
        assert!(multi.to_string().contains("10:5"));
        assert!(multi.to_string().contains("12:10"));
    }
}
