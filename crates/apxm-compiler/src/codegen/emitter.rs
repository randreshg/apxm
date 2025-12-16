//! Rust source code emitter with formatting and indentation support.
//!
//! This module provides utilities for emitting well-formatted Rust code.
//! It handles indentation, line breaks, and common formatting patterns.

use std::fmt::Write;

/// Rust source code emitter
///
/// Manages code generation with proper indentation and formatting.
/// Uses a string buffer for efficient accumulation of generated code.
///
/// # Example
///
/// ```rust,ignore
/// let mut emitter = RustEmitter::new();
///
/// emitter.emit_line("fn main() {");
/// emitter.indent();
/// emitter.emit_line("println!(\"Hello, world!\");");
/// emitter.dedent();
/// emitter.emit_line("}");
///
/// let code = emitter.finalize();
/// ```
pub struct RustEmitter {
    /// The buffer containing the generated code
    buffer: String,
    /// Current indentation level (0-based)
    indent_level: usize,
    /// Whether we're at the start of a new line
    at_line_start: bool,
}

impl RustEmitter {
    /// Create a new emitter with empty buffer
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(8192), // Pre-allocate reasonable size
            indent_level: 0,
            at_line_start: true,
        }
    }

    /// Create a new emitter with initial capacity hint
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
            indent_level: 0,
            at_line_start: true,
        }
    }

    /// Increase indentation level
    pub fn indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation level
    ///
    /// # Panics
    ///
    /// Panics if indentation level is already 0 (in debug builds only)
    pub fn dedent(&mut self) {
        debug_assert!(self.indent_level > 0, "Cannot dedent below 0");
        self.indent_level = self.indent_level.saturating_sub(1);
    }

    /// Set indentation level directly
    pub fn set_indent(&mut self, level: usize) {
        self.indent_level = level;
    }

    /// Get current indentation level
    pub fn indent_level(&self) -> usize {
        self.indent_level
    }

    /// Add indentation if at line start
    fn apply_indent(&mut self) {
        if self.at_line_start && self.indent_level > 0 {
            // 4 spaces per indentation level
            for _ in 0..self.indent_level {
                self.buffer.push_str("    ");
            }
            self.at_line_start = false;
        }
    }

    /// Emit text without newline
    ///
    /// Automatically adds indentation if at the start of a line.
    pub fn emit(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        self.apply_indent();
        self.buffer.push_str(text);
    }

    /// Emit text with newline
    ///
    /// Automatically adds indentation if at the start of a line.
    pub fn emit_line(&mut self, text: &str) {
        self.apply_indent();
        self.buffer.push_str(text);
        self.newline();
    }

    /// Emit a newline character
    pub fn newline(&mut self) {
        self.buffer.push('\n');
        self.at_line_start = true;
    }

    /// Emit a blank line (preserves indentation state)
    pub fn blank_line(&mut self) {
        // Don't apply indent for blank lines
        if !self.at_line_start {
            self.buffer.push('\n');
        }
        self.buffer.push('\n');
        self.at_line_start = true;
    }

    /// Emit formatted text without newline
    pub fn emit_fmt(&mut self, args: std::fmt::Arguments) {
        self.apply_indent();
        let _ = self.buffer.write_fmt(args);
    }

    /// Emit formatted text with newline
    pub fn emit_line_fmt(&mut self, args: std::fmt::Arguments) {
        self.apply_indent();
        let _ = self.buffer.write_fmt(args);
        self.newline();
    }

    /// Emit a comment
    pub fn emit_comment(&mut self, text: &str) {
        self.emit_line(&format!("// {}", text));
    }

    /// Emit a doc comment
    pub fn emit_doc_comment(&mut self, text: &str) {
        self.emit_line(&format!("/// {}", text));
    }

    /// Emit a multi-line doc comment
    pub fn emit_doc_comments(&mut self, lines: &[&str]) {
        for line in lines {
            self.emit_doc_comment(line);
        }
    }

    /// Emit a block of code with automatic indentation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// emitter.emit_block("fn example() {", |e| {
    ///     e.emit_line("let x = 42;");
    /// });
    /// // Automatically emits closing brace at previous indent level
    /// ```
    pub fn emit_block<F>(&mut self, header: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_line(header);
        self.indent();
        f(self);
        self.dedent();
        self.emit_line("}");
    }

    /// Emit an async block
    pub fn emit_async_block<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_block("async {", f);
    }

    /// Emit a match expression
    pub fn emit_match<F>(&mut self, expr: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_line(&format!("match {} {{", expr));
        self.indent();
        f(self);
        self.dedent();
        self.emit_line("}");
    }

    /// Emit a match arm
    pub fn emit_match_arm(&mut self, pattern: &str, body: &str) {
        self.emit_line(&format!("{} => {},", pattern, body));
    }

    /// Emit a match arm with block body
    pub fn emit_match_arm_block<F>(&mut self, pattern: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_line(&format!("{} => {{", pattern));
        self.indent();
        f(self);
        self.dedent();
        self.emit_line("}");
    }

    /// Emit an if statement
    pub fn emit_if<F>(&mut self, condition: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_block(&format!("if {} {{", condition), f);
    }

    /// Emit an else block
    pub fn emit_else<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_block("else {", f);
    }

    /// Emit an else if block
    pub fn emit_else_if<F>(&mut self, condition: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_block(&format!("else if {} {{", condition), f);
    }

    /// Emit a for loop
    pub fn emit_for<F>(&mut self, pattern: &str, iterator: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_block(&format!("for {} in {} {{", pattern, iterator), f);
    }

    /// Emit a while loop
    pub fn emit_while<F>(&mut self, condition: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_block(&format!("while {} {{", condition), f);
    }

    /// Emit a function declaration
    pub fn emit_function<F>(&mut self, signature: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_block(&format!("{} {{", signature), f);
    }

    /// Emit an async function declaration
    pub fn emit_async_function<F>(&mut self, name: &str, args: &str, return_type: &str, f: F)
    where
        F: FnOnce(&mut Self),
    {
        let sig = if return_type.is_empty() {
            format!("async fn {}({})", name, args)
        } else {
            format!("async fn {}({}) -> {}", name, args, return_type)
        };
        self.emit_function(&sig, f);
    }

    /// Emit struct field
    pub fn emit_struct_field(&mut self, name: &str, value: &str) {
        self.emit_line(&format!("{}: {},", name, value));
    }

    /// Emit a use statement
    pub fn emit_use(&mut self, path: &str) {
        self.emit_line(&format!("use {};", path));
    }

    /// Emit multiple use statements
    pub fn emit_uses(&mut self, paths: &[&str]) {
        for path in paths {
            self.emit_use(path);
        }
        if !paths.is_empty() {
            self.blank_line();
        }
    }

    /// Get current buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get a reference to the current buffer
    pub fn as_str(&self) -> &str {
        &self.buffer
    }

    /// Clear the buffer and reset state
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.indent_level = 0;
        self.at_line_start = true;
    }

    /// Finalize and return the generated code
    ///
    /// Consumes the emitter and returns the buffer.
    pub fn finalize(self) -> String {
        self.buffer
    }

    /// Take the buffer and reset the emitter
    pub fn take(&mut self) -> String {
        let buffer = std::mem::take(&mut self.buffer);
        self.at_line_start = true;
        buffer
    }
}

impl Default for RustEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RustEmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_emission() {
        let mut emitter = RustEmitter::new();
        emitter.emit("hello");
        emitter.emit(" ");
        emitter.emit("world");
        assert_eq!(emitter.as_str(), "hello world");
    }

    #[test]
    fn test_emit_line() {
        let mut emitter = RustEmitter::new();
        emitter.emit_line("first");
        emitter.emit_line("second");
        assert_eq!(emitter.as_str(), "first\nsecond\n");
    }

    #[test]
    fn test_indentation() {
        let mut emitter = RustEmitter::new();
        emitter.emit_line("level 0");
        emitter.indent();
        emitter.emit_line("level 1");
        emitter.indent();
        emitter.emit_line("level 2");
        emitter.dedent();
        emitter.emit_line("level 1");
        emitter.dedent();
        emitter.emit_line("level 0");

        let expected = "level 0\n    level 1\n        level 2\n    level 1\nlevel 0\n";
        assert_eq!(emitter.as_str(), expected);
    }

    #[test]
    fn test_block_emission() {
        let mut emitter = RustEmitter::new();
        emitter.emit_block("fn main() {", |e| {
            e.emit_line("let x = 42;");
            e.emit_line("println!(\"{}\", x);");
        });

        let expected = "fn main() {\n    let x = 42;\n    println!(\"{}\", x);\n}\n";
        assert_eq!(emitter.as_str(), expected);
    }

    #[test]
    fn test_nested_blocks() {
        let mut emitter = RustEmitter::new();
        emitter.emit_block("fn outer() {", |e| {
            e.emit_line("let x = 1;");
            e.emit_block("if x > 0 {", |e| {
                e.emit_line("println!(\"positive\");");
            });
        });

        assert!(emitter.as_str().contains("    let x = 1;"));
        assert!(emitter.as_str().contains("        println!(\"positive\");"));
    }

    #[test]
    fn test_blank_line() {
        let mut emitter = RustEmitter::new();
        emitter.emit_line("first");
        emitter.blank_line();
        emitter.emit_line("second");

        assert_eq!(emitter.as_str(), "first\n\nsecond\n");
    }

    #[test]
    fn test_comment() {
        let mut emitter = RustEmitter::new();
        emitter.emit_comment("This is a comment");
        emitter.emit_doc_comment("This is a doc comment");

        assert_eq!(
            emitter.as_str(),
            "// This is a comment\n/// This is a doc comment\n"
        );
    }

    #[test]
    fn test_emit_if() {
        let mut emitter = RustEmitter::new();
        emitter.emit_if("x > 0", |e| {
            e.emit_line("println!(\"positive\");");
        });

        let expected = "if x > 0 {\n    println!(\"positive\");\n}\n";
        assert_eq!(emitter.as_str(), expected);
    }

    #[test]
    fn test_emit_for() {
        let mut emitter = RustEmitter::new();
        emitter.emit_for("i", "0..10", |e| {
            e.emit_line("println!(\"{}\", i);");
        });

        let expected = "for i in 0..10 {\n    println!(\"{}\", i);\n}\n";
        assert_eq!(emitter.as_str(), expected);
    }

    #[test]
    fn test_emit_match() {
        let mut emitter = RustEmitter::new();
        emitter.emit_match("value", |e| {
            e.emit_match_arm("Some(x)", "x");
            e.emit_match_arm("None", "0");
        });

        assert!(emitter.as_str().contains("match value {"));
        assert!(emitter.as_str().contains("    Some(x) => x,"));
        assert!(emitter.as_str().contains("    None => 0,"));
    }

    #[test]
    fn test_clear() {
        let mut emitter = RustEmitter::new();
        emitter.emit_line("test");
        emitter.indent();
        assert!(!emitter.is_empty());

        emitter.clear();
        assert!(emitter.is_empty());
        assert_eq!(emitter.indent_level(), 0);
    }

    #[test]
    fn test_take() {
        let mut emitter = RustEmitter::new();
        emitter.emit_line("first");

        let taken = emitter.take();
        assert_eq!(taken, "first\n");
        assert!(emitter.is_empty());

        emitter.emit_line("second");
        assert_eq!(emitter.as_str(), "second\n");
    }

    #[test]
    fn test_emit_uses() {
        let mut emitter = RustEmitter::new();
        emitter.emit_uses(&["std::collections::HashMap", "std::sync::Arc"]);

        assert!(emitter.as_str().contains("use std::collections::HashMap;"));
        assert!(emitter.as_str().contains("use std::sync::Arc;"));
    }

    #[test]
    fn test_async_function() {
        let mut emitter = RustEmitter::new();
        emitter.emit_async_function("process", "data: String", "Result<()>", |e| {
            e.emit_line("Ok(())");
        });

        assert!(
            emitter
                .as_str()
                .contains("async fn process(data: String) -> Result<()>")
        );
    }
}
