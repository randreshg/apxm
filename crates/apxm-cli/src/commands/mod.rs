//! CLI commands.
//!
//! Each command is implemented in its own module for single responsibility
//! and extensibility.

pub mod compile;

pub use compile::CompileArgs;
