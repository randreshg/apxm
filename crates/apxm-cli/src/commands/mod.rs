//! CLI commands.
//!
//! Each command is implemented in its own module for single responsibility
//! and extensibility.

pub mod compile;
pub mod run;

pub use compile::CompileArgs;
pub use run::RunArgs;
