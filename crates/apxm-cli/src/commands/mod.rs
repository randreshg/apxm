//! CLI commands.
//!
//! Each command is implemented in its own module for single responsibility
//! and extensibility.

pub mod chat;
pub mod compile;
pub mod run;

pub use chat::ChatArgs;
pub use compile::CompileArgs;
pub use run::RunArgs;
