//! APxM Chat Interface Library
//!
//! Provides a chat interface with natural language to DSL translation,
//! session management, and command support.

mod commands;
mod config;
mod error;
mod session;
mod storage;
mod translator;

pub use commands::{ConfigAction, SlashCommand};
pub use config::ChatConfig;
pub use error::{ChatError, ChatResult};
pub use session::{ChatResponse, ChatSession, ChatStreamEvent};
pub use storage::{Message, SessionInfo, SessionStorage};
pub use translator::{TranslationResult, Translator};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod ui;
