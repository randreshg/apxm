//! Slash command system for chat

use crate::{
    error::{ChatError, ChatResult},
    session::ChatSession,
};

/// Slash command actions
#[derive(Debug, Clone)]
pub enum SlashCommand {
    /// Show help
    Help,

    /// Show or change model
    Model(Option<String>),

    /// Configuration actions
    Config(ConfigAction),

    /// Start new session
    New,

    /// List all sessions
    Sessions,

    /// Load a session
    Load(String),

    /// Clear conversation context
    Clear,

    /// Exit chat
    Exit,
}

/// Configuration actions
#[derive(Debug, Clone)]
pub enum ConfigAction {
    /// Set a config value
    Set { key: String, value: String },

    /// Get a config value
    Get { key: String },

    /// List all config
    List,
}

/// Metadata describing a slash command for UI suggestions.
#[derive(Debug, Clone)]
pub struct SlashMetadata {
    /// Canonical usage syntax (e.g. "/help").
    pub usage: &'static str,
    /// Short description for the UI.
    pub description: &'static str,
    /// Keywords/aliases used for prefix matching.
    pub keywords: &'static [&'static str],
}

/// All slash command definitions used for autocomplete.
pub const SLASH_COMMANDS: &[SlashMetadata] = &[
    SlashMetadata {
        usage: "/help",
        description: "Show all available slash commands",
        keywords: &["help", "h", "?"],
    },
    SlashMetadata {
        usage: "/model [name]",
        description: "Show or switch the default LLM model",
        keywords: &["model", "m"],
    },
    SlashMetadata {
        usage: "/config list",
        description: "List chat configuration values",
        keywords: &["config", "cfg"],
    },
    SlashMetadata {
        usage: "/config get <key>",
        description: "Inspect a specific config key",
        keywords: &["config", "cfg"],
    },
    SlashMetadata {
        usage: "/config set <key> <value>",
        description: "(Future) persist an override",
        keywords: &["config", "cfg"],
    },
    SlashMetadata {
        usage: "/sessions",
        description: "List stored sessions",
        keywords: &["sessions", "ls"],
    },
    SlashMetadata {
        usage: "/load <session-id>",
        description: "Load an existing session by ID",
        keywords: &["load", "l"],
    },
    SlashMetadata {
        usage: "/clear",
        description: "Clear the conversation view",
        keywords: &["clear", "cls"],
    },
    SlashMetadata {
        usage: "/exit",
        description: "Exit the chat UI",
        keywords: &["exit", "quit", "q"],
    },
];

/// Return matching slash-command metadata for the current composer text.
pub fn slash_suggestions(input: &str) -> Vec<&'static SlashMetadata> {
    if !input.trim_start().starts_with('/') {
        return Vec::new();
    }

    let query = input.trim_start_matches('/');
    let prefix = query.split_whitespace().next().unwrap_or("").to_lowercase();

    let mut matches: Vec<&SlashMetadata> = SLASH_COMMANDS
        .iter()
        .filter(|meta| {
            if prefix.is_empty() {
                return true;
            }
            meta.keywords
                .iter()
                .any(|kw| kw.starts_with(prefix.as_str()))
        })
        .collect();

    matches.truncate(5);
    matches
}

impl SlashCommand {
    /// Parse input into a slash command
    pub fn parse(input: &str) -> Option<Self> {
        if !input.starts_with('/') {
            return None;
        }

        let parts: Vec<&str> = input[1..].split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        match parts[0].to_lowercase().as_str() {
            "help" | "h" | "?" => Some(SlashCommand::Help),

            "model" | "m" => Some(SlashCommand::Model(parts.get(1).map(|s| s.to_string()))),

            "config" | "cfg" => Self::parse_config_command(&parts[1..]),

            "new" | "n" => Some(SlashCommand::New),

            "sessions" | "ls" => Some(SlashCommand::Sessions),

            "load" | "l" => Some(SlashCommand::Load(
                parts.get(1).map(|s| s.to_string()).unwrap_or_default(),
            )),

            "clear" | "cls" => Some(SlashCommand::Clear),

            "exit" | "quit" | "q" => Some(SlashCommand::Exit),

            _ => None,
        }
    }

    /// Parse config subcommand
    fn parse_config_command(parts: &[&str]) -> Option<SlashCommand> {
        if parts.is_empty() {
            return Some(SlashCommand::Config(ConfigAction::List));
        }

        match parts[0].to_lowercase().as_str() {
            "set" if parts.len() >= 3 => Some(SlashCommand::Config(ConfigAction::Set {
                key: parts[1].to_string(),
                value: parts[2..].join(" "),
            })),

            "get" if parts.len() == 2 => Some(SlashCommand::Config(ConfigAction::Get {
                key: parts[1].to_string(),
            })),

            "list" | "ls" => Some(SlashCommand::Config(ConfigAction::List)),

            _ => None,
        }
    }

    /// Execute the command on a session
    pub async fn execute(&self, session: &mut ChatSession) -> ChatResult<String> {
        match self {
            SlashCommand::Help => Ok(Self::help_text()),

            SlashCommand::Model(None) => {
                Ok(format!("Current model: {}", session.config().default_model))
            }

            SlashCommand::Model(Some(model)) => {
                session.config_mut().default_model = model.clone();
                session.reinit_linker().await?;
                Ok(format!("Switched to model: {}", model))
            }

            SlashCommand::Config(ConfigAction::Set { key: _, value: _ }) => {
                // TODO: Implement TOML write-back
                Err(ChatError::Command(
                    "Config write-back not yet implemented".to_string(),
                ))
            }

            SlashCommand::Config(ConfigAction::Get { key }) => {
                // Simple key lookup (basic implementation)
                match key.as_str() {
                    "model" | "default_model" => Ok(format!(
                        "default_model = {:?}",
                        session.config().default_model
                    )),
                    "planning_model" => Ok(format!(
                        "planning_model = {:?}",
                        session.config().planning_model
                    )),
                    "max_context_tokens" => Ok(format!(
                        "max_context_tokens = {}",
                        session.config().max_context_tokens
                    )),
                    _ => Err(ChatError::Command(format!("Unknown config key: {}", key))),
                }
            }

            SlashCommand::Config(ConfigAction::List) => {
                let config = session.config();
                let mut output = String::from("Configuration:\n");
                output.push_str(&format!("  default_model = {:?}\n", config.default_model));
                output.push_str(&format!("  planning_model = {:?}\n", config.planning_model));
                output.push_str(&format!(
                    "  max_context_tokens = {}\n",
                    config.max_context_tokens
                ));
                output.push_str(&format!(
                    "  session_storage_path = {:?}\n",
                    config.session_storage_path
                ));
                output.push_str(&format!(
                    "  llm_backends = {} configured\n",
                    config.apxm_config.llm_backends.len()
                ));
                Ok(output)
            }

            SlashCommand::New => Err(ChatError::Command(
                "Use /exit and restart to create a new session (or use --new flag)".to_string(),
            )),

            SlashCommand::Sessions => {
                let sessions =
                    ChatSession::list_sessions(&session.config().session_storage_path).await?;

                if sessions.is_empty() {
                    return Ok("No sessions found.".to_string());
                }

                let mut output = format!("Sessions ({}):\n", sessions.len());
                for (idx, info) in sessions.iter().enumerate().take(10) {
                    let current = if info.id == session.id() {
                        " (current)"
                    } else {
                        ""
                    };
                    output.push_str(&format!(
                        "  {}. {} - {} messages - {} (updated: {}){}\n",
                        idx + 1,
                        &info.id[..8],
                        info.message_count,
                        info.model,
                        info.updated_at.format("%Y-%m-%d %H:%M"),
                        current
                    ));
                }

                if sessions.len() > 10 {
                    output.push_str(&format!("  ... and {} more\n", sessions.len() - 10));
                }

                Ok(output)
            }

            SlashCommand::Load(id) => {
                if id.is_empty() {
                    return Err(ChatError::Command("Usage: /load <session_id>".to_string()));
                }

                Err(ChatError::Command(
                    "Session loading requires restarting with --session flag".to_string(),
                ))
            }

            SlashCommand::Clear => {
                session.clear_messages();
                Ok("Conversation cleared.".to_string())
            }

            SlashCommand::Exit => {
                session.save().await?;
                Err(ChatError::Command("exit".to_string()))
            }
        }
    }

    /// Get help text
    fn help_text() -> String {
        r#"Available commands:
  /help, /h, /?              Show this help message
  /model, /m [name]          Show current model or switch to [name]
  /config, /cfg [action]     Manage configuration
    /config list             List all configuration
    /config get <key>        Get configuration value
    /config set <key> <val>  Set configuration value
  /new, /n                   Create new session
  /sessions, /ls             List all sessions
  /load, /l <id>             Load a session
  /clear, /cls               Clear conversation history
  /exit, /quit, /q           Save and exit chat

Usage:
  Just type your message to chat
  Start with / for commands
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_help() {
        assert!(matches!(
            SlashCommand::parse("/help"),
            Some(SlashCommand::Help)
        ));
        assert!(matches!(
            SlashCommand::parse("/h"),
            Some(SlashCommand::Help)
        ));
        assert!(matches!(
            SlashCommand::parse("/?"),
            Some(SlashCommand::Help)
        ));
    }

    #[test]
    fn test_parse_model() {
        assert!(matches!(
            SlashCommand::parse("/model"),
            Some(SlashCommand::Model(None))
        ));
        assert!(matches!(
            SlashCommand::parse("/model gpt-4"),
            Some(SlashCommand::Model(Some(_)))
        ));
    }

    #[test]
    fn test_parse_config() {
        assert!(matches!(
            SlashCommand::parse("/config"),
            Some(SlashCommand::Config(ConfigAction::List))
        ));
        assert!(matches!(
            SlashCommand::parse("/config list"),
            Some(SlashCommand::Config(ConfigAction::List))
        ));
    }

    #[test]
    fn test_parse_invalid() {
        assert!(SlashCommand::parse("/invalid").is_none());
        assert!(SlashCommand::parse("not a command").is_none());
        assert!(SlashCommand::parse("").is_none());
    }
}
