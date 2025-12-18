use clap::Args;

/// Arguments for the chat command
#[derive(Debug, Args)]
pub struct ChatArgs {
    /// Session ID to resume
    #[arg(short, long, value_name = "SESSION_ID")]
    pub session: Option<String>,

    /// Model to use for chat
    #[arg(short, long, value_name = "MODEL")]
    pub model: Option<String>,

    /// Create new session
    #[arg(short, long)]
    pub new: bool,
}
