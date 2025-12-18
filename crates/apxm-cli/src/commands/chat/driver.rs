use std::path::PathBuf;

use apxm_chat::{ChatConfig, ChatSession};
use apxm_config::ApXmConfig;
use apxm_core::error::cli::{CliError, CliResult};

use super::args::ChatArgs;

async fn load_config(path: Option<PathBuf>) -> Result<ApXmConfig, CliError> {
    match path {
        Some(path) => ApXmConfig::from_file(&path).map_err(|e| CliError::Config {
            message: e.to_string(),
        }),
        None => Ok(ApXmConfig::load_scoped().unwrap_or_default()),
    }
}

pub async fn execute(args: ChatArgs, config_path: Option<PathBuf>) -> CliResult<()> {
    // 1. Load config
    let apxm_config = load_config(config_path).await?;

    // 2. Build chat config
    let mut chat_config =
        ChatConfig::from_apxm_config(apxm_config).map_err(|e| CliError::Config {
            message: e.to_string(),
        })?;

    // 3. Apply CLI overrides
    if let Some(model) = args.model {
        chat_config.default_model = model;
    }

    // 4. Create or load session
    let session = if args.new {
        ChatSession::new(chat_config).await
    } else if let Some(id) = args.session {
        ChatSession::load(&id, chat_config).await
    } else {
        ChatSession::new(chat_config).await
    }
    .map_err(|e| CliError::Runtime {
        message: e.to_string(),
    })?;

    apxm_chat::ui::run_chat_ui(session)
        .await
        .map_err(|e| CliError::Runtime {
            message: e.to_string(),
        })
}
