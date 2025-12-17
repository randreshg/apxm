//! Model definitions per provider.
//!
//! Each provider has an enum of supported models for easy extensibility.

use apxm_core::types::ModelInfo;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Anthropic Claude models.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Default
)]
pub enum AnthropicModel {
    #[serde(rename = "claude-opus-4-5")]
    ClaudeOpus4_5,
    #[serde(rename = "claude-sonnet-4-5")]
    #[default]
    ClaudeSonnet4_5,
    #[serde(rename = "claude-3-7-sonnet")]
    Claude3_7Sonnet,
    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Claude3_5Sonnet20241022,
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus20240229,
    #[serde(rename = "claude-3-sonnet-20240229")]
    Claude3Sonnet20240229,
    #[serde(rename = "claude-3-haiku-20240307")]
    Claude3Haiku20240307,
}

impl AnthropicModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnthropicModel::ClaudeOpus4_5 => "claude-opus-4-5",
            AnthropicModel::ClaudeSonnet4_5 => "claude-sonnet-4-5",
            AnthropicModel::Claude3_7Sonnet => "claude-3-7-sonnet",
            AnthropicModel::Claude3_5Sonnet20241022 => "claude-3-5-sonnet-20241022",
            AnthropicModel::Claude3Opus20240229 => "claude-3-opus-20240229",
            AnthropicModel::Claude3Sonnet20240229 => "claude-3-sonnet-20240229",
            AnthropicModel::Claude3Haiku20240307 => "claude-3-haiku-20240307",
        }
    }

    pub fn all_models() -> &'static [AnthropicModel] {
        &[
            AnthropicModel::ClaudeOpus4_5,
            AnthropicModel::ClaudeSonnet4_5,
            AnthropicModel::Claude3_7Sonnet,
            AnthropicModel::Claude3_5Sonnet20241022,
            AnthropicModel::Claude3Opus20240229,
            AnthropicModel::Claude3Sonnet20240229,
            AnthropicModel::Claude3Haiku20240307,
        ]
    }

    /// Convert an AnthropicModel variant into a canonical `ModelInfo`.
    pub fn to_model_info(&self) -> ModelInfo {
        match self {
            AnthropicModel::ClaudeOpus4_5 => ModelInfo {
                id: self.as_str().to_string(),
                name: "Claude Opus 4.5".to_string(),
                context_window: 200_000,
                supports_vision: true,
                supports_functions: false,
            },
            AnthropicModel::ClaudeSonnet4_5 => ModelInfo {
                id: self.as_str().to_string(),
                name: "Claude Sonnet 4.5".to_string(),
                context_window: 200_000,
                supports_vision: true,
                supports_functions: false,
            },
            AnthropicModel::Claude3_7Sonnet => ModelInfo {
                id: self.as_str().to_string(),
                name: "Claude 3.7 Sonnet".to_string(),
                context_window: 200_000,
                supports_vision: true,
                supports_functions: false,
            },
            AnthropicModel::Claude3_5Sonnet20241022 => ModelInfo {
                id: self.as_str().to_string(),
                name: "Claude 3.5 Sonnet".to_string(),
                context_window: 200_000,
                supports_vision: true,
                supports_functions: false,
            },
            AnthropicModel::Claude3Opus20240229 => ModelInfo {
                id: self.as_str().to_string(),
                name: "Claude 3 Opus".to_string(),
                context_window: 200_000,
                supports_vision: true,
                supports_functions: false,
            },
            AnthropicModel::Claude3Sonnet20240229 => ModelInfo {
                id: self.as_str().to_string(),
                name: "Claude 3 Sonnet".to_string(),
                context_window: 200_000,
                supports_vision: true,
                supports_functions: false,
            },
            AnthropicModel::Claude3Haiku20240307 => ModelInfo {
                id: self.as_str().to_string(),
                name: "Claude 3 Haiku".to_string(),
                context_window: 200_000,
                supports_vision: true,
                supports_functions: false,
            },
        }
    }
}

impl fmt::Display for AnthropicModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_models() {
        assert_eq!(
            AnthropicModel::ClaudeSonnet4_5.as_str(),
            "claude-sonnet-4-5"
        );
        assert_eq!(AnthropicModel::default(), AnthropicModel::ClaudeSonnet4_5);
    }
}
