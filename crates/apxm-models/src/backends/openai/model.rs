use apxm_core::types::ModelInfo;
use serde::{Deserialize, Serialize};
use std::fmt;

/// OpenAI models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenAIModel {
    #[serde(rename = "gpt-4")]
    Gpt4,
    #[serde(rename = "gpt-4-turbo")]
    Gpt4Turbo,
    #[serde(rename = "gpt-4o")]
    Gpt4o,
    #[serde(rename = "gpt-4o-mini")]
    Gpt4oMini,
    #[serde(rename = "gpt-3.5-turbo")]
    Gpt35Turbo,
    #[serde(rename = "gpt-5")]
    Gpt5,
    #[serde(rename = "gpt-5-mini")]
    Gpt5Mini,
    #[serde(rename = "gpt-5-nano")]
    Gpt5Nano,
    #[serde(rename = "gpt-5.1")]
    Gpt5PointOne,
    #[serde(rename = "gpt-5.2")]
    Gpt5PointTwo,
    #[serde(rename = "o1")]
    O1,
    #[serde(rename = "o1-mini")]
    O1Mini,
    #[serde(rename = "o1-preview")]
    O1Preview,
}

impl OpenAIModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            OpenAIModel::Gpt4 => "gpt-4",
            OpenAIModel::Gpt4Turbo => "gpt-4-turbo",
            OpenAIModel::Gpt4o => "gpt-4o",
            OpenAIModel::Gpt4oMini => "gpt-4o-mini",
            OpenAIModel::Gpt35Turbo => "gpt-3.5-turbo",
            OpenAIModel::Gpt5 => "gpt-5",
            OpenAIModel::Gpt5Mini => "gpt-5-mini",
            OpenAIModel::Gpt5Nano => "gpt-5-nano",
            OpenAIModel::Gpt5PointOne => "gpt-5.1",
            OpenAIModel::Gpt5PointTwo => "gpt-5.2",
            OpenAIModel::O1 => "o1",
            OpenAIModel::O1Mini => "o1-mini",
            OpenAIModel::O1Preview => "o1-preview",
        }
    }

    pub fn all_models() -> &'static [OpenAIModel] {
        &[
            OpenAIModel::Gpt4o,
            OpenAIModel::Gpt4oMini,
            OpenAIModel::Gpt4Turbo,
            OpenAIModel::Gpt4,
            OpenAIModel::Gpt35Turbo,
            OpenAIModel::Gpt5,
            OpenAIModel::Gpt5Mini,
            OpenAIModel::Gpt5Nano,
            OpenAIModel::Gpt5PointOne,
            OpenAIModel::Gpt5PointTwo,
            OpenAIModel::O1,
            OpenAIModel::O1Mini,
            OpenAIModel::O1Preview,
        ]
    }

    /// Convert this enum variant into a canonical `ModelInfo`.
    /// Keeps model metadata centralized so backends can derive lists from the enum.
    pub fn to_model_info(&self) -> ModelInfo {
        match self {
            OpenAIModel::Gpt4oMini => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-4o Mini".to_string(),
                context_window: 128_000,
                supports_vision: true,
                supports_functions: true,
            },
            OpenAIModel::Gpt4o => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-4o (Omni)".to_string(),
                context_window: 128_000,
                supports_vision: true,
                supports_functions: true,
            },
            OpenAIModel::Gpt4Turbo => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-4 Turbo".to_string(),
                context_window: 128_000,
                supports_vision: true,
                supports_functions: true,
            },
            OpenAIModel::Gpt4 => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-4".to_string(),
                context_window: 8_192,
                supports_vision: false,
                supports_functions: true,
            },
            OpenAIModel::Gpt35Turbo => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-3.5 Turbo".to_string(),
                context_window: 16_385,
                supports_vision: false,
                supports_functions: true,
            },
            OpenAIModel::Gpt5 => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-5".to_string(),
                context_window: 272_000,
                supports_vision: true,
                supports_functions: true,
            },
            OpenAIModel::Gpt5Mini => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-5 Mini".to_string(),
                context_window: 128_000,
                supports_vision: true,
                supports_functions: true,
            },
            OpenAIModel::Gpt5Nano => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-5 Nano".to_string(),
                context_window: 128_000,
                supports_vision: false,
                supports_functions: true,
            },
            OpenAIModel::Gpt5PointOne => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-5.1".to_string(),
                context_window: 400_000,
                supports_vision: true,
                supports_functions: true,
            },
            OpenAIModel::Gpt5PointTwo => ModelInfo {
                id: self.as_str().to_string(),
                name: "GPT-5.2".to_string(),
                context_window: 400_000,
                supports_vision: true,
                supports_functions: true,
            },
            // reasonable defaults for less-common variants
            OpenAIModel::O1 | OpenAIModel::O1Mini | OpenAIModel::O1Preview => ModelInfo {
                id: self.as_str().to_string(),
                name: self.as_str().to_string(),
                context_window: 200_000,
                supports_vision: false,
                supports_functions: false,
            },
        }
    }
}

impl fmt::Display for OpenAIModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Default to the backend's default model identifier (gpt-4o-mini).
impl Default for OpenAIModel {
    fn default() -> Self {
        OpenAIModel::Gpt4oMini
    }
}
