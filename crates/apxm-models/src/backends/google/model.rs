use apxm_core::types::ModelInfo;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Google Gemini models.
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
pub enum GoogleModel {
    #[serde(rename = "gemini-2.0-flash-exp")]
    Gemini2_0FlashExp,
    #[serde(rename = "gemini-2.0-pro")]
    Gemini2_0Pro,
    #[serde(rename = "gemini-2.1-pro")]
    Gemini2_1Pro,
    #[serde(rename = "gemini-1.5-pro")]
    #[default]
    Gemini1_5Pro,
    #[serde(rename = "gemini-1.5-flash")]
    Gemini1_5Flash,
    #[serde(rename = "gemini-1.0-pro")]
    Gemini1_0Pro,
}

impl GoogleModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            GoogleModel::Gemini2_0FlashExp => "gemini-2.0-flash-exp",
            GoogleModel::Gemini2_0Pro => "gemini-2.0-pro",
            GoogleModel::Gemini2_1Pro => "gemini-2.1-pro",
            GoogleModel::Gemini1_5Pro => "gemini-1.5-pro",
            GoogleModel::Gemini1_5Flash => "gemini-1.5-flash",
            GoogleModel::Gemini1_0Pro => "gemini-1.0-pro",
        }
    }

    pub fn all_models() -> &'static [GoogleModel] {
        &[
            GoogleModel::Gemini2_0FlashExp,
            GoogleModel::Gemini2_0Pro,
            GoogleModel::Gemini2_1Pro,
            GoogleModel::Gemini1_5Pro,
            GoogleModel::Gemini1_5Flash,
            GoogleModel::Gemini1_0Pro,
        ]
    }

    /// Convert a GoogleModel variant into a canonical `ModelInfo`.
    pub fn to_model_info(&self) -> ModelInfo {
        match self {
            GoogleModel::Gemini2_0FlashExp => ModelInfo {
                id: self.as_str().to_string(),
                name: "Gemini 2.0 Flash (exp)".to_string(),
                context_window: 1_000_000,
                supports_vision: true,
                supports_functions: true,
            },
            GoogleModel::Gemini2_0Pro => ModelInfo {
                id: self.as_str().to_string(),
                name: "Gemini 2.0 Pro".to_string(),
                context_window: 1_000_000,
                supports_vision: true,
                supports_functions: true,
            },
            GoogleModel::Gemini2_1Pro => ModelInfo {
                id: self.as_str().to_string(),
                name: "Gemini 2.1 Pro".to_string(),
                context_window: 1_000_000,
                supports_vision: true,
                supports_functions: true,
            },
            GoogleModel::Gemini1_5Pro => ModelInfo {
                id: self.as_str().to_string(),
                name: "Gemini 1.5 Pro".to_string(),
                context_window: 1_000_000,
                supports_vision: true,
                supports_functions: true,
            },
            GoogleModel::Gemini1_5Flash => ModelInfo {
                id: self.as_str().to_string(),
                name: "Gemini 1.5 Flash".to_string(),
                context_window: 1_000_000,
                supports_vision: true,
                supports_functions: true,
            },
            GoogleModel::Gemini1_0Pro => ModelInfo {
                id: self.as_str().to_string(),
                name: "Gemini 1.0 Pro".to_string(),
                context_window: 1_000_000,
                supports_vision: true,
                supports_functions: true,
            },
        }
    }
}

impl fmt::Display for GoogleModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
