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
    #[serde(rename = "gemini-2.5-flash")]
    #[default]
    Gemini2_5Flash,
    #[serde(rename = "gemini-2.0-pro")]
    Gemini2_0Pro,
}

impl GoogleModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            GoogleModel::Gemini2_5Flash => "gemini-2.5-flash",
            GoogleModel::Gemini2_0Pro => "gemini-2.0-pro",
        }
    }

    pub fn all_models() -> &'static [GoogleModel] {
        &[GoogleModel::Gemini2_5Flash, GoogleModel::Gemini2_0Pro]
    }

    /// Convert a GoogleModel variant into a canonical `ModelInfo`.
    pub fn to_model_info(&self) -> ModelInfo {
        match self {
            GoogleModel::Gemini2_5Flash => ModelInfo {
                id: self.as_str().to_string(),
                name: "Gemini 2.5 Flash".to_string(),
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
        }
    }
}

impl fmt::Display for GoogleModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
