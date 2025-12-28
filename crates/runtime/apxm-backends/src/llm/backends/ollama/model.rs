use serde::{Deserialize, Serialize};
use std::fmt;

// Ollama local models (dynamic).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OllamaModel {
    #[serde(rename = "gpt-oss:120b-cloud")]
    #[default]
    GptOss120bCloud,
    #[serde(rename = "llama3.3")]
    Llama3_3,
    #[serde(rename = "llama3.2")]
    Llama3_2,
    #[serde(rename = "llama3.1")]
    Llama3_1,
    #[serde(rename = "qwen2.5")]
    Qwen2_5,
    #[serde(rename = "mistral")]
    Mistral,
    #[serde(rename = "phi3")]
    Phi3,
    #[serde(rename = "deepseek-r1")]
    DeepseekR1,
    /// Custom model name (for locally available models not in enum)
    Custom(String),
}

impl OllamaModel {
    pub fn as_str(&self) -> &str {
        match self {
            OllamaModel::GptOss120bCloud => "gpt-oss:120b-cloud",
            OllamaModel::Llama3_3 => "llama3.3",
            OllamaModel::Llama3_2 => "llama3.2",
            OllamaModel::Llama3_1 => "llama3.1",
            OllamaModel::Qwen2_5 => "qwen2.5",
            OllamaModel::Mistral => "mistral",
            OllamaModel::Phi3 => "phi3",
            OllamaModel::DeepseekR1 => "deepseek-r1",
            OllamaModel::Custom(name) => name,
        }
    }

    pub fn from_string(s: impl Into<String>) -> Self {
        let s = s.into();
        match s.as_str() {
            "gpt-oss:120b-cloud" => OllamaModel::GptOss120bCloud,
            "llama3.3" => OllamaModel::Llama3_3,
            "llama3.2" => OllamaModel::Llama3_2,
            "llama3.1" => OllamaModel::Llama3_1,
            "qwen2.5" => OllamaModel::Qwen2_5,
            "mistral" => OllamaModel::Mistral,
            "phi3" => OllamaModel::Phi3,
            "deepseek-r1" => OllamaModel::DeepseekR1,
            _ => OllamaModel::Custom(s),
        }
    }

    pub fn common_models() -> &'static [&'static str] {
        &[
            "gpt-oss:120b-cloud",
            "llama3.3",
            "llama3.2",
            "llama3.1",
            "qwen2.5",
            "mistral",
            "phi3",
            "deepseek-r1",
        ]
    }
}

impl fmt::Display for OllamaModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
