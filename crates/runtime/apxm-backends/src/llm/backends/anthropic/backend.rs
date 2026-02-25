//! Anthropic backend implementation.
//!
//! Implements the LLMBackend trait for Anthropic's Claude API.
//! This file updates the default model and the set of models returned by
//! `list_models()` to include newer Claude model identifiers.

use crate::llm::backends::{LLMBackend, LLMRequest, LLMResponse, ToolChoice};
use anyhow::{Context, Result};
use apxm_core::log_debug;
use apxm_core::types::{FinishReason, ModelCapabilities, ModelInfo, TokenUsage, ToolCall};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_MODEL: &str = "claude-opus-4";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic LLM backend.
pub struct AnthropicBackend {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl AnthropicBackend {
    /// Create a new Anthropic backend.
    pub async fn new(api_key: &str, config: Option<serde_json::Value>) -> Result<Self> {
        let model = config
            .as_ref()
            .and_then(|c| c.get("model"))
            .and_then(|m| m.as_str())
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        let base_url = config
            .as_ref()
            .and_then(|c| c.get("base_url"))
            .and_then(|u| u.as_str())
            .unwrap_or(DEFAULT_BASE_URL)
            .to_string();

        Ok(AnthropicBackend {
            api_key: api_key.to_string(),
            model,
            base_url,
            client: reqwest::Client::new(),
        })
    }

    /// Build request body for Anthropic API.
    fn build_request_body(&self, request: &LLMRequest) -> serde_json::Value {
        let mut body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "temperature": request.temperature,
        });

        // Add system prompt if provided
        if let Some(system) = &request.system_prompt {
            body["system"] = json!(system);
        }

        // Add optional parameters
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }

        if !request.stop_sequences.is_empty() {
            body["stop_sequences"] = json!(request.stop_sequences);
        }

        // Add tools if provided (Anthropic format)
        if let Some(tools) = &request.tools
            && !tools.is_empty()
        {
            let anthropic_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters
                    })
                })
                .collect();
            body["tools"] = json!(anthropic_tools);

            // Add tool_choice if specified
            if let Some(choice) = &request.tool_choice {
                body["tool_choice"] = match choice {
                    ToolChoice::Auto => json!({"type": "auto"}),
                    ToolChoice::None => json!({"type": "none"}),
                    ToolChoice::Required => json!({"type": "any"}),
                    ToolChoice::Specific(name) => json!({
                        "type": "tool",
                        "name": name
                    }),
                };
            }
        }

        body
    }

    /// Parse Anthropic API response.
    fn parse_response(&self, response: AnthropicResponse) -> Result<LLMResponse> {
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        // Process content blocks - can be text or tool_use
        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    text_content.push_str(text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall::new(id.clone(), name.clone(), input.clone()));
                }
            }
        }

        // Determine finish reason
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolUse
        } else {
            FinishReason::from_string(&response.stop_reason)
        };

        let usage = TokenUsage::new(response.usage.input_tokens, response.usage.output_tokens);

        Ok(
            LLMResponse::new(text_content, &self.model, usage, finish_reason)
                .with_tool_calls(tool_calls),
        )
    }
}

#[async_trait]
impl LLMBackend for AnthropicBackend {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse> {
        request.validate()?;

        let body = self.build_request_body(&request);
        let url = format!("{}/messages", self.base_url);

        log_debug!(
            "models::anthropic",
            model = %self.model,
            url = %url,
            "Sending request to Anthropic"
        );

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Anthropic")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Anthropic API error (status {}): {}", status, error_text);
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic response")?;

        self.parse_response(api_response)
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn health_check(&self) -> Result<()> {
        // Anthropic doesn't have a lightweight public health endpoint for all
        // models; perform a minimal generation request as a check.
        let test_request = LLMRequest::new("test").with_max_tokens(1);

        match self.generate(test_request).await {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("Anthropic health check failed: {}", e)),
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        // Derive Anthropic model metadata from the centralized enum helpers in models.rs.
        Ok(crate::llm::backends::AnthropicModel::all_models()
            .iter()
            .map(|m| m.to_model_info())
            .collect())
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            streaming: true,
            vision: self.model.starts_with("claude-") && !self.model.contains("legacy"),
            functions: true, // Claude models support tool use
            batch: false,
            fine_tuning: false,
        }
    }
}

// Anthropic API response types
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    stop_reason: String,
    usage: Usage,
}

/// Content block from Anthropic API - can be text or tool_use
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: usize,
    output_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::backends::{LLMRequest, ToolDefinition};

    #[test]
    fn test_build_request_body() {
        let backend = AnthropicBackend {
            api_key: "test".to_string(),
            model: "claude-opus-4".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        };

        let request = LLMRequest::new("Hello")
            .with_system_prompt("You are helpful")
            .with_temperature(0.9);

        let body = backend.build_request_body(&request);

        assert_eq!(body["model"], "claude-opus-4");
        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["system"], "You are helpful");
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let backend = AnthropicBackend {
            api_key: "test".to_string(),
            model: "claude-opus-4".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        };

        let tools = vec![
            ToolDefinition::new(
                "bash",
                "Execute shell commands",
                json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }),
            ),
            ToolDefinition::new(
                "read",
                "Read file contents",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }),
            ),
        ];

        let request = LLMRequest::new("List files")
            .with_tools(tools)
            .with_tool_choice(ToolChoice::Auto);

        let body = backend.build_request_body(&request);

        assert!(body.get("tools").is_some());
        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 2);
        assert_eq!(tools_arr[0]["name"], "bash");
        assert_eq!(tools_arr[1]["name"], "read");
        // Anthropic uses input_schema instead of parameters
        assert!(tools_arr[0].get("input_schema").is_some());
        assert_eq!(body["tool_choice"]["type"], "auto");
    }

    #[test]
    fn test_build_request_body_with_specific_tool_choice() {
        let backend = AnthropicBackend {
            api_key: "test".to_string(),
            model: "claude-opus-4".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        };

        let tools = vec![ToolDefinition::new("bash", "Execute shell", json!({}))];

        let request = LLMRequest::new("Run command")
            .with_tools(tools)
            .with_tool_choice(ToolChoice::Specific("bash".to_string()));

        let body = backend.build_request_body(&request);

        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "bash");
    }

    #[test]
    fn test_parse_content_block_text() {
        let json_str = r#"{"type": "text", "text": "Hello world"}"#;
        let block: ContentBlock = serde_json::from_str(json_str).unwrap();
        match block {
            ContentBlock::Text { text } => assert_eq!(text, "Hello world"),
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_parse_content_block_tool_use() {
        let json_str = r#"{
            "type": "tool_use",
            "id": "call_123",
            "name": "bash",
            "input": {"command": "ls -la"}
        }"#;
        let block: ContentBlock = serde_json::from_str(json_str).unwrap();
        match block {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "bash");
                assert_eq!(input["command"], "ls -la");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }
}
