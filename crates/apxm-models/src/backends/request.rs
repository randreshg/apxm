//! LLM request types and builders.

use std::collections::HashMap;

/// Request to send to an LLM backend.
#[derive(Debug, Clone)]
pub struct LLMRequest {
    /// The main prompt/input text
    pub prompt: String,
    /// Optional system prompt providing context
    pub system_prompt: Option<String>,
    /// Temperature controls randomness (0.0-2.0, typical: 0.7)
    pub temperature: f64,
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    /// Nucleus sampling - keep top p probability mass
    pub top_p: Option<f64>,
    /// Frequency penalty - penalize repeated tokens
    pub frequency_penalty: Option<f64>,
    /// Presence penalty - encourage new topics
    pub presence_penalty: Option<f64>,
    /// Stop sequences where generation stops
    pub stop_sequences: Vec<String>,
    /// Custom metadata passed through to provider
    pub metadata: HashMap<String, serde_json::Value>,
    /// Explicitly requested backend (for routing)
    pub backend: Option<String>,
    /// Explicitly requested model (for routing)
    pub model: Option<String>,
    /// Operation type for intelligent routing (e.g., "reason", "plan", "generate")
    pub operation_type: Option<String>,
}

impl LLMRequest {
    /// Create a new request with just a prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        LLMRequest {
            prompt: prompt.into(),
            system_prompt: None,
            temperature: 0.7,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Vec::new(),
            metadata: HashMap::new(),
            backend: None,
            model: None,
            operation_type: None,
        }
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
        self
    }

    /// Set temperature (0.0-2.0).
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set maximum tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set top_p (nucleus sampling).
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    /// Set frequency penalty.
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set presence penalty.
    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Add a stop sequence.
    pub fn add_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    /// Add custom metadata.
    pub fn with_metadata_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set explicit backend for routing.
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    /// Set explicit model for routing.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set operation type for routing (e.g., "reason", "plan", "generate").
    pub fn with_operation_type(mut self, operation: impl Into<String>) -> Self {
        self.operation_type = Some(operation.into());
        self
    }

    /// Validate request parameters.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.prompt.is_empty() {
            return Err(anyhow::anyhow!("Prompt cannot be empty"));
        }

        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(anyhow::anyhow!(
                "Temperature must be between 0.0 and 2.0, got {}",
                self.temperature
            ));
        }

        if let Some(top_p) = self.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(anyhow::anyhow!(
                "top_p must be between 0.0 and 1.0, got {}",
                top_p
            ));
        }

        Ok(())
    }
}

/// Builder pattern helper for complex request construction.
pub struct RequestBuilder {
    request: LLMRequest,
}

impl RequestBuilder {
    /// Create a new builder with the given prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        RequestBuilder {
            request: LLMRequest::new(prompt),
        }
    }

    /// Add system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.request = self.request.with_system_prompt(system);
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f64) -> Self {
        self.request = self.request.with_temperature(temp);
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.request = self.request.with_max_tokens(max);
        self
    }

    /// Set top_p.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.request = self.request.with_top_p(top_p);
        self
    }

    /// Add stop sequence.
    pub fn stop(mut self, stop: impl Into<String>) -> Self {
        self.request = self.request.add_stop_sequence(stop);
        self
    }

    /// Build the final request.
    pub fn build(self) -> anyhow::Result<LLMRequest> {
        self.request.validate()?;
        Ok(self.request)
    }
}

/// Reusable generation configuration template.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Config name for reference
    pub name: String,
    /// Temperature setting
    pub temperature: f64,
    /// Max tokens setting
    pub max_tokens: Option<usize>,
    /// Top P setting
    pub top_p: Option<f64>,
}

impl GenerationConfig {
    /// Create a new config with given name.
    pub fn new(name: impl Into<String>, temperature: f64) -> Self {
        GenerationConfig {
            name: name.into(),
            temperature: temperature.clamp(0.0, 2.0),
            max_tokens: None,
            top_p: None,
        }
    }

    /// Predefined: balanced generation
    pub fn balanced() -> Self {
        GenerationConfig::new("balanced", 0.7)
    }

    /// Predefined: creative generation
    pub fn creative() -> Self {
        GenerationConfig::new("creative", 1.2)
    }

    /// Predefined: deterministic generation
    pub fn deterministic() -> Self {
        GenerationConfig::new("deterministic", 0.0)
    }

    /// Apply config to a request.
    pub fn apply(self, mut request: LLMRequest) -> LLMRequest {
        request.temperature = self.temperature;
        if let Some(max_tokens) = self.max_tokens {
            request.max_tokens = Some(max_tokens);
        }
        if let Some(top_p) = self.top_p {
            request.top_p = Some(top_p);
        }
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let req = LLMRequest::new("Hello");
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.temperature, 0.7);
    }

    #[test]
    fn test_request_builder() -> Result<(), Box<dyn std::error::Error>> {
        let req = RequestBuilder::new("Test")
            .temperature(0.9)
            .max_tokens(500)
            .build()?;

        assert_eq!(req.temperature, 0.9);
        assert_eq!(req.max_tokens, Some(500));
        Ok(())
    }

    #[test]
    fn test_request_validation() {
        let req = LLMRequest::new("");
        assert!(req.validate().is_err());

        let req = LLMRequest {
            temperature: 3.0,
            ..LLMRequest::new("test")
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_generation_configs() {
        let balanced = GenerationConfig::balanced();
        assert_eq!(balanced.temperature, 0.7);

        let creative = GenerationConfig::creative();
        assert_eq!(creative.temperature, 1.2);

        let deterministic = GenerationConfig::deterministic();
        assert_eq!(deterministic.temperature, 0.0);
    }
}
