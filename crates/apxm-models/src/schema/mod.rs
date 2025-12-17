//! Schema validation and output parsing.
//!
//! Provides JSON Schema validation and parsing utilities for LLM outputs.

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;

/// JSON Schema validator for LLM output.
pub struct JsonSchema {
    schema: Value,
    validator: jsonschema::JSONSchema,
}

impl JsonSchema {
    /// Create schema from JSON value.
    pub fn from_value(schema: Value) -> Result<Self> {
        let validator = jsonschema::JSONSchema::compile(&schema)
            .map_err(|e| anyhow::anyhow!("Invalid JSON schema: {}", e))?;

        Ok(JsonSchema { schema, validator })
    }

    /// Create schema from JSON string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(schema_str: &str) -> Result<Self> {
        Self::parse_json_str(schema_str)
    }

    fn parse_json_str(schema_str: &str) -> Result<Self> {
        let schema: Value =
            serde_json::from_str(schema_str).context("Failed to parse schema JSON")?;
        Self::from_value(schema)
    }

    /// Validate a value against the schema.
    pub fn validate(&self, value: &Value) -> Result<()> {
        self.validator.validate(value).map_err(|errors| {
            let error_messages: Vec<String> = errors
                .map(|e| format!("{}: {}", e.instance_path, e))
                .collect();

            anyhow::anyhow!("Schema validation failed:\n{}", error_messages.join("\n"))
        })
    }

    /// Get the underlying schema.
    pub fn schema(&self) -> &Value {
        &self.schema
    }
}

impl std::str::FromStr for JsonSchema {
    type Err = anyhow::Error;

    fn from_str(schema_str: &str) -> Result<Self, Self::Err> {
        JsonSchema::parse_json_str(schema_str)
    }
}

/// Parser for extracting structured data from LLM output.
pub struct OutputParser;

impl OutputParser {
    /// Parse JSON from text, attempting multiple strategies.
    pub fn parse_json(text: &str) -> Result<Value> {
        // Strategy 1: Try direct JSON parsing
        if let Ok(value) = serde_json::from_str::<Value>(text) {
            return Ok(value);
        }

        // Strategy 2: Extract from markdown code block
        if let Ok(value) = Self::extract_from_markdown(text) {
            return Ok(value);
        }

        // Strategy 3: Find JSON object in text
        if let Ok(value) = Self::extract_json_from_text(text) {
            return Ok(value);
        }

        Err(anyhow::anyhow!(
            "Failed to parse JSON from text. Tried direct parsing, markdown extraction, and text extraction."
        ))
    }

    /// Extract JSON from markdown code block.
    pub fn extract_from_markdown(text: &str) -> Result<Value> {
        // Match ```json ... ``` or ``` ... ```
        let re =
            Regex::new(r"```(?:json)?\s*\n?([\s\S]*?)\n?```").context("Failed to create regex")?;

        let captures = re.captures(text).context("No markdown code block found")?;

        let json_str = captures
            .get(1)
            .context("No content in code block")?
            .as_str()
            .trim();

        serde_json::from_str(json_str).context("Failed to parse JSON from markdown block")
    }

    /// Extract JSON object or array from anywhere in text.
    fn extract_json_from_text(text: &str) -> Result<Value> {
        // Find JSON objects {...} or arrays [...]
        let object_re = Regex::new(r"\{[\s\S]*\}").context("Failed to create object regex")?;
        let array_re = Regex::new(r"\[[\s\S]*\]").context("Failed to create array regex")?;

        // Try object first
        if let Some(captures) = object_re.find(text) {
            if let Ok(value) = serde_json::from_str::<Value>(captures.as_str()) {
                return Ok(value);
            }
        }

        // Try array
        if let Some(captures) = array_re.find(text) {
            if let Ok(value) = serde_json::from_str::<Value>(captures.as_str()) {
                return Ok(value);
            }
        }

        Err(anyhow::anyhow!("No valid JSON found in text"))
    }

    /// Parse and validate against a schema.
    pub fn parse_with_schema(text: &str, schema: &JsonSchema) -> Result<Value> {
        let value = Self::parse_json(text)?;
        schema.validate(&value)?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_schema_validation() {
        let schema = JsonSchema::from_value(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        }))
        .unwrap();

        // Valid
        let valid = json!({"name": "Alice", "age": 30});
        assert!(schema.validate(&valid).is_ok());

        // Invalid - missing required field
        let invalid = json!({"age": 30});
        assert!(schema.validate(&invalid).is_err());

        // Invalid - wrong type
        let invalid = json!({"name": 123});
        assert!(schema.validate(&invalid).is_err());
    }

    #[test]
    fn test_parse_direct_json() {
        let text = r#"{"name": "Bob", "value": 42}"#;
        let result = OutputParser::parse_json(text).unwrap();
        assert_eq!(result["name"], "Bob");
        assert_eq!(result["value"], 42);
    }

    #[test]
    fn test_parse_markdown_json() {
        let text = r#"
Here's the result:
```json
{
  "status": "success",
  "count": 5
}
```
That's it!"#;

        let result = OutputParser::parse_json(text).unwrap();
        assert_eq!(result["status"], "success");
        assert_eq!(result["count"], 5);
    }

    #[test]
    fn test_parse_embedded_json() {
        let text = "The answer is {\"result\": 42, \"valid\": true} as you can see.";

        let result = OutputParser::parse_json(text).unwrap();
        assert_eq!(result["result"], 42);
        assert_eq!(result["valid"], true);
    }

    #[test]
    fn test_parse_with_schema() {
        let schema = JsonSchema::from_value(json!({
            "type": "object",
            "properties": {
                "result": {"type": "number"}
            },
            "required": ["result"]
        }))
        .unwrap();

        let text = "```json\n{\"result\": 42}\n```";
        let result = OutputParser::parse_with_schema(text, &schema).unwrap();
        assert_eq!(result["result"], 42);
    }
}
