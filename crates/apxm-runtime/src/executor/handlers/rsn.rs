//! RSN operation - Reasoning with LLM and structured output parsing
//!
//! Supports:
//! - LLM-based reasoning
//! - Structured output parsing (belief updates, goals, result)
//! - Retry logic with exponential backoff
//! - Context management

use super::{
    ExecutionContext, Node, Result, Value, get_optional_string_attribute, get_string_attribute,
};
use crate::aam::{Goal as AamGoal, GoalId, GoalStatus, TransitionLabel};
use apxm_core::error::RuntimeError;
use apxm_models::backends::request::LLMRequest;
use serde::de::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Structured output from RSN operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredRsnOutput {
    /// Belief updates to apply to memory
    #[serde(default)]
    pub belief_updates: HashMap<String, Value>,

    /// New goals to add
    #[serde(default)]
    pub new_goals: Vec<Goal>,

    /// The main result value
    pub result: Value,
}

/// Goal definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub description: String,
    #[serde(default = "default_priority")]
    pub priority: u32,
}

fn default_priority() -> u32 {
    50
}

/// Execute RSN operation - LLM-based reasoning with structured output
///
/// The RSN operation supports two modes:
/// 1. **Simple mode**: Returns plain text response
/// 2. **Structured mode**: Parses JSON response with belief_updates, new_goals, and result
///
/// # Structured Output Format
///
/// ```json
/// {
///   "belief_updates": {
///     "key1": "value1",
///     "key2": 42
///   },
///   "new_goals": [
///     {"description": "Goal 1", "priority": 80},
///     {"description": "Goal 2", "priority": 60}
///   ],
///   "result": "The reasoning result"
/// }
/// ```
pub async fn execute(ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    let prompt = get_string_attribute(node, "prompt")
        .or_else(|_| get_string_attribute(node, "template_str"))?;
    let model = get_optional_string_attribute(node, "model")?;
    let system_prompt = get_optional_string_attribute(node, "system_prompt")?;
    let max_retries = node
        .attributes
        .get("max_retries")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as u32;

    // Build LLM request
    let mut request = LLMRequest::new(prompt.clone());

    if let Some(sys) = system_prompt {
        request = request.with_system_prompt(sys);
    } else {
        // Use system prompt from apxm-prompts
        let system_prompt = apxm_prompts::render_prompt("rsn_system", &serde_json::json!({}))
            .unwrap_or_else(|_| {
                "You are a helpful AI assistant. When providing structured responses, \
                 use JSON format with fields: belief_updates (object), new_goals (array), \
                 and result (any type)."
                    .to_string()
            });
        request = request.with_system_prompt(system_prompt);
    }

    if let Some(model_name) = model {
        request = request.with_model(model_name);
    }

    // Execute with retries
    let mut last_error = None;
    for attempt in 0..=max_retries {
        if attempt > 0 {
            tracing::warn!(
                execution_id = %ctx.execution_id,
                attempt = attempt,
                "Retrying RSN operation"
            );

            // Exponential backoff
            let backoff_ms = 100 * 2_u64.pow(attempt - 1);
            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
        }

        match execute_rsn_once(ctx, node, &request).await {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(e);
                tracing::debug!(
                    execution_id = %ctx.execution_id,
                    attempt = attempt,
                    error = %last_error.as_ref().unwrap(),
                    "RSN attempt failed"
                );
            }
        }
    }

    Err(last_error.unwrap_or_else(|| RuntimeError::LLM {
        message: "All retry attempts exhausted".to_string(),
        backend: None,
    }))
}

/// Execute a single RSN attempt
async fn execute_rsn_once(
    ctx: &ExecutionContext,
    node: &Node,
    request: &LLMRequest,
) -> Result<Value> {
    // Execute LLM request through registry
    let response = ctx
        .llm_registry
        .generate(request.clone())
        .await
        .map_err(|e| RuntimeError::LLM {
            message: e.to_string(),
            backend: None,
        })?;

    let content = response.content;

    // Try to parse as structured output
    if let Ok(structured) = parse_structured_output(&content) {
        let label = TransitionLabel::operation(node.id, format!("{:?}", node.op_type));
        // Apply belief updates to LTM
        for (key, value) in structured.belief_updates {
            ctx.memory
                .write(crate::memory::MemorySpace::Ltm, key.clone(), value.clone())
                .await
                .ok(); // Don't fail on memory errors

            ctx.aam.set_belief(key, value, label.clone());
        }

        // Store goals in memory (for future retrieval)
        if !structured.new_goals.is_empty() {
            let goals_value = Value::Array(
                structured
                    .new_goals
                    .iter()
                    .map(|g| {
                        let mut obj = HashMap::new();
                        obj.insert(
                            "description".to_string(),
                            Value::String(g.description.clone()),
                        );
                        obj.insert(
                            "priority".to_string(),
                            Value::Number(apxm_core::types::values::Number::Integer(
                                g.priority as i64,
                            )),
                        );
                        Value::Object(obj)
                    })
                    .collect(),
            );

            ctx.memory
                .write(
                    crate::memory::MemorySpace::Stm,
                    format!("goals:{}", ctx.execution_id),
                    goals_value,
                )
                .await
                .ok();

            for goal in &structured.new_goals {
                let aam_goal = AamGoal {
                    id: GoalId::new(),
                    description: goal.description.clone(),
                    priority: goal.priority,
                    status: GoalStatus::Active,
                };
                ctx.aam.add_goal(aam_goal, label.clone());
            }
        }

        // Return the result value
        Ok(structured.result)
    } else {
        // Fall back to plain text response
        Ok(Value::String(content))
    }
}

/// Parse structured output from LLM response
fn parse_structured_output(
    content: &str,
) -> std::result::Result<StructuredRsnOutput, serde_json::Error> {
    // Try to find JSON in the content
    let trimmed = content.trim();

    // Try direct parse first
    if let Ok(output) = serde_json::from_str::<StructuredRsnOutput>(trimmed) {
        return Ok(output);
    }

    // Try to extract JSON from markdown code block
    if let Some(json_str) = extract_json_from_markdown(trimmed) {
        if let Ok(output) = serde_json::from_str::<StructuredRsnOutput>(&json_str) {
            return Ok(output);
        }
    }

    // If all parsing fails, return error
    Err(serde_json::Error::custom(
        "Failed to parse structured output",
    ))
}

/// Extract JSON from markdown code block
fn extract_json_from_markdown(content: &str) -> Option<String> {
    // Look for ```json ... ``` or ``` ... ```
    if let Some(start) = content.find("```json") {
        if let Some(end) = content[start + 7..].find("```") {
            return Some(content[start + 7..start + 7 + end].trim().to_string());
        }
    }

    if let Some(start) = content.find("```") {
        if let Some(end) = content[start + 3..].find("```") {
            let extracted = content[start + 3..start + 3 + end].trim();
            // Only return if it looks like JSON
            if extracted.starts_with('{') || extracted.starts_with('[') {
                return Some(extracted.to_string());
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_structured_output() {
        let json = r#"{
            "belief_updates": {"key1": "value1"},
            "new_goals": [{"description": "Test goal", "priority": 80}],
            "result": "Test result"
        }"#;

        let output = parse_structured_output(json).unwrap();
        assert_eq!(output.belief_updates.len(), 1);
        assert_eq!(output.new_goals.len(), 1);
        assert_eq!(output.result, Value::String("Test result".to_string()));
    }

    #[test]
    fn test_parse_markdown_wrapped_json() {
        let content = r#"Here is the result:

```json
{
    "belief_updates": {},
    "new_goals": [],
    "result": "Markdown wrapped"
}
```

That's all."#;

        let output = parse_structured_output(content).unwrap();
        assert_eq!(output.result, Value::String("Markdown wrapped".to_string()));
    }

    #[test]
    fn test_extract_json_from_markdown() {
        let content = "```json\n{\"test\": 123}\n```";
        let extracted = extract_json_from_markdown(content).unwrap();
        assert_eq!(extracted, "{\"test\": 123}");
    }
}
