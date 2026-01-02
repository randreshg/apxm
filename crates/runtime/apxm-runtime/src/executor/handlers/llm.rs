//! LLM operations handler - Unified handler for Ask, Think, Reason
//!
//! Three operation types, one underlying LLM executor:
//! - Ask: Simple Q&A (LOW latency) - returns plain text
//! - Think: Extended thinking with budget (HIGH latency) - deep reasoning
//! - Reason: Structured reasoning (MEDIUM latency) - belief/goal updates
//!
//! The operation type serves as a marker for runtime config lookup.
//! Actual LLM parameters come from runtime configuration.

use super::{
    ExecutionContext, Node, Result, Value, execute_llm_request, get_optional_string_attribute,
    get_optional_u64_attribute, get_string_attribute,
    inner_plan::{InnerPlanOptions, execute_inner_plan},
};
use crate::aam::{Goal as AamGoal, GoalId, GoalStatus, TransitionLabel};
use apxm_backends::LLMRequest;
use apxm_core::InnerPlanDsl;
use apxm_core::apxm_llm;
use apxm_core::error::RuntimeError;
use apxm_core::types::operations::AISOperationType;
use serde::de::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LLM operation mode (derived from operation type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmMode {
    /// Simple Q&A - no extended thinking, plain text response
    Ask,
    /// Extended thinking with budget_tokens
    Think,
    /// Structured reasoning with belief/goal updates
    Reason,
}

impl From<&AISOperationType> for LlmMode {
    fn from(op_type: &AISOperationType) -> Self {
        match op_type {
            AISOperationType::Ask => LlmMode::Ask,
            AISOperationType::Think => LlmMode::Think,
            AISOperationType::Reason => LlmMode::Reason,
            _ => LlmMode::Ask, // Fallback for unexpected types
        }
    }
}

/// Structured output from Reason operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredReasonOutput {
    /// Belief updates to apply to memory
    #[serde(default)]
    pub belief_updates: HashMap<String, Value>,

    /// New goals to add
    #[serde(default)]
    pub new_goals: Vec<Goal>,

    /// Optional inner plan emitted by the model
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_plan: Option<InnerPlanDsl>,

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

/// Execute LLM operation - unified handler for Ask, Think, Reason
///
/// # Mode Behavior
///
/// - **Ask**: Simple Q&A, returns plain text, no structured parsing
/// - **Think**: Extended thinking with budget_tokens, uses thinking mode
/// - **Reason**: Structured output with belief_updates, new_goals, inner_plan
pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let mode = LlmMode::from(&node.op_type);
    let mode_name = match mode {
        LlmMode::Ask => "ASK",
        LlmMode::Think => "THINK",
        LlmMode::Reason => "REASON",
    };

    let base_prompt = get_string_attribute(node, "template_str")
        .or_else(|_| get_string_attribute(node, "prompt"))?;
    let model = get_optional_string_attribute(node, "model")?;
    let budget = get_optional_u64_attribute(node, "budget")?;
    let max_retries = node
        .attributes
        .get("max_retries")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as u32;

    // Inner plan support only for Reason mode
    let supports_inner_plan = mode == LlmMode::Reason
        && node
            .attributes
            .get("inner_plan_supported")
            .and_then(|v| v.as_boolean())
            .unwrap_or(false);
    let enable_inner_plan = mode == LlmMode::Reason
        && node
            .attributes
            .get("enable_inner_plan")
            .and_then(|v| v.as_boolean())
            .unwrap_or(supports_inner_plan);
    let bind_outputs = node
        .attributes
        .get("bind_inner_plan_outputs")
        .and_then(|v| v.as_boolean())
        .unwrap_or(true);

    // Build prompt with context
    let mut prompt = base_prompt.clone();
    if !inputs.is_empty() {
        let mut ctx_lines = Vec::new();
        for (idx, value) in inputs.iter().enumerate() {
            let rendered = value
                .to_json()
                .map(|j| j.to_string())
                .unwrap_or_else(|_| value.to_string());
            ctx_lines.push(format!("Context {}: {}", idx + 1, rendered));
        }
        prompt = format!("{}\n\n{}", base_prompt, ctx_lines.join("\n"));
    }

    let mut request = LLMRequest::new(prompt.clone());

    // Apply mode-specific configuration
    match mode {
        LlmMode::Ask => {
            // Simple Q&A - minimal system prompt
            let system_prompt = apxm_backends::render_prompt("ask_system", &serde_json::json!({}))
                .unwrap_or_else(|_| "You are a helpful AI assistant. Answer concisely.".to_string());
            request = request.with_system_prompt(system_prompt);
        }
        LlmMode::Think => {
            // Extended thinking - set budget via metadata for backends that support it
            if let Some(budget_tokens) = budget {
                request = request.with_metadata_value(
                    "thinking_budget",
                    serde_json::json!(budget_tokens),
                );
            }
            let system_prompt = apxm_backends::render_prompt("think_system", &serde_json::json!({}))
                .unwrap_or_else(|_| {
                    "You are a deep reasoning AI. Think through problems carefully and thoroughly."
                        .to_string()
                });
            request = request.with_system_prompt(system_prompt);
        }
        LlmMode::Reason => {
            // Structured reasoning - request JSON output
            let system_prompt = apxm_backends::render_prompt("reason_system", &serde_json::json!({}))
                .unwrap_or_else(|_| {
                    "You are a helpful AI assistant. When providing structured responses, \
                     use JSON format with fields: belief_updates (object), new_goals (array), \
                     and result (any type)."
                        .to_string()
                });
            request = request.with_system_prompt(system_prompt);
        }
    }

    if let Some(model_name) = model {
        request = request.with_model(model_name);
    }

    // Execute with retries
    let mut last_error = None;
    for attempt in 0..=max_retries {
        if attempt > 0 {
            apxm_llm!(warn,
                execution_id = %ctx.execution_id,
                mode = mode_name,
                attempt = attempt,
                "Retrying LLM operation"
            );

            // Exponential backoff
            let backoff_ms = 100 * 2_u64.pow(attempt - 1);
            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
        }

        match execute_llm_once(ctx, node, &request, mode, enable_inner_plan, bind_outputs).await {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(e);
                apxm_llm!(debug,
                    execution_id = %ctx.execution_id,
                    mode = mode_name,
                    attempt = attempt,
                    error = %last_error.as_ref().unwrap(),
                    "LLM attempt failed"
                );
            }
        }
    }

    Err(last_error.unwrap_or_else(|| RuntimeError::LLM {
        message: format!("{} operation: all retry attempts exhausted", mode_name),
        backend: None,
    }))
}

/// Execute a single LLM attempt
async fn execute_llm_once(
    ctx: &ExecutionContext,
    node: &Node,
    request: &LLMRequest,
    mode: LlmMode,
    enable_inner_plan: bool,
    bind_outputs: bool,
) -> Result<Value> {
    let mode_name = match mode {
        LlmMode::Ask => "ASK",
        LlmMode::Think => "THINK",
        LlmMode::Reason => "REASON",
    };

    apxm_llm!(debug,
        execution_id = %ctx.execution_id,
        mode = mode_name,
        prompt_len = request.prompt.len(),
        "Sending LLM request"
    );

    // Execute LLM request through registry
    let response = execute_llm_request(ctx, mode_name, request).await?;
    let content = response.content;

    apxm_llm!(info,
        execution_id = %ctx.execution_id,
        mode = mode_name,
        response_len = content.len(),
        tokens_in = response.usage.input_tokens,
        tokens_out = response.usage.output_tokens,
        "LLM response received"
    );

    apxm_llm!(trace,
        execution_id = %ctx.execution_id,
        mode = mode_name,
        raw_response = %content,
        "LLM model response content"
    );

    // Process response based on mode
    match mode {
        LlmMode::Ask | LlmMode::Think => {
            // Plain text response for Ask and Think
            Ok(Value::String(content))
        }
        LlmMode::Reason => {
            // Try to parse as structured output for Reason
            if let Ok(structured) = parse_structured_output(&content) {
                process_structured_output(ctx, node, structured, enable_inner_plan, bind_outputs)
                    .await
            } else {
                // Fall back to plain text response
                Ok(Value::String(content))
            }
        }
    }
}

/// Process structured output from Reason mode
async fn process_structured_output(
    ctx: &ExecutionContext,
    node: &Node,
    structured: StructuredReasonOutput,
    enable_inner_plan: bool,
    bind_outputs: bool,
) -> Result<Value> {
    let label = TransitionLabel::operation(node.id, format!("{:?}", node.op_type));

    // Apply belief updates to LTM
    for (key, value) in structured.belief_updates {
        ctx.memory
            .write(crate::memory::MemorySpace::Ltm, key.clone(), value.clone())
            .await
            .ok(); // Don't fail on memory errors

        ctx.aam.set_belief(key, value, label.clone());
    }

    // Store goals in memory
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
                        Value::Number(apxm_core::types::values::Number::Integer(g.priority as i64)),
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

    // Execute inner plan if enabled
    if enable_inner_plan {
        if let Some(inner_plan) = structured.inner_plan.clone() {
            apxm_llm!(info,
                execution_id = %ctx.execution_id,
                "REASON provided inner plan DSL"
            );

            let inserted_nodes = execute_inner_plan(
                ctx,
                node,
                &inner_plan,
                InnerPlanOptions {
                    bind_outer_outputs: bind_outputs,
                },
            )
            .await?;

            if inserted_nodes > 0 {
                apxm_llm!(debug,
                    execution_id = %ctx.execution_id,
                    inserted_nodes = inserted_nodes,
                    "Inner plan merged into DAG"
                );
                ctx.memory
                    .write(
                        crate::memory::MemorySpace::Episodic,
                        format!("inner_plan_spliced:{}", ctx.execution_id),
                        Value::String(format!(
                            "REASON inner plan merged into DAG with {} nodes",
                            inserted_nodes
                        )),
                    )
                    .await
                    .ok();
            }
        } else {
            apxm_llm!(trace,
                execution_id = %ctx.execution_id,
                "REASON inner plan enabled but model omitted DSL"
            );
        }
    }

    // Return the result value
    Ok(structured.result)
}

/// Parse structured output from LLM response
fn parse_structured_output(
    content: &str,
) -> std::result::Result<StructuredReasonOutput, serde_json::Error> {
    // Try to find JSON in the content
    let trimmed = content.trim();

    // Try direct parse first
    if let Ok(output) = serde_json::from_str::<StructuredReasonOutput>(trimmed) {
        return Ok(output);
    }

    // Try to extract JSON from markdown code block
    if let Some(json_str) = extract_json_from_markdown(trimmed)
        && let Ok(output) = serde_json::from_str::<StructuredReasonOutput>(&json_str)
    {
        return Ok(output);
    }

    // If all parsing fails, return error
    Err(serde_json::Error::custom(
        "Failed to parse structured output",
    ))
}

/// Extract JSON from markdown code block
fn extract_json_from_markdown(content: &str) -> Option<String> {
    // Look for ```json ... ``` or ``` ... ```
    if let Some(start) = content.find("```json")
        && let Some(end) = content[start + 7..].find("```")
    {
        return Some(content[start + 7..start + 7 + end].trim().to_string());
    }

    if let Some(start) = content.find("```")
        && let Some(end) = content[start + 3..].find("```")
    {
        let extracted = content[start + 3..start + 3 + end].trim();
        // Only return if it looks like JSON
        if extracted.starts_with('{') || extracted.starts_with('[') {
            return Some(extracted.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_mode_from_op_type() {
        assert_eq!(LlmMode::from(&AISOperationType::Ask), LlmMode::Ask);
        assert_eq!(LlmMode::from(&AISOperationType::Think), LlmMode::Think);
        assert_eq!(LlmMode::from(&AISOperationType::Reason), LlmMode::Reason);
    }

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

    #[test]
    fn test_parse_output_with_inner_plan() {
        let json = r#"{
            "belief_updates": {},
            "new_goals": [],
            "inner_plan": {"dsl": "agent InnerPlan { }"},
            "result": "ok"
        }"#;

        let output = parse_structured_output(json).unwrap();
        assert!(output.inner_plan.is_some());
    }
}
