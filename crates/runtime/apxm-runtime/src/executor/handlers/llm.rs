//! LLM operations handler - Unified handler for Ask, Think, Reason
//!
//! Three operation types, one underlying LLM executor:
//! - Ask: Simple Q&A (LOW latency) - returns plain text
//! - Think: Extended thinking with budget (HIGH latency) - deep reasoning
//! - Reason: Structured reasoning (MEDIUM latency) - belief/goal updates
//!
//! The operation type serves as a marker for runtime config lookup.
//! Actual LLM parameters come from runtime configuration.
//!
//! ## Tool Support (V1)
//!
//! The Ask operation supports tool usage via a tool loop:
//! 1. Send prompt + tool schemas to LLM
//! 2. LLM returns tool_calls (or text)
//! 3. Execute tool calls via CapabilitySystem
//! 4. Feed results back to LLM
//! 5. Repeat until LLM returns text (no tool calls)

use super::{
    ExecutionContext, Node, Result, Value, execute_llm_request, get_optional_string_attribute,
    get_optional_u64_attribute, get_string_attribute,
    inner_plan::{InnerPlanOptions, execute_inner_plan},
};
use crate::aam::{Goal as AamGoal, GoalId, GoalStatus, TransitionLabel};
use apxm_backends::{LLMRequest, ToolChoice, ToolDefinition};
use apxm_core::InnerPlanDsl;
use apxm_core::apxm_llm;
use apxm_core::error::RuntimeError;
use apxm_core::types::operations::AISOperationType;
use apxm_core::types::{ToolCall, ToolResult};
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

/// Maximum number of tool loop iterations to prevent infinite loops
const MAX_TOOL_ITERATIONS: usize = 10;

/// Get tool definitions from the capability system for LLM requests
fn get_tool_definitions_from_capabilities(ctx: &ExecutionContext) -> Vec<ToolDefinition> {
    ctx.capability_system
        .list_capabilities()
        .into_iter()
        .map(|meta| ToolDefinition::new(&meta.name, &meta.description, meta.parameters_schema))
        .collect()
}

/// Get specific tools by name from the capability system
fn get_tools_by_names(ctx: &ExecutionContext, names: &[String]) -> Vec<ToolDefinition> {
    names
        .iter()
        .filter_map(|name| {
            ctx.capability_system.get_metadata(name).map(|meta| {
                ToolDefinition::new(&meta.name, &meta.description, meta.parameters_schema)
            })
        })
        .collect()
}

/// Execute a single tool call via the capability system
async fn execute_tool_call(ctx: &ExecutionContext, tool_call: &ToolCall) -> ToolResult {
    apxm_llm!(debug,
        execution_id = %ctx.execution_id,
        tool_name = %tool_call.name,
        tool_id = %tool_call.id,
        "Executing tool call"
    );

    // Convert JSON args to HashMap<String, Value>
    let args: HashMap<String, Value> = match &tool_call.args {
        serde_json::Value::Object(obj) => obj
            .iter()
            .map(|(k, v)| (k.clone(), json_to_value(v)))
            .collect(),
        _ => HashMap::new(),
    };

    // Invoke the capability
    match ctx.capability_system.invoke(&tool_call.name, args).await {
        Ok(result) => {
            let content = match result {
                Value::String(s) => s,
                other => other.to_string(),
            };
            apxm_llm!(info,
                execution_id = %ctx.execution_id,
                tool_name = %tool_call.name,
                "Tool call succeeded"
            );
            ToolResult::success(&tool_call.id, content)
        }
        Err(e) => {
            apxm_llm!(warn,
                execution_id = %ctx.execution_id,
                tool_name = %tool_call.name,
                error = %e,
                "Tool call failed"
            );
            ToolResult::error(&tool_call.id, e.to_string())
        }
    }
}

/// Convert serde_json::Value to apxm_core Value
fn json_to_value(json: &serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(apxm_core::types::values::Number::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Value::Number(apxm_core::types::values::Number::Float(f))
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::Array(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        ),
    }
}

/// Format tool results as a message for the LLM
fn format_tool_results_message(results: &[ToolResult]) -> String {
    results
        .iter()
        .map(|r| {
            if r.success {
                format!(
                    "<tool_result id=\"{}\">\n{}\n</tool_result>",
                    r.tool_call_id, r.content
                )
            } else {
                format!(
                    "<tool_error id=\"{}\">\n{}\n</tool_error>",
                    r.tool_call_id, r.content
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
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
        // Check for {n} placeholder pattern (from BuildPrompt pass)
        if base_prompt.contains("{0}") {
            // Substitute placeholders with context values
            prompt = base_prompt.clone();
            for (idx, value) in inputs.iter().enumerate() {
                let placeholder = format!("{{{}}}", idx);
                if prompt.contains(&placeholder) {
                    let rendered = value
                        .as_string()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| value.to_string());
                    prompt = prompt.replace(&placeholder, &rendered);
                }
            }
        } else if !base_prompt.trim().is_empty() {
            // Original behavior - append context as labeled lines
            let mut ctx_lines = Vec::new();
            for (idx, value) in inputs.iter().enumerate() {
                let rendered = value
                    .to_json()
                    .map(|j| j.to_string())
                    .unwrap_or_else(|_| value.to_string());
                ctx_lines.push(format!("Context {}: {}", idx + 1, rendered));
            }
            prompt = format!("{}\n\n{}", base_prompt, ctx_lines.join("\n"));
        } else {
            // Empty template without placeholder - use first context as prompt
            // (backward compat for artifacts compiled before BuildPrompt pass)
            prompt = inputs[0]
                .as_string()
                .map(|s| s.to_string())
                .unwrap_or_else(|| inputs[0].to_string());
        }
    }

    let mut request = LLMRequest::new(prompt.clone());

    // Apply mode-specific configuration
    match mode {
        LlmMode::Ask => {
            // Simple Q&A - minimal system prompt
            // Priority: 1) node attribute (agent context), 2) config instruction, 3) template, 4) hardcoded fallback
            let system_prompt = get_optional_string_attribute(node, "system_prompt")?
                .or_else(|| ctx.instruction_config.ask.clone())
                .or_else(|| apxm_backends::render_prompt("ask_system", &serde_json::json!({})).ok())
                .unwrap_or_else(|| "You are a helpful AI assistant. Answer concisely.".to_string());
            request = request.with_system_prompt(system_prompt);
        }
        LlmMode::Think => {
            // Extended thinking - set budget via metadata for backends that support it
            if let Some(budget_tokens) = budget {
                request = request
                    .with_metadata_value("thinking_budget", serde_json::json!(budget_tokens));
            }
            // Priority: 1) config instruction, 2) template, 3) hardcoded fallback
            let system_prompt = ctx
                .instruction_config
                .think
                .clone()
                .or_else(|| {
                    apxm_backends::render_prompt("think_system", &serde_json::json!({})).ok()
                })
                .unwrap_or_else(|| {
                    "You are a deep reasoning AI. Think through problems carefully and thoroughly."
                        .to_string()
                });
            request = request.with_system_prompt(system_prompt);
        }
        LlmMode::Reason => {
            // Structured reasoning - request JSON output
            // Priority: 1) config instruction, 2) template, 3) hardcoded fallback
            let system_prompt = ctx
                .instruction_config
                .reason
                .clone()
                .or_else(|| {
                    apxm_backends::render_prompt("reason_system", &serde_json::json!({})).ok()
                })
                .unwrap_or_else(|| {
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

    // Tool configuration (Ask mode only)
    let tools_enabled = mode == LlmMode::Ask
        && node
            .attributes
            .get("tools_enabled")
            .and_then(|v| v.as_boolean())
            .unwrap_or(true); // Enable by default for Ask

    if tools_enabled && mode == LlmMode::Ask {
        // Get tool names from node attributes, or use all registered capabilities
        let tool_names: Option<Vec<String>> = node
            .attributes
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_string().map(|s| s.to_string()))
                    .collect()
            });

        let tools = match tool_names {
            Some(names) if !names.is_empty() => get_tools_by_names(ctx, &names),
            _ => get_tool_definitions_from_capabilities(ctx),
        };

        if !tools.is_empty() {
            apxm_llm!(debug,
                execution_id = %ctx.execution_id,
                tool_count = tools.len(),
                "Attaching tools to ASK request"
            );
            request = request.with_tools(tools).with_tool_choice(ToolChoice::Auto);
        }
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

    // For Ask mode with tools, use the tool loop
    if mode == LlmMode::Ask && request.has_tools() {
        return execute_ask_with_tools(ctx, request).await;
    }

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

/// Execute Ask operation with tool loop
///
/// This implements the tool use cycle:
/// 1. Send prompt + tool schemas to LLM
/// 2. LLM returns tool_calls (or text)
/// 3. Execute tool calls via CapabilitySystem
/// 4. Feed results back to LLM
/// 5. Repeat until LLM returns text (no tool calls)
async fn execute_ask_with_tools(
    ctx: &ExecutionContext,
    initial_request: &LLMRequest,
) -> Result<Value> {
    let mut current_request = initial_request.clone();
    let mut accumulated_tool_results: Vec<ToolResult> = Vec::new();
    let mut total_input_tokens = 0usize;
    let mut total_output_tokens = 0usize;

    for iteration in 0..MAX_TOOL_ITERATIONS {
        apxm_llm!(debug,
            execution_id = %ctx.execution_id,
            iteration = iteration,
            prompt_len = current_request.prompt.len(),
            tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            "Sending ASK request with tools"
        );

        // Execute LLM request
        let response = execute_llm_request(ctx, "ASK", &current_request).await?;

        total_input_tokens += response.usage.input_tokens;
        total_output_tokens += response.usage.output_tokens;

        apxm_llm!(info,
            execution_id = %ctx.execution_id,
            iteration = iteration,
            response_len = response.content.len(),
            tool_calls = response.tool_calls.len(),
            tokens_in = response.usage.input_tokens,
            tokens_out = response.usage.output_tokens,
            "ASK response received"
        );

        // If no tool calls, return the text response
        if response.tool_calls.is_empty() {
            apxm_llm!(info,
                execution_id = %ctx.execution_id,
                iterations = iteration + 1,
                total_tokens_in = total_input_tokens,
                total_tokens_out = total_output_tokens,
                tools_invoked = accumulated_tool_results.len(),
                "ASK tool loop completed"
            );
            return Ok(Value::String(response.content));
        }

        // Execute each tool call
        let mut tool_results = Vec::new();
        for tool_call in &response.tool_calls {
            let result = execute_tool_call(ctx, tool_call).await;
            tool_results.push(result);
        }

        // Store tool results in STM for later reference
        if !tool_results.is_empty() {
            let results_value = Value::Array(
                tool_results
                    .iter()
                    .map(|r| {
                        let mut obj = HashMap::new();
                        obj.insert(
                            "tool_call_id".to_string(),
                            Value::String(r.tool_call_id.clone()),
                        );
                        obj.insert("content".to_string(), Value::String(r.content.clone()));
                        obj.insert("success".to_string(), Value::Bool(r.success));
                        Value::Object(obj)
                    })
                    .collect(),
            );

            ctx.memory
                .write(
                    crate::memory::MemorySpace::Stm,
                    format!("tool_results:{}:{}", ctx.execution_id, iteration),
                    results_value,
                )
                .await
                .ok();
        }

        accumulated_tool_results.extend(tool_results.clone());

        // Build continuation prompt with tool results
        let tool_results_message = format_tool_results_message(&tool_results);
        let continuation_prompt = format!(
            "{}\n\n{}\n\nBased on the tool results above, please continue.",
            current_request.prompt, tool_results_message
        );

        // Update request for next iteration
        current_request = LLMRequest::new(continuation_prompt)
            .with_system_prompt(current_request.system_prompt.clone().unwrap_or_default())
            .with_temperature(current_request.temperature);

        // Keep tools available for subsequent calls
        if let Some(tools) = &initial_request.tools {
            current_request = current_request.with_tools(tools.clone());
        }
        if let Some(choice) = &initial_request.tool_choice {
            current_request = current_request.with_tool_choice(choice.clone());
        }
        if let Some(model) = &initial_request.model {
            current_request = current_request.with_model(model.clone());
        }
    }

    // Max iterations exceeded
    apxm_llm!(warn,
        execution_id = %ctx.execution_id,
        max_iterations = MAX_TOOL_ITERATIONS,
        "ASK tool loop exceeded max iterations"
    );

    Err(RuntimeError::LLM {
        message: format!(
            "Tool loop exceeded maximum iterations ({}). Last {} tool calls executed.",
            MAX_TOOL_ITERATIONS,
            accumulated_tool_results.len()
        ),
        backend: None,
    })
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

    #[test]
    fn test_json_to_value() {
        // Test string
        let json_str = serde_json::json!("hello");
        assert_eq!(json_to_value(&json_str), Value::String("hello".to_string()));

        // Test number
        let json_num = serde_json::json!(42);
        assert!(matches!(json_to_value(&json_num), Value::Number(_)));

        // Test bool
        let json_bool = serde_json::json!(true);
        assert_eq!(json_to_value(&json_bool), Value::Bool(true));

        // Test null
        let json_null = serde_json::json!(null);
        assert_eq!(json_to_value(&json_null), Value::Null);

        // Test array
        let json_arr = serde_json::json!([1, 2, 3]);
        let val = json_to_value(&json_arr);
        assert!(matches!(val, Value::Array(_)));

        // Test object
        let json_obj = serde_json::json!({"key": "value"});
        let val = json_to_value(&json_obj);
        assert!(matches!(val, Value::Object(_)));
    }

    #[test]
    fn test_format_tool_results_message() {
        let results = vec![
            ToolResult::success("call_1", "file1.txt\nfile2.txt"),
            ToolResult::error("call_2", "Permission denied"),
        ];

        let message = format_tool_results_message(&results);

        assert!(message.contains("<tool_result id=\"call_1\">"));
        assert!(message.contains("file1.txt"));
        assert!(message.contains("<tool_error id=\"call_2\">"));
        assert!(message.contains("Permission denied"));
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("id123", "output");
        assert_eq!(result.tool_call_id, "id123");
        assert_eq!(result.content, "output");
        assert!(result.success);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("id123", "error message");
        assert_eq!(result.tool_call_id, "id123");
        assert_eq!(result.content, "error message");
        assert!(!result.success);
    }
}
