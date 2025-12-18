//! PLAN operation - Planning with LLM and inner/outer plan execution
//!
//! Supports:
//! - LLM-based planning
//! - Inner plan (sub-DAG execution)
//! - Outer plan (continuation after inner plan)
//! - Structured plan output
//! - Goal-oriented planning

use super::{
    ExecutionContext, Node, Result, Value, get_optional_string_attribute, get_string_attribute,
    inner_plan::{InnerPlanDsl, InnerPlanOptions, execute_inner_plan},
    llm_error,
};
use crate::aam::{Goal as AamGoal, GoalId, GoalStatus, TransitionLabel};
use apxm_core::error::RuntimeError;
use apxm_models::backends::request::LLMRequest;
use serde::de::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Structured plan output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOutput {
    /// The generated plan (steps or strategy)
    pub plan: Vec<PlanStep>,

    /// Optional inner plan (DSL to be compiled and executed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_plan: Option<InnerPlanDsl>,

    /// Result/summary of the plan
    pub result: String,
}

/// A single step in a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub description: String,
    #[serde(default)]
    pub priority: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Vec<String>>,
}

/// Execute PLAN operation - LLM-based planning with inner/outer plan support
///
/// The PLAN operation supports:
/// - **Simple planning**: Returns a list of steps
/// - **Inner plan execution**: Executes a sub-DAG and continues with outer plan
/// - **Goal-based planning**: Generates plans to achieve specific goals
///
/// # Inner/Outer Plan Flow
///
/// 1. **Generate outer plan** (high-level strategy)
/// 2. **If inner_plan exists**: Execute the inner DAG
/// 3. **Return to outer plan**: Continue with remaining steps
///
/// # Structured Output Format
///
/// ```json
/// {
///   "plan": [
///     {"description": "Step 1", "priority": 90},
///     {"description": "Step 2", "priority": 80, "dependencies": ["Step 1"]}
///   ],
///   "result": "Plan summary"
/// }
/// ```
pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let _ = inputs;
    let goal = get_string_attribute(node, "goal")?;
    let model = get_optional_string_attribute(node, "model")?;
    let context_key = get_optional_string_attribute(node, "context_key")?;
    let supports_inner_plan = node
        .attributes
        .get("inner_plan_supported")
        .and_then(|v| v.as_boolean())
        .unwrap_or(false);
    let enable_inner_plan = node
        .attributes
        .get("enable_inner_plan")
        .and_then(|v| v.as_boolean())
        .unwrap_or(supports_inner_plan);
    let bind_outputs = node
        .attributes
        .get("bind_inner_plan_outputs")
        .and_then(|v| v.as_boolean())
        .unwrap_or(true);

    // Retrieve context from memory if specified
    let mut context_info = String::new();
    if let Some(key) = context_key {
        if let Ok(Some(value)) = ctx.memory.read(crate::memory::MemorySpace::Ltm, &key).await {
            context_info = format!("\n\nContext: {:?}", value);
        }
    }

    // Build planning prompt using apxm-prompts
    let prompt_context = serde_json::json!({
        "goal": goal,
        "context": context_info,
    });

    let planning_prompt = if enable_inner_plan {
        apxm_prompts::render_prompt("plan_multi_level", &prompt_context).unwrap_or_else(|_| {
            format!(
                "Create a detailed multi-level plan to achieve the following goal:\n\n{}\
                     {}\n\nProvide both:\n\
                     1. High-level strategy (outer plan)\n\
                     2. Detailed sub-tasks (can become inner plan)\n\n\
                     Respond in JSON format with: plan (array of steps) and result (summary).",
                goal, context_info
            )
        })
    } else {
        apxm_prompts::render_prompt("plan_system", &prompt_context)
            .unwrap_or_else(|_| {
                format!(
                    "Create a detailed plan to achieve the following goal:\n\n{}\
                     {}\n\nRespond in JSON format with: plan (array of steps with description and priority) and result (summary).",
                    goal, context_info
                )
            })
    };

    // Combine goal and context into planning_prompt for the LLM
    let final_prompt = format!("Goal: {}\n\n{}", goal, planning_prompt);

    // Build LLM request
    let mut request = LLMRequest::new(final_prompt);

    if let Some(m) = &model {
        request = request.with_model(m.clone());
    }

    // Use system prompt from apxm-prompts
    let system_prompt = if enable_inner_plan {
        apxm_prompts::render_prompt("plan_multi_level", &serde_json::json!({})).unwrap_or_else(
            |_| {
                "You are an expert planning assistant. Generate structured, actionable plans. \
                 Always respond in valid JSON format."
                    .to_string()
            },
        )
    } else {
        apxm_prompts::render_prompt("plan_system", &serde_json::json!({})).unwrap_or_else(|_| {
            "You are an expert planning assistant. Generate structured, actionable plans. \
                 Always respond in valid JSON format."
                .to_string()
        })
    };
    request = request.with_system_prompt(system_prompt);

    // Execute with retries
    let mut last_error = None;
    for attempt in 0..=3 {
        if attempt > 0 {
            tracing::warn!(
                execution_id = %ctx.execution_id,
                attempt = attempt,
                "Retrying PLAN operation"
            );

            let backoff_ms = 100 * 2_u64.pow(attempt - 1);
            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
        }

        match execute_plan_once(
            ctx,
            node,
            &request,
            enable_inner_plan,
            bind_outputs,
            &goal,
            model.as_deref(),
        )
        .await
        {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(e);
                tracing::debug!(
                    execution_id = %ctx.execution_id,
                    attempt = attempt,
                    error = %last_error.as_ref().unwrap(),
                    "PLAN attempt failed"
                );
            }
        }
    }

    Err(last_error.unwrap_or_else(|| RuntimeError::LLM {
        message: "All retry attempts exhausted for PLAN".to_string(),
        backend: None,
    }))
}

/// Execute a single PLAN attempt
async fn execute_plan_once(
    ctx: &ExecutionContext,
    node: &Node,
    request: &LLMRequest,
    enable_inner_plan: bool,
    bind_outputs: bool,
    original_goal: &str,
    model_override: Option<&str>,
) -> Result<Value> {
    let transition_label = TransitionLabel::operation(node.id, format!("{:?}", node.op_type));
    // Execute LLM request
    let response = ctx
        .llm_registry
        .generate(request.clone())
        .await
        .map_err(|e| llm_error(ctx, "PLAN", request, e))?;

    let content = response.content;
    tracing::info!(
        execution_id = %ctx.execution_id,
        raw_response = %content,
        "PLAN model response"
    );

    // Try to parse as structured plan
    if let Ok(mut plan_output) = parse_plan_output(&content) {
        if enable_inner_plan {
            match generate_inner_plan(ctx, node, &plan_output, original_goal, model_override).await
            {
                Ok(Some(dsl)) => {
                    plan_output.inner_plan = Some(InnerPlanDsl { dsl });
                }
                Ok(None) => {
                    tracing::warn!(
                        execution_id = %ctx.execution_id,
                        "Inner plan was requested but the model did not provide DSL"
                    );
                }
                Err(err) => {
                    tracing::error!(
                        execution_id = %ctx.execution_id,
                        error = %err,
                        "Inner plan generation failed"
                    );
                }
            }
        }

        // Store plan in memory
        let plan_json = serde_json::to_value(&plan_output.plan)
            .map_err(|e| RuntimeError::Serialization(format!("Failed to serialize plan: {}", e)))?;
        let plan_value: Value = plan_json
            .try_into()
            .unwrap_or(Value::String("<invalid plan>".into()));

        ctx.memory
            .write(
                crate::memory::MemorySpace::Stm,
                format!("plan:{}", ctx.execution_id),
                plan_value.clone(),
            )
            .await
            .ok();

        ctx.aam.set_belief(
            format!("goal:{}", ctx.execution_id),
            Value::String(original_goal.to_string()),
            transition_label.clone(),
        );

        ctx.aam.set_belief(
            format!("plan:{}", ctx.execution_id),
            plan_value.clone(),
            transition_label.clone(),
        );

        let plan_goal = AamGoal {
            id: GoalId::new(),
            description: plan_output.result.clone(),
            priority: plan_output
                .plan
                .iter()
                .map(|step| step.priority)
                .max()
                .unwrap_or(50),
            status: GoalStatus::Active,
        };
        ctx.aam.add_goal(plan_goal, transition_label.clone());

        // If inner plan exists and enabled, execute it
        if enable_inner_plan {
            if let Some(inner_plan) = plan_output.inner_plan.clone() {
                tracing::info!(
                    execution_id = %ctx.execution_id,
                    "Linking inner plan DAG"
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

                ctx.memory
                    .write(
                        crate::memory::MemorySpace::Episodic,
                        format!("inner_plan_spliced:{}", ctx.execution_id),
                        Value::String(format!(
                            "Inner plan merged into DAG with {} nodes",
                            inserted_nodes
                        )),
                    )
                    .await
                    .ok();
            }
        }

        // Return plan as structured value
        let mut result_obj = HashMap::new();
        result_obj.insert(
            "plan".to_string(),
            Value::Array(
                plan_output
                    .plan
                    .iter()
                    .map(|step| {
                        let mut step_obj = HashMap::new();
                        step_obj.insert(
                            "description".to_string(),
                            Value::String(step.description.clone()),
                        );
                        step_obj.insert(
                            "priority".to_string(),
                            Value::Number(apxm_core::types::values::Number::Integer(
                                step.priority as i64,
                            )),
                        );
                        if let Some(deps) = &step.dependencies {
                            step_obj.insert(
                                "dependencies".to_string(),
                                Value::Array(
                                    deps.iter().map(|d| Value::String(d.clone())).collect(),
                                ),
                            );
                        }
                        Value::Object(step_obj)
                    })
                    .collect(),
            ),
        );
        result_obj.insert("result".to_string(), Value::String(plan_output.result));

        Ok(Value::Object(result_obj))
    } else {
        // Fall back to plain text response
        Ok(Value::String(content))
    }
}

/// Parse plan output from LLM response
fn parse_plan_output(content: &str) -> std::result::Result<PlanOutput, serde_json::Error> {
    let trimmed = content.trim();

    // Try direct parse
    if let Ok(output) = serde_json::from_str::<PlanOutput>(trimmed) {
        return Ok(output);
    }

    // Try to extract JSON from markdown
    if let Some(json_str) = extract_json_from_markdown(trimmed) {
        if let Ok(output) = serde_json::from_str::<PlanOutput>(&json_str) {
            return Ok(output);
        }
    }

    Err(serde_json::Error::custom("Failed to parse plan output"))
}

/// Extract JSON from markdown code block
fn extract_json_from_markdown(content: &str) -> Option<String> {
    if let Some(start) = content.find("```json") {
        if let Some(end) = content[start + 7..].find("```") {
            return Some(content[start + 7..start + 7 + end].trim().to_string());
        }
    }

    if let Some(start) = content.find("```") {
        if let Some(end) = content[start + 3..].find("```") {
            let extracted = content[start + 3..start + 3 + end].trim();
            if extracted.starts_with('{') || extracted.starts_with('[') {
                return Some(extracted.to_string());
            }
        }
    }

    None
}

async fn generate_inner_plan(
    ctx: &ExecutionContext,
    _node: &Node,
    plan_output: &PlanOutput,
    goal: &str,
    model_override: Option<&str>,
) -> Result<Option<String>> {
    let prompt_context = serde_json::json!({
        "goal": goal,
        "plan": plan_output.plan,
        "result": plan_output.result,
    });

    let user_prompt = apxm_prompts::render_prompt("inner_plan", &prompt_context).map_err(|e| {
        RuntimeError::LLM {
            message: format!("Failed to render inner_plan prompt: {e}"),
            backend: None,
        }
    })?;

    let mut request = LLMRequest::new(user_prompt);
    if let Some(model_name) = model_override {
        request = request.with_model(model_name.to_string());
    }

    let response = ctx
        .llm_registry
        .generate(request.clone())
        .await
        .map_err(|e| llm_error(ctx, "INNER_PLAN", &request, e))?;

    let content = response.content;
    tracing::info!(
        execution_id = %ctx.execution_id,
        raw_response = %content,
        "INNER PLAN model response"
    );

    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    Ok(Some(trimmed.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plan_output() {
        let json = r#"{
            "plan": [
                {"description": "Step 1", "priority": 90},
                {"description": "Step 2", "priority": 80, "dependencies": ["Step 1"]}
            ],
            "result": "Plan created successfully"
        }"#;

        let output = parse_plan_output(json).unwrap();
        assert_eq!(output.plan.len(), 2);
        assert_eq!(output.plan[0].description, "Step 1");
        assert_eq!(output.plan[1].priority, 80);
        assert_eq!(output.result, "Plan created successfully");
    }

    #[test]
    fn test_parse_markdown_wrapped_plan() {
        let content = r#"Here is the plan:

```json
{
    "plan": [
        {"description": "Analyze", "priority": 100}
    ],
    "result": "Analysis plan"
}
```

Done."#;

        let output = parse_plan_output(content).unwrap();
        assert_eq!(output.plan.len(), 1);
        assert_eq!(output.plan[0].description, "Analyze");
    }
}
