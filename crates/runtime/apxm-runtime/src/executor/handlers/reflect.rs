//! REFLECT operation - Meta-reasoning on execution history with LLM
//!
//! Supports:
//! - Reflection on episodic memory
//! - LLM-based analysis of execution patterns
//! - Insights extraction from history
//! - Performance analysis
//! - Retry logic with exponential backoff

use super::{
    ExecutionContext, Node, Result, Value, execute_llm_request, get_optional_string_attribute,
};
use apxm_backends::LLMRequest;
use apxm_core::error::RuntimeError;
use serde::de::Error;
use serde::{Deserialize, Serialize};

/// Structured reflection output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionOutput {
    /// Insights extracted from history
    #[serde(default)]
    pub insights: Vec<String>,

    /// Patterns identified
    #[serde(default)]
    pub patterns: Vec<String>,

    /// Recommendations for improvement
    #[serde(default)]
    pub recommendations: Vec<String>,

    /// Summary of reflection
    pub summary: String,
}

/// Execute REFLECT operation - LLM-based reflection on execution history
///
/// The REFLECT operation analyzes episodic memory to:
/// - Identify patterns in execution
/// - Extract insights from past operations
/// - Generate recommendations for improvement
/// - Summarize performance and outcomes
///
/// # Structured Output Format
///
/// ```json
/// {
///   "insights": ["Pattern A detected", "Bottleneck at step X"],
///   "patterns": ["Frequent error type Y"],
///   "recommendations": ["Optimize operation Z"],
///   "summary": "Overall execution analysis"
/// }
/// ```
pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Get prompt from attribute or use default
    let prompt = get_optional_string_attribute(node, "prompt")?
        .or_else(|| {
            get_optional_string_attribute(node, "trace_id")
                .ok()
                .flatten()
        })
        .unwrap_or_default();

    let model = get_optional_string_attribute(node, "model")?;
    let limit = node
        .attributes
        .get("history_limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    let max_retries = node
        .attributes
        .get("max_retries")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as u32;

    // Build context from inputs (the content to reflect on)
    let input_context = if !inputs.is_empty() {
        inputs
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n\n")
    } else {
        String::new()
    };

    // Get recent episodic history
    let episodes = ctx.memory.query_episodes(&ctx.execution_id).await?;
    let history_slice = episodes.iter().rev().take(limit).collect::<Vec<_>>();

    // Format history for LLM
    let history_text = history_slice
        .iter()
        .map(|e| format!("- {}: {}", e.event_type, e.payload))
        .collect::<Vec<_>>()
        .join("\n");

    // Build reflection prompt - use input context as primary content
    let effective_prompt = if !input_context.is_empty() {
        format!(
            "Analyze and reflect on the following content, identifying strengths, weaknesses, and areas for improvement:\n\n{}\n\n{}",
            input_context,
            if prompt.is_empty() { "" } else { &prompt }
        )
    } else if !prompt.is_empty() {
        prompt.clone()
    } else {
        "Reflect on recent execution history and provide insights.".to_string()
    };

    // Build reflection prompt using apxm-prompts
    let prompt_context = serde_json::json!({
        "prompt": effective_prompt,
        "history": history_text,
        "episode_count": history_slice.len(),
    });

    let history_section = if history_text.is_empty() {
        String::new()
    } else {
        format!("Execution history:\n{}", history_text)
    };

    let full_prompt = apxm_backends::render_prompt("reflect_system", &prompt_context)
        .unwrap_or_else(|_| {
            format!(
                "{}\n\n{}\n\n\
                 Respond in JSON format with: insights (array), patterns (array), \
                 recommendations (array), and summary (string).",
                effective_prompt, history_section
            )
        });

    // Build LLM request
    let mut request = LLMRequest::new(full_prompt);

    if let Some(m) = model {
        request = request.with_model(m);
    }

    // Load system prompt from config, template, or fallback
    // Priority: 1) config instruction, 2) template, 3) hardcoded fallback
    let system_prompt = ctx
        .instruction_config
        .reflect
        .clone()
        .or_else(|| apxm_backends::render_prompt("reflect_system", &serde_json::json!({})).ok())
        .unwrap_or_else(|| {
            "You are an expert at analyzing execution patterns and extracting insights. \
             Always respond in valid JSON format with structured analysis."
                .to_string()
        });
    request = request.with_system_prompt(system_prompt);

    // Execute with retries
    let mut last_error = None;
    for attempt in 0..=max_retries {
        if attempt > 0 {
            tracing::warn!(
                execution_id = %ctx.execution_id,
                attempt = attempt,
                "Retrying REFLECT operation"
            );

            let backoff_ms = 100 * 2_u64.pow(attempt - 1);
            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
        }

        match execute_reflect_once(ctx, &request).await {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(e);
                tracing::debug!(
                    execution_id = %ctx.execution_id,
                    attempt = attempt,
                    error = %last_error.as_ref().unwrap(),
                    "REFLECT attempt failed"
                );
            }
        }
    }

    Err(last_error.unwrap_or_else(|| RuntimeError::LLM {
        message: "All retry attempts exhausted for REFLECT".to_string(),
        backend: None,
    }))
}

/// Execute a single REFLECT attempt
async fn execute_reflect_once(ctx: &ExecutionContext, request: &LLMRequest) -> Result<Value> {
    // Execute LLM request
    let response = execute_llm_request(ctx, "REFLECT", request).await?;

    let content = response.content;

    // Try to parse as structured reflection
    if let Ok(reflection) = parse_reflection_output(&content) {
        // Store insights in episodic memory
        if !reflection.insights.is_empty() {
            for insight in &reflection.insights {
                ctx.memory
                    .write(
                        crate::memory::MemorySpace::Episodic,
                        format!("insight:{}:{}", ctx.execution_id, uuid::Uuid::now_v7()),
                        Value::String(insight.clone()),
                    )
                    .await
                    .ok();
            }
        }

        // Return structured reflection as value
        let reflection_value = serde_json::to_value(&reflection).map_err(|e| {
            RuntimeError::Serialization(format!("Failed to serialize reflection: {}", e))
        })?;

        Ok(reflection_value
            .try_into()
            .unwrap_or(Value::String(reflection.summary)))
    } else {
        // Fall back to plain text response
        Ok(Value::String(content))
    }
}

/// Parse reflection output from LLM response
fn parse_reflection_output(
    content: &str,
) -> std::result::Result<ReflectionOutput, serde_json::Error> {
    let trimmed = content.trim();

    // Try direct parse
    if let Ok(output) = serde_json::from_str::<ReflectionOutput>(trimmed) {
        return Ok(output);
    }

    // Try to extract JSON from markdown
    if let Some(json_str) = extract_json_from_markdown(trimmed)
        && let Ok(output) = serde_json::from_str::<ReflectionOutput>(&json_str)
    {
        return Ok(output);
    }

    Err(serde_json::Error::custom(
        "Failed to parse reflection output",
    ))
}

/// Extract JSON from markdown code block
fn extract_json_from_markdown(content: &str) -> Option<String> {
    if let Some(start) = content.find("```json")
        && let Some(end) = content[start + 7..].find("```")
    {
        return Some(content[start + 7..start + 7 + end].trim().to_string());
    }

    if let Some(start) = content.find("```")
        && let Some(end) = content[start + 3..].find("```")
    {
        let extracted = content[start + 3..start + 3 + end].trim();
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
    fn test_parse_reflection_output() {
        let json = r#"{
            "insights": ["Pattern detected", "Bottleneck found"],
            "patterns": ["Error type X"],
            "recommendations": ["Optimize Y"],
            "summary": "Good performance overall"
        }"#;

        let output = parse_reflection_output(json).unwrap();
        assert_eq!(output.insights.len(), 2);
        assert_eq!(output.patterns.len(), 1);
        assert_eq!(output.recommendations.len(), 1);
        assert_eq!(output.summary, "Good performance overall");
    }

    #[test]
    fn test_parse_markdown_wrapped_reflection() {
        let content = r#"Here is the reflection:

```json
{
    "insights": ["Good"],
    "patterns": [],
    "recommendations": [],
    "summary": "Test"
}
```

Done."#;

        let output = parse_reflection_output(content).unwrap();
        assert_eq!(output.insights.len(), 1);
        assert_eq!(output.summary, "Test");
    }
}
