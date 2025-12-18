//! Natural language to DSL translation pipeline

use crate::error::{ChatError, ChatResult};
use crate::storage::Message;
use apxm_models::{
    backends::{LLMRequest, LLMResponse},
    registry::LLMRegistry,
    schema::OutputParser,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::Write as _;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Outer plan generated from natural language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuterPlan {
    /// Plan steps
    pub plan: Vec<PlanStep>,

    /// Summary of what will be accomplished
    pub result: String,
}

/// A single step in the plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step description
    pub description: String,

    /// Priority level (0-100, higher = more important)
    pub priority: u32,

    /// Dependencies on other steps
    #[serde(default)]
    pub dependencies: Vec<String>,
}

/// Result of translation process
#[derive(Debug)]
pub struct TranslationResult {
    /// The outer plan
    pub outer_plan: OuterPlan,

    /// The generated DSL code
    pub dsl_code: String,
}

/// Translator for natural language to DSL
pub struct Translator {
    registry: Arc<LLMRegistry>,
    planning_model: String,
    dsl_model: String,
}

impl Translator {
    /// Create a new translator
    pub fn new(
        registry: Arc<LLMRegistry>,
        planning_model: String,
        dsl_model: Option<String>,
    ) -> Self {
        let dsl_model = dsl_model.unwrap_or_else(|| planning_model.clone());

        Self {
            registry,
            planning_model,
            dsl_model,
        }
    }

    /// Translate natural language to DSL via outer plan
    pub async fn translate(
        &self,
        goal: &str,
        context: &[Message],
    ) -> ChatResult<TranslationResult> {
        // Step 1: Generate outer plan
        let outer_plan = self.generate_outer_plan(goal, context).await?;

        // Step 2: Generate DSL from outer plan
        let dsl_code = self.generate_dsl(goal, &outer_plan).await?;

        Ok(TranslationResult {
            outer_plan,
            dsl_code,
        })
    }

    /// Generate outer plan from natural language goal
    async fn generate_outer_plan(&self, goal: &str, context: &[Message]) -> ChatResult<OuterPlan> {
        // Format conversation context
        let context_str = context
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let mut planner_prompt = String::from(
            "You are the APXM planning specialist. Generate a JSON object with fields `plan` (array of steps) and `result` (string summary).\n\nEach plan step must have: description (string), priority (0-100), and optional dependencies (array of strings). Return strictly valid JSON with double quotes.\n",
        );
        planner_prompt.push_str(
            r#"Example:
{"plan":[{"description":"Step 1","priority":90}]}

"#,
        );
        planner_prompt.push_str("Goal:\n");
        planner_prompt.push_str(goal);
        if !context_str.is_empty() {
            planner_prompt.push_str("\n\nConversation context:\n");
            planner_prompt.push_str(&context_str);
        }

        let prompt = apxm_prompts::render_prompt("chat_system", &json!({ "text": planner_prompt }))
            .map_err(|e| ChatError::Translation(format!("Failed to render prompt: {}", e)))?;

        // Build LLM request
        let request = LLMRequest::new(prompt)
            .with_max_tokens(2048)
            .with_temperature(0.7);

        // Call LLM
        let response: LLMResponse = self
            .registry
            .generate_with_backend(&self.planning_model, request)
            .await
            .map_err(|err| {
                tracing::error!(
                    target: "chat::translator",
                    stage = "outer_plan",
                    model = %self.planning_model,
                    error = %err,
                    "Planning model request failed"
                );
                ChatError::model_failure(
                    format!(
                        "Unable to generate plan with model '{}'. Enable tracing logs for backend details.",
                        self.planning_model.as_str()
                    ),
                    err,
                )
            })?;

        // Parse JSON response via shared output parser
        let plan_value = OutputParser::parse_json(&response.content)
            .map_err(|e| ChatError::Translation(format!("Failed to parse plan JSON: {}", e)))?;

        let outer_plan: OuterPlan = serde_json::from_value(plan_value)
            .map_err(|e| ChatError::Translation(format!("Failed to decode plan JSON: {}", e)))?;

        Ok(outer_plan)
    }

    /// Generate DSL code from outer plan
    async fn generate_dsl(&self, goal: &str, plan: &OuterPlan) -> ChatResult<String> {
        self.generate_dsl_with_feedback(goal, plan, None).await
    }

    async fn generate_dsl_with_feedback(
        &self,
        goal: &str,
        plan: &OuterPlan,
        feedback: Option<&str>,
    ) -> ChatResult<String> {
        let mut plan_text = String::new();
        let _ = writeln!(&mut plan_text, "Goal: {}", goal);
        let _ = writeln!(&mut plan_text, "Plan summary: {}", plan.result);
        if !plan.plan.is_empty() {
            let _ = writeln!(&mut plan_text, "\nPlan steps:");
            for (idx, step) in plan.plan.iter().enumerate() {
                let _ = writeln!(
                    &mut plan_text,
                    "{}. {} (priority {})",
                    idx + 1,
                    step.description,
                    step.priority
                );
                if !step.dependencies.is_empty() {
                    let deps = step.dependencies.join(", ");
                    let _ = writeln!(&mut plan_text, "   depends on: {}", deps);
                }
            }
        }

        if let Some(notes) = feedback {
            let _ = writeln!(
                &mut plan_text,
                "\nCompiler feedback to address:\n{}\nEnsure the regenerated DSL fixes these issues while staying minimal.",
                notes
            );
        }

        let prompt_payload = json!({
            "text": format!(
                "{}\n\nPlease emit only APXM DSL code that satisfies the plan while remaining minimal.",
                plan_text.trim()
            )
        });

        let prompt = apxm_prompts::render_prompt("chat_system", &prompt_payload)
            .map_err(|e| ChatError::Translation(format!("Failed to render prompt: {}", e)))?;

        let request = LLMRequest::new(prompt)
            .with_max_tokens(4096)
            .with_temperature(0.3);

        let response: LLMResponse = self
            .registry
            .generate_with_backend(&self.dsl_model, request)
            .await
            .map_err(|err| {
                tracing::error!(
                    target: "chat::translator",
                    stage = "dsl_generation",
                    model = %self.dsl_model,
                    error = %err,
                    "DSL generation model request failed"
                );
                ChatError::model_failure(
                    format!(
                        "Unable to synthesize DSL with model '{}'. Enable tracing logs for backend details.",
                        self.dsl_model.as_str()
                    ),
                    err,
                )
            })?;

        Ok(response.content.trim().to_string())
    }

    pub async fn regenerate_dsl_with_feedback(
        &self,
        goal: &str,
        plan: &OuterPlan,
        feedback: &str,
    ) -> ChatResult<String> {
        self.generate_dsl_with_feedback(goal, plan, Some(feedback))
            .await
    }

    /// Stream response (for future use)
    pub async fn translate_streaming(
        &self,
        goal: &str,
        context: &[Message],
    ) -> ChatResult<mpsc::Receiver<String>> {
        let (tx, rx) = mpsc::channel(100);

        // For now, just send the complete result
        // TODO: Implement actual streaming when LLM backends support it
        let translation = self.translate(goal, context).await?;

        tokio::spawn(async move {
            let _ = tx
                .send(format!("Plan: {}", translation.outer_plan.result))
                .await;
            let _ = tx.send(format!("\nDSL:\n{}", translation.dsl_code)).await;
        });

        Ok(rx)
    }
}
