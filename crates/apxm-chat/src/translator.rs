//! Natural language to DSL translation pipeline

use crate::error::{ChatError, ChatResult};
use crate::storage::Message;
use apxm_core::Plan;
use apxm_linker::Linker;
use apxm_models::{
    backends::{LLMRequest, LLMResponse},
    registry::LLMRegistry,
    schema::OutputParser,
};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Extract DSL code from a model response which may include markdown fences.
///
/// Strategy:
/// - If the response contains a fenced code block starting with "```", return the content between the first pair.
/// - Otherwise trim leading/trailing backticks and whitespace and return the trimmed text.
fn extract_dsl_from_response(resp: &str) -> String {
    let lines: Vec<&str> = resp.lines().collect();
    let mut start: Option<usize> = None;
    let mut end: Option<usize> = None;

    for (idx, ln) in lines.iter().enumerate() {
        if ln.trim_start().starts_with("```") {
            if start.is_none() {
                start = Some(idx);
            } else {
                end = Some(idx);
                break;
            }
        }
    }

    if let (Some(s), Some(e)) = (start, end) {
        return lines[s + 1..e].join("\n").trim().to_string();
    }

    // If a starting fence exists but no closing fence, return content after the first fence
    if let Some(s) = start {
        if s + 1 < lines.len() {
            return lines[s + 1..].join("\n").trim().to_string();
        }
    }

    // No fenced block found - remove surrounding backticks and trim whitespace
    let s = resp.trim();
    let stripped = s.trim_matches('`').trim();
    stripped.to_string()
}

/// Validate DSL using the real compiler if available, otherwise use heuristics.
///
/// This function provides two levels of validation:
/// 1. **Real compiler validation** (when linker is available): 100% accurate
/// 2. **Heuristic validation** (fallback): Conservative checks for common errors
///
/// The real compiler is always preferred when available.
/// Conservative heuristic-based DSL validator (fallback).
///
/// This catches common LLM generation mistakes:
/// - Missing semicolons
/// - Malformed `if` conditions
/// - Extremely large outputs
/// - Multiple agent declarations
///
/// Note: This is intentionally conservative and not a full parser.
fn validate_dsl_heuristic(dsl: &str) -> Result<(), String> {
    // Basic checks
    let trimmed = dsl.trim();
    if trimmed.is_empty() {
        return Err("Generated DSL is empty".to_string());
    }

    // Protect against runaway generation
    const MAX_LINES: usize = 100_000;
    let lines: Vec<&str> = dsl.lines().collect();
    if lines.len() > MAX_LINES {
        return Err(format!(
            "Generated DSL is too long ({} lines). The model may have produced invalid output.",
            lines.len()
        ));
    }

    // Heuristic diagnostics
    let mut diagnostics: Vec<String> = Vec::new();
    let mut stmt_like_count: usize = 0;
    let mut agent_mentions: usize = 0;

    for (idx, raw) in lines.iter().enumerate() {
        let ln_no = idx + 1;
        let s = raw.trim();

        // Skip empty lines and full-line comments
        if s.is_empty() || s.starts_with("//") || s.starts_with("/*") {
            continue;
        }

        // Count simple statement-like lines (approximate)
        if s.ends_with(';') {
            stmt_like_count += 1;
        }

        // Detect missing semicolons for typical statement lines:
        // If a line doesn't end with ';', '{', '}', or ':' (labels), and is not a
        // control construct (starts with if/else/while/for/func), flag it.
        let first_token = s.split_whitespace().next().unwrap_or("");
        let is_control = matches!(
            first_token,
            "if" | "else" | "while" | "for" | "func" | "switch" | "match" | "return"
        );
        let ends_ok = s.ends_with(';') || s.ends_with('{') || s.ends_with('}') || s.ends_with(':');

        if !is_control && !ends_ok {
            // Lines that look like assignments or calls but lack a trailing semicolon
            // are the most common DSL generation error from LLMs.
            // Be conservative: require at least one punctuation char at end for safety.
            diagnostics.push(format!("Line {}: possible missing semicolon", ln_no));
        }

        // 'if' should include parentheses for the condition in this DSL style.
        if s.starts_with("if") {
            if !s.contains('(') || !s.contains(')') {
                diagnostics.push(format!(
                    "Line {}: 'if' without parentheses around condition",
                    ln_no
                ));
            }
        }

        // Track 'agent' mentions (some DSLs expect a single agent declaration)
        if s.contains("agent ") || s.contains("agent:") {
            agent_mentions += 1;
        }
    }

    // Report too many statements as a sign of runaway generation
    if stmt_like_count > 50_000 {
        return Err("Generated DSL contains an excessive number of statements; the model output looks runaway.".to_string());
    }

    // If multiple agent declarations are found, warn (likely model error)
    if agent_mentions > 1 {
        diagnostics.push(format!("Multiple 'agent' declarations detected ({} occurrences). Consider using a single agent declaration.", agent_mentions));
    }

    if diagnostics.is_empty() {
        Ok(())
    } else {
        // Combine diagnostics into a compact message suitable for LLM feedback.
        let summary = diagnostics
            .into_iter()
            .take(10)
            .collect::<Vec<_>>()
            .join("; ");
        let suffix = if dsl.len() > 1024 {
            " (DSL truncated for brevity)"
        } else {
            ""
        };
        Err(format!("Validator found issues: {}{}", summary, suffix))
    }
}

/// Result of translation process
#[derive(Debug)]
pub struct TranslationResult {
    /// The unified plan
    pub plan: Plan,

    /// The generated DSL code
    pub dsl_code: String,
}

/// Translator for natural language to DSL
pub struct Translator {
    registry: Arc<LLMRegistry>,
    planning_model: String,
    dsl_model: String,
    /// List of capability names that are available in the runtime for this session.
    /// This is passed into generation prompts so models don't invent unsupported capabilities.
    available_capabilities: Vec<String>,
    /// Linker for DSL compilation/validation
    linker: Option<Arc<Linker>>,
}

impl Translator {
    /// Create a new translator
    ///
    /// # Parameters
    /// - `registry`: LLM registry for model access
    /// - `planning_model`: Model to use for plan generation
    /// - `dsl_model`: Model to use for DSL generation (defaults to planning_model)
    /// - `available_capabilities`: List of runtime capabilities for validation
    /// - `linker`: Optional linker for real compiler validation
    pub fn new(
        registry: Arc<LLMRegistry>,
        planning_model: String,
        dsl_model: Option<String>,
        available_capabilities: Option<Vec<String>>,
        linker: Option<Arc<Linker>>,
    ) -> Self {
        let dsl_model = dsl_model.unwrap_or_else(|| planning_model.clone());
        let available_capabilities = available_capabilities.unwrap_or_default();

        Self {
            registry,
            planning_model,
            dsl_model,
            available_capabilities,
            linker,
        }
    }

    /// Translate natural language to DSL via plan
    pub async fn translate(
        &self,
        goal: &str,
        context: &[Message],
    ) -> ChatResult<TranslationResult> {
        // Step 1: Generate plan
        let plan = self.generate_plan(goal, context).await?;

        // Step 2: Generate DSL from plan
        let dsl_code = self.generate_dsl(goal, &plan).await?;

        Ok(TranslationResult {
            plan,
            dsl_code,
        })
    }

    /// Validate DSL using the real compiler if available, otherwise use heuristics.
    async fn validate_dsl_with_compiler(&self, dsl: &str) -> Result<(), String> {
        if let Some(linker) = &self.linker {
            return self.validate_with_real_compiler(linker, dsl).await;
        }

        validate_dsl_heuristic(dsl)
    }

    /// Validate DSL using the real compiler.
    async fn validate_with_real_compiler(
        &self,
        linker: &Linker,
        dsl: &str,
    ) -> Result<(), String> {
        let tmp_dir = std::env::temp_dir();
        let temp_file = tmp_dir.join(format!("validate_{}.ais", uuid::Uuid::now_v7()));

        std::fs::write(&temp_file, dsl)
            .map_err(|e| format!("Failed to write DSL to temp file: {}", e))?;

        tracing::debug!(
            temp_file = %temp_file.display(),
            "Validating DSL with real compiler"
        );
        let result = linker.compile_only(&temp_file, false);

        let _ = std::fs::remove_file(&temp_file);

        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(err.to_string()),
        }
    }

    /// Generate plan from natural language goal
    async fn generate_plan(&self, goal: &str, context: &[Message]) -> ChatResult<Plan> {
        // Format conversation context
        let context_str = context
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Build prompt data for plan_system template
        let mut user_request = String::from("Goal:\n");
        user_request.push_str(goal);
        if !context_str.is_empty() {
            user_request.push_str("\n\nConversation context:\n");
            user_request.push_str(&context_str);
        }

        let prompt =
            apxm_prompts::render_prompt("plan_system", &json!({ "text": user_request }))
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

        let plan: Plan = serde_json::from_value(plan_value)
            .map_err(|e| ChatError::Translation(format!("Failed to decode plan JSON: {}", e)))?;

        Ok(plan)
    }

    /// Generate DSL code from plan
    async fn generate_dsl(&self, goal: &str, plan: &Plan) -> ChatResult<String> {
        self.generate_dsl_with_feedback(goal, plan, None).await
    }

    async fn generate_dsl_with_feedback(
        &self,
        goal: &str,
        plan: &Plan,
        feedback: Option<&str>,
    ) -> ChatResult<String> {
        // Build the goal text, optionally including compiler feedback
        let goal_with_feedback = if let Some(notes) = feedback {
            format!(
                "{}\n\nCompiler feedback to address:\n{}\nEnsure the regenerated DSL fixes these issues while staying minimal.",
                goal, notes
            )
        } else {
            goal.to_string()
        };

        // Prepare template variables for inner_plan template
        // Include a concise list of available capabilities so the model uses only allowed names.
        // Serialize complex fields into Values to avoid compile-time macro errors with `json!`.
        let plan_value = serde_json::to_value(&plan.steps).map_err(|e| {
            ChatError::Translation(format!(
                "Failed to serialize plan for prompt payload: {}",
                e
            ))
        })?;
        let caps_value = serde_json::to_value(&self.available_capabilities).map_err(|e| {
            ChatError::Translation(format!(
                "Failed to serialize available_capabilities for prompt payload: {}",
                e
            ))
        })?;
        let mut payload_map = serde_json::Map::new();
        payload_map.insert(
            "goal".to_string(),
            serde_json::Value::String(goal_with_feedback),
        );
        payload_map.insert(
            "result".to_string(),
            serde_json::Value::String(plan.result.clone()),
        );
        payload_map.insert("plan".to_string(), plan_value);
        payload_map.insert("available_capabilities".to_string(), caps_value);
        let prompt_payload = serde_json::Value::Object(payload_map);

        let prompt = apxm_prompts::render_prompt("inner_plan", &prompt_payload)
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

        // Extract DSL from response (strip fences if present).
        let mut dsl_code = extract_dsl_from_response(&response.content);

        // --- Capability check (single-shot repair) ---
        // Quickly detect capability-like tokens (foo.bar style) and ensure they are allowed.
        // This is conservative: tokens that contain at least one '.' are considered capability-like.
        let allowed_caps = &self.available_capabilities;
        if !allowed_caps.is_empty() {
            let mut unknown_caps: Vec<String> = Vec::new();
            for token in dsl_code
                .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
                .filter(|s| !s.is_empty())
            {
                if token.contains('.') {
                    // Filter trivial tokens like "." or leading/trailing dots
                    let t = token.trim_matches('.');
                    if t.is_empty() {
                        continue;
                    }
                    // If not in allowed list, record it
                    if !allowed_caps.iter().any(|a| a == t)
                        && !unknown_caps.contains(&t.to_string())
                    {
                        unknown_caps.push(t.to_string());
                    }
                }
            }

            if !unknown_caps.is_empty() {
                // Build a concise repair instruction and ask the DSL model to regenerate once.
                let repair_feedback = format!(
                    "The generated DSL references unavailable capability(ies): {}. Allowed capabilities: {}. \
Please regenerate the DSL using ONLY allowed capability names and output only raw DSL (no surrounding explanation or fences).",
                    unknown_caps.join(", "),
                    allowed_caps.join(", ")
                );

                let goal_with_repair = format!(
                    "{}\n\nCompiler feedback to address:\n{}\nEnsure the regenerated DSL fixes these issues while staying minimal.",
                    goal, repair_feedback
                );

                let plan_value = serde_json::to_value(&plan.steps).map_err(|e| {
                    ChatError::Translation(format!(
                        "Failed to serialize plan for prompt payload: {}",
                        e
                    ))
                })?;
                let caps_value =
                    serde_json::to_value(&self.available_capabilities).map_err(|e| {
                        ChatError::Translation(format!(
                            "Failed to serialize available_capabilities for prompt payload: {}",
                            e
                        ))
                    })?;
                let mut payload_map = serde_json::Map::new();
                payload_map.insert(
                    "goal".to_string(),
                    serde_json::Value::String(goal_with_repair),
                );
                payload_map.insert(
                    "result".to_string(),
                    serde_json::Value::String(plan.result.clone()),
                );
                payload_map.insert("plan".to_string(), plan_value);
                payload_map.insert("available_capabilities".to_string(), caps_value);
                let prompt_payload = serde_json::Value::Object(payload_map);

                let prompt =
                    apxm_prompts::render_prompt("inner_plan", &prompt_payload).map_err(|e| {
                        ChatError::Translation(format!("Failed to render prompt: {}", e))
                    })?;

                let request = LLMRequest::new(prompt)
                    .with_max_tokens(4096)
                    .with_temperature(0.3);

                let repaired_response: LLMResponse = self
                    .registry
                    .generate_with_backend(&self.dsl_model, request)
                    .await
                    .map_err(|err| {
                        tracing::error!(
                            target: "chat::translator",
                            stage = "dsl_regeneration_capability_fix",
                            model = %self.dsl_model,
                            error = %err,
                            "DSL regeneration model request failed"
                        );
                        ChatError::model_failure(
                            format!(
                                "Unable to synthesize DSL with model '{}'. Enable tracing logs for backend details.",
                                self.dsl_model.as_str()
                            ),
                            err,
                        )
                    })?;

                dsl_code = extract_dsl_from_response(&repaired_response.content);

                // If the regenerated DSL still contains unknown capabilities, fail fast with
                // a clear message so the caller can decide how to proceed.
                let mut still_unknown: Vec<String> = Vec::new();
                for token in dsl_code
                    .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
                    .filter(|s| !s.is_empty())
                {
                    if token.contains('.') {
                        let t = token.trim_matches('.');
                        if t.is_empty() {
                            continue;
                        }
                        if !allowed_caps.iter().any(|a| a == t)
                            && !still_unknown.contains(&t.to_string())
                        {
                            still_unknown.push(t.to_string());
                        }
                    }
                }
                if !still_unknown.is_empty() {
                    return Err(ChatError::Translation(format!(
                        "Generated DSL references unknown capabilities after one repair attempt: {}. Allowed: {}",
                        still_unknown.join(", "),
                        allowed_caps.join(", ")
                    )));
                }
            }
        }
        // --- end capability check ---

        // Validate using the real compiler (if available) or heuristics (fallback).
        // If parsing fails we allow one regeneration attempt with focused feedback.
        let mut attempt: usize = 0;
        loop {
            match self.validate_dsl_with_compiler(&dsl_code).await {
                Ok(()) => break,
                Err(problem) => {
                    // If the caller already provided feedback, return the compiler error immediately.
                    if feedback.is_some() {
                        return Err(ChatError::Translation(format!(
                            "Generated DSL failed compiler validation: {}",
                            problem
                        )));
                    }

                    // Only allow one regeneration attempt to avoid infinite loops.
                    if attempt >= 1 {
                        return Err(ChatError::Translation(format!(
                            "Generated DSL failed compiler validation after {} attempts: {}",
                            attempt + 1,
                            problem
                        )));
                    }

                    // Build repair feedback and ask the DSL model to regenerate once.
                    let repair_feedback = format!(
                        "The generated DSL failed compiler validation: {}. Please output only valid APXM DSL without any markdown fences or surrounding explanation.",
                        problem
                    );

                    let goal_with_repair = format!(
                        "{}\n\nCompiler feedback to address:\n{}\nEnsure the regenerated DSL fixes these issues while staying minimal.",
                        goal, repair_feedback
                    );

                    let plan_value = serde_json::to_value(&plan.steps).map_err(|e| {
                        ChatError::Translation(format!(
                            "Failed to serialize plan for prompt payload: {}",
                            e
                        ))
                    })?;
                    let caps_value =
                        serde_json::to_value(&self.available_capabilities).map_err(|e| {
                            ChatError::Translation(format!(
                                "Failed to serialize available_capabilities for prompt payload: {}",
                                e
                            ))
                        })?;
                    let mut payload_map = serde_json::Map::new();
                    payload_map.insert(
                        "goal".to_string(),
                        serde_json::Value::String(goal_with_repair),
                    );
                    payload_map.insert(
                        "result".to_string(),
                        serde_json::Value::String(plan.result.clone()),
                    );
                    payload_map.insert("plan".to_string(), plan_value);
                    payload_map.insert("available_capabilities".to_string(), caps_value);
                    let prompt_payload = serde_json::Value::Object(payload_map);

                    let prompt = apxm_prompts::render_prompt("inner_plan", &prompt_payload)
                        .map_err(|e| {
                            ChatError::Translation(format!("Failed to render prompt: {}", e))
                        })?;

                    let request = LLMRequest::new(prompt)
                        .with_max_tokens(4096)
                        .with_temperature(0.3);

                    let repaired_response: LLMResponse = self
                        .registry
                        .generate_with_backend(&self.dsl_model, request)
                        .await
                        .map_err(|err| {
                            tracing::error!(
                                target: "chat::translator",
                                stage = "dsl_regeneration",
                                model = %self.dsl_model,
                                error = %err,
                                "DSL regeneration model request failed"
                            );
                            ChatError::model_failure(
                                format!(
                                    "Unable to synthesize DSL with model '{}'. Enable tracing logs for backend details.",
                                    self.dsl_model.as_str()
                                ),
                                err,
                            )
                        })?;

                    dsl_code = extract_dsl_from_response(&repaired_response.content);

                    attempt += 1;
                    continue;
                }
            }
        }

        // Final trim and safety checks
        let dsl_code = dsl_code.trim().to_string();

        // Validate DSL code length to prevent memory exhaustion
        const MAX_DSL_LENGTH: usize = 1_000_000; // 1MB limit
        if dsl_code.len() > MAX_DSL_LENGTH {
            return Err(ChatError::Translation(format!(
                "Generated DSL exceeds maximum allowed size ({} bytes > {} bytes). \
                 The model may have generated invalid output.",
                dsl_code.len(),
                MAX_DSL_LENGTH
            )));
        }

        // Validate that DSL is not empty (redundant safety check)
        if dsl_code.is_empty() {
            return Err(ChatError::Translation(
                "DSL generation produced empty output. Please try again.".to_string(),
            ));
        }

        Ok(dsl_code)
    }

    pub async fn regenerate_dsl_with_feedback(
        &self,
        goal: &str,
        plan: &Plan,
        feedback: &str,
    ) -> ChatResult<String> {
        self.generate_dsl_with_feedback(goal, plan, Some(feedback))
            .await
    }

    /// Synthesize a concise natural-language summary from a Plan.
    ///
    /// This helper produces a short human-friendly summary that can be used
    /// as a fallback message when no execution results are available. The
    /// method is intentionally deterministic and dependency-free so it can be
    /// used synchronously from other components.
    pub fn synthesize_nl_from_plan(&self, plan: &Plan) -> String {
        // Start with the high-level result/goal text.
        let mut out = String::new();
        let result = plan.result.trim();
        if result.is_empty() {
            out.push_str("Plan summary:");
        } else {
            out.push_str(&format!("Plan: {}", result));
        }

        // If there are steps, render them compactly as numbered items.
        if !plan.steps.is_empty() {
            out.push_str(" Steps:");
            for (i, step) in plan.steps.iter().enumerate() {
                let desc = step.description.trim();
                if desc.is_empty() {
                    continue;
                }
                // Use semicolon-separated short items to keep the summary concise.
                if i > 0 {
                    out.push_str(" ");
                }
                out.push_str(&format!("{}. {}", i + 1, desc));
                if i + 1 < plan.steps.len() {
                    out.push(';');
                }
            }
        }

        out
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
                .send(format!("Plan: {}", translation.plan.result))
                .await;
            let _ = tx.send(format!("\nDSL:\n{}", translation.dsl_code)).await;
        });

        Ok(rx)
    }
}
