//! Runtime configuration types shared across APXM components.

use serde::{Deserialize, Serialize};

/// System prompts for LLM operations (ask, think, reason, plan, reflect).
///
/// These prompts are shared between APXM runtime and LangGraph benchmarks
/// to ensure consistent behavior across both implementations.
///
/// Configuration is typically loaded from `~/.apxm/config.toml` under `[instruction]`:
///
/// ```toml
/// [instruction]
/// ask = "You are a helpful AI assistant."
/// think = "Think step by step."
/// reason = "Provide structured reasoning."
/// plan = "Create actionable plans."
/// reflect = "Analyze execution patterns."
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InstructionConfig {
    /// System prompt for ASK operations (simple Q&A).
    pub ask: Option<String>,

    /// System prompt for THINK operations (extended reasoning).
    pub think: Option<String>,

    /// System prompt for REASON operations (structured reasoning with belief/goal updates).
    pub reason: Option<String>,

    /// System prompt for PLAN operations (task decomposition and planning).
    pub plan: Option<String>,

    /// System prompt for REFLECT operations (execution analysis and insights).
    pub reflect: Option<String>,
}

impl InstructionConfig {
    /// Create a new empty instruction config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get system prompt for the specified operation, returning None if not configured.
    pub fn get(&self, operation: &str) -> Option<&str> {
        match operation.to_lowercase().as_str() {
            "ask" => self.ask.as_deref(),
            "think" => self.think.as_deref(),
            "reason" => self.reason.as_deref(),
            "plan" => self.plan.as_deref(),
            "reflect" => self.reflect.as_deref(),
            _ => None,
        }
    }
}

// Tests removed: test_instruction_config_get was a trivial delegation test.
