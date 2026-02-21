//! Capability interception hooks.

use apxm_core::types::values::Value;
use async_trait::async_trait;
use std::collections::HashMap;

/// Decision returned from `pre_invoke`.
#[derive(Debug, Clone)]
pub enum InterceptDecision {
    /// Continue execution unchanged.
    Allow,
    /// Deny invocation with a reason.
    Deny { reason: String },
    /// Continue execution with updated arguments.
    EditArgs { args: HashMap<String, Value> },
}

/// Optional interception hook around capability execution.
#[async_trait]
pub trait CapabilityInterceptor: Send + Sync {
    /// Stable interceptor name for deduplication.
    fn name(&self) -> &str {
        "interceptor"
    }

    /// Called before capability execution.
    async fn pre_invoke(&self, _name: &str, _args: &HashMap<String, Value>) -> InterceptDecision {
        InterceptDecision::Allow
    }

    /// Called after capability execution (success path).
    async fn post_invoke(&self, _name: &str, _result: &Value) {}
}
