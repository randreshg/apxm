//! Shared string constants for cross-crate protocol fields.
//!
//! Keep graph contract keys and common node attribute keys centralized here
//! so APXM and AgentMate frontends/backends stay consistent.

pub mod diagnostics {
    /// Compile diagnostics mode for canonical graph input.
    pub const MODE_GRAPH: &str = "graph";
}

pub mod inner_plan {
    /// Payload key for graph JSON in structured inner-plan outputs.
    pub const GRAPH_PAYLOAD: &str = "graph";
    /// Payload key for structured codelet DAG in inner-plan outputs.
    pub const CODELET_DAG: &str = "codelet_dag";
}

pub mod graph {
    pub mod metadata {
        /// Graph metadata flag indicating entry flow.
        pub const IS_ENTRY: &str = "is_entry";
    }

    pub mod attrs {
        pub const AGENT_NAME: &str = "agent_name";
        pub const FLOW_NAME: &str = "flow_name";
        pub const MODEL: &str = "model";
        pub const PROVIDER: &str = "provider";
        pub const API_KEY: &str = "api_key";
        pub const BASE_URL: &str = "base_url";
        pub const TEMPERATURE: &str = "temperature";
        pub const SYSTEM_PROMPT: &str = "system_prompt";
        pub const TOOLS_CONFIG: &str = "tools_config";
        pub const TOKEN_BUDGET: &str = "token_budget";
        pub const OUTPUT_SCHEMA: &str = "output_schema";
        pub const MAX_SCHEMA_RETRIES: &str = "max_schema_retries";
        pub const MAX_ITERATIONS: &str = "max_iterations";
        pub const HANDOFF_TARGETS: &str = "handoff_targets";
        pub const INNER_PLAN_SUPPORTED: &str = "inner_plan_supported";
        pub const ENABLE_INNER_PLAN: &str = "enable_inner_plan";
        pub const BIND_INNER_PLAN_OUTPUTS: &str = "bind_inner_plan_outputs";
        pub const TEMPLATE_STR: &str = "template_str";
        pub const PROMPT: &str = "prompt";
        pub const TEMPLATE: &str = "template";
        pub const QUERY: &str = "query";
        pub const MEMORY_TIER: &str = "memory_tier";
        pub const SPACE: &str = "space";
        pub const CAPABILITY: &str = "capability";
        pub const PARAMS_JSON: &str = "params_json";
        pub const GOAL: &str = "goal";
        pub const TRACE_ID: &str = "trace_id";
        pub const TRACE: &str = "trace";
        pub const TRUE_LABEL: &str = "true_label";
        pub const FALSE_LABEL: &str = "false_label";
        pub const CASE_LABELS: &str = "case_labels";
        pub const LABEL: &str = "label";
        pub const TRY_LABEL: &str = "try_label";
        pub const CATCH_LABEL: &str = "catch_label";
        pub const RECOVERY_TEMPLATE: &str = "recovery_template";
        pub const TOOLS_ENABLED: &str = "tools_enabled";
        pub const TOOLS: &str = "tools";
        pub const MESSAGE: &str = "message";
        pub const RECIPIENT: &str = "recipient";
        pub const TARGET: &str = "target";
        pub const PROTOCOL: &str = "protocol";
        pub const CONDITION: &str = "condition";
        pub const VALUE: &str = "value";
        pub const KEY: &str = "key";
        pub const QUEUE: &str = "queue";
        pub const CHECKPOINT: &str = "checkpoint";
        pub const CHECKPOINT_ID: &str = "checkpoint_id";
        pub const SERVER_URL: &str = "server_url";
        pub const ACTION: &str = "action";
        pub const GOAL_ID: &str = "goal_id";
        pub const PRIORITY: &str = "priority";
        pub const ON_FAIL: &str = "on_fail";
        pub const ERROR_MESSAGE: &str = "error_message";
        pub const STRATEGY: &str = "strategy";
        pub const TIMEOUT_MS: &str = "timeout_ms";
        pub const BUDGET: &str = "budget";
        pub const MAX_RETRIES: &str = "max_retries";
        pub const HISTORY_LIMIT: &str = "history_limit";
        pub const LIMIT: &str = "limit";
        pub const STAGING_ID: &str = "staging_id";
        pub const CONTEXT_KEY: &str = "context_key";
        pub const LEASE_MS: &str = "lease_ms";
        pub const MAX_WAIT_MS: &str = "max_wait_ms";
        pub const NOTIFICATION_URL: &str = "notification_url";
        pub const POLL_INTERVAL_MS: &str = "poll_interval_ms";
        pub const POLL_MAX_ATTEMPTS: &str = "poll_max_attempts";
        pub const CASE_REGIONS: &str = "case_regions";
        pub const DEFAULT_REGION: &str = "default_region";
        pub const BACKEND: &str = "backend";
        pub const CLAIM_TEXT: &str = "claim";
        pub const EVIDENCE: &str = "evidence";
        pub const CODE: &str = "code";
        pub const COUNT: &str = "count";
        pub const DISCRIMINANT: &str = "discriminant";
        pub const HANDOFF: &str = "handoff";
        pub const HANDOFF_FROM: &str = "handoff_from";
        pub const HANDOFF_TO: &str = "handoff_to";
        pub const GUARDRAIL_KIND: &str = "guardrail_kind";
    }
}

pub mod defaults {
    pub const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:18800";
    pub const DEFAULT_TIMEOUT_MS: u64 = 30_000;
    pub const DEFAULT_LEASE_MS: u64 = 60_000;
    pub const DEFAULT_MAX_WAIT_MS: u64 = 5_000;
    pub const DEFAULT_MEMORY_LIMIT: u64 = 10;
    pub const DEFAULT_MAX_RETRIES: u32 = 3;
    pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 8192;
}
