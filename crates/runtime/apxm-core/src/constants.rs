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
    }
}
