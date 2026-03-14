//! AIS Operation Definitions - Single Source of Truth
//!
//! This module contains the complete specification for all 32 AIS operations
//! (29 public + 1 metadata + 2 internal). Both the compiler and runtime use
//! these definitions to ensure consistent semantics.

use super::category::OperationCategory;
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Operation Type Enum
// ============================================================================

/// Represents all AIS operation types.
///
/// This enum is the canonical list of operations (32 total):
/// - 1 metadata operation (AgentOp)
/// - 29 public operations
/// - 2 internal operations (ConstStr, Yield)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AISOperationType {
    // Metadata Operation (1) - matches tablegen ais.agent
    /// Agent metadata declaration (memory, beliefs, goals, capabilities).
    Agent,

    // Memory Operations (2)
    /// Query memory (read from memory system).
    #[serde(rename = "QMEM")]
    QMem,
    /// Update memory (write to memory system).
    #[serde(rename = "UMEM")]
    UMem,

    // LLM Operations (3) - Compiler markers for critical path analysis
    /// Simple Q&A (no extended thinking) - LOW latency marker.
    Ask,
    /// Extended thinking with budget - HIGH latency marker.
    Think,
    /// Structured reasoning with beliefs/goals - MEDIUM latency marker.
    Reason,

    // Planning & Analysis Operations (3)
    /// Planning operation (generate a plan using LLM).
    Plan,
    /// Reflection operation (analyze execution trace).
    Reflect,
    /// Verification operation (fact-check against evidence).
    Verify,

    // Tool Operations (3)
    /// Invoke a capability (tool/function call).
    Inv,
    /// Execute code in sandbox.
    Exc,
    /// Print output to stdout.
    Print,

    // Control Flow Operations (7)
    /// Unconditional jump to label.
    Jump,
    /// Branch based on value comparison.
    BranchOnValue,
    /// Loop start marker.
    LoopStart,
    /// Loop end marker.
    LoopEnd,
    /// Return from subgraph with result.
    Return,
    /// Multi-way branch based on string value (switch/case).
    Switch,
    /// Call a flow on another agent.
    FlowCall,

    // Synchronization Operations (3)
    /// Merge multiple tokens into one.
    Merge,
    /// Memory fence (synchronization barrier).
    Fence,
    /// Wait for all input tokens to be ready.
    WaitAll,

    // Error Handling Operations (2)
    /// Try-catch exception handling.
    TryCatch,
    /// Error handler invocation.
    Err,

    // Communication Operations (1)
    /// Communication between agents.
    Communicate,

    // Coordination Operations (Phase 1 ISA Extensions)
    /// Update agent goals at runtime (set/remove/clear).
    UpdateGoal,
    /// Enforce preconditions before execution continues.
    Guard,
    /// Atomically claim a task from a shared work queue.
    Claim,
    /// Suspend execution pending human-in-the-loop review.
    Pause,
    /// Resume a suspended execution from a PAUSE checkpoint.
    Resume,

    // Internal Operations (not part of public AIS)
    /// String constant (compiler internal).
    ConstStr,
    /// Yield value from switch case region (compiler internal).
    Yield,
}

impl fmt::Display for AISOperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Metadata
            AISOperationType::Agent => write!(f, "AGENT"),
            // Memory
            AISOperationType::QMem => write!(f, "QMEM"),
            AISOperationType::UMem => write!(f, "UMEM"),
            // LLM Operations (markers for runtime config lookup)
            AISOperationType::Ask => write!(f, "ASK"),
            AISOperationType::Think => write!(f, "THINK"),
            AISOperationType::Reason => write!(f, "REASON"),
            // Planning & Analysis
            AISOperationType::Plan => write!(f, "PLAN"),
            AISOperationType::Reflect => write!(f, "REFLECT"),
            AISOperationType::Verify => write!(f, "VERIFY"),
            // Tools
            AISOperationType::Inv => write!(f, "INV"),
            AISOperationType::Exc => write!(f, "EXC"),
            AISOperationType::Print => write!(f, "PRINT"),
            // Control Flow
            AISOperationType::Jump => write!(f, "JUMP"),
            AISOperationType::BranchOnValue => write!(f, "BRANCH_ON_VALUE"),
            AISOperationType::LoopStart => write!(f, "LOOP_START"),
            AISOperationType::LoopEnd => write!(f, "LOOP_END"),
            AISOperationType::Return => write!(f, "RETURN"),
            AISOperationType::Switch => write!(f, "SWITCH"),
            AISOperationType::FlowCall => write!(f, "FLOW_CALL"),
            // Synchronization
            AISOperationType::Merge => write!(f, "MERGE"),
            AISOperationType::Fence => write!(f, "FENCE"),
            AISOperationType::WaitAll => write!(f, "WAIT_ALL"),
            // Error Handling
            AISOperationType::TryCatch => write!(f, "TRY_CATCH"),
            AISOperationType::Err => write!(f, "ERR"),
            // Communication
            AISOperationType::Communicate => write!(f, "COMMUNICATE"),
            // Coordination (Phase 1 ISA Extensions)
            AISOperationType::UpdateGoal => write!(f, "UPDATE_GOAL"),
            AISOperationType::Guard => write!(f, "GUARD"),
            AISOperationType::Claim => write!(f, "CLAIM"),
            AISOperationType::Pause => write!(f, "PAUSE"),
            AISOperationType::Resume => write!(f, "RESUME"),
            // Internal
            AISOperationType::ConstStr => write!(f, "CONST_STR"),
            AISOperationType::Yield => write!(f, "YIELD"),
        }
    }
}

impl AISOperationType {
    /// Returns the MLIR mnemonic for this operation (e.g., "rsn", "qmem").
    pub fn mlir_mnemonic(&self) -> &'static str {
        match self {
            AISOperationType::Agent => "agent",
            AISOperationType::QMem => "qmem",
            AISOperationType::UMem => "umem",
            AISOperationType::Ask => "ask",
            AISOperationType::Think => "think",
            AISOperationType::Reason => "reason",
            AISOperationType::Plan => "plan",
            AISOperationType::Reflect => "reflect",
            AISOperationType::Verify => "verify",
            AISOperationType::Inv => "inv",
            AISOperationType::Exc => "exc",
            AISOperationType::Print => "print",
            AISOperationType::Jump => "jump",
            AISOperationType::BranchOnValue => "branch_on_value",
            AISOperationType::LoopStart => "loop_start",
            AISOperationType::LoopEnd => "loop_end",
            AISOperationType::Return => "return",
            AISOperationType::Switch => "switch",
            AISOperationType::FlowCall => "flow_call",
            AISOperationType::Merge => "merge",
            AISOperationType::Fence => "fence",
            AISOperationType::WaitAll => "wait_all",
            AISOperationType::TryCatch => "try_catch",
            AISOperationType::Err => "err",
            AISOperationType::Communicate => "communicate",
            AISOperationType::UpdateGoal => "update_goal",
            AISOperationType::Guard => "guard",
            AISOperationType::Claim => "claim",
            AISOperationType::Pause => "pause",
            AISOperationType::Resume => "resume",
            AISOperationType::ConstStr => "const_str",
            AISOperationType::Yield => "yield",
        }
    }

    /// Maps a wire-format operation kind index (u32) to an `AISOperationType`.
    ///
    /// This is the **single source of truth** for the u32→AISOperationType mapping
    /// used by both the compiler artifact parser and the runtime sub-DAG parser.
    /// Must stay in sync with the C++ `OperationKind` enum in `ArtifactEmitter.cpp`:
    ///   Inv=0, Ask=1, QMem=2, ..., Print=22, Think=23, Reason=24
    pub fn from_wire_index(index: u32) -> Option<AISOperationType> {
        /// Wire-format operation kind table. Index = OperationKind from ArtifactEmitter.cpp.
        const WIRE_OP_KIND_MAP: [AISOperationType; 25] = [
            AISOperationType::Inv,           // 0
            AISOperationType::Ask,           // 1
            AISOperationType::QMem,          // 2
            AISOperationType::UMem,          // 3
            AISOperationType::Plan,          // 4
            AISOperationType::WaitAll,       // 5
            AISOperationType::Merge,         // 6
            AISOperationType::Fence,         // 7
            AISOperationType::Exc,           // 8
            AISOperationType::Communicate,   // 9
            AISOperationType::Reflect,       // 10
            AISOperationType::Verify,        // 11
            AISOperationType::Err,           // 12
            AISOperationType::Return,        // 13
            AISOperationType::Jump,          // 14
            AISOperationType::BranchOnValue, // 15
            AISOperationType::LoopStart,     // 16
            AISOperationType::LoopEnd,       // 17
            AISOperationType::TryCatch,      // 18
            AISOperationType::ConstStr,      // 19
            AISOperationType::Switch,        // 20
            AISOperationType::FlowCall,      // 21
            AISOperationType::Print,         // 22
            AISOperationType::Think,         // 23
            AISOperationType::Reason,        // 24
        ];
        WIRE_OP_KIND_MAP.get(index as usize).copied()
    }

    /// Get all operation types (32 total: 27 original + 5 phase-1 extensions).
    pub fn all_operations() -> &'static [AISOperationType] {
        &[
            AISOperationType::Agent,
            AISOperationType::QMem,
            AISOperationType::UMem,
            AISOperationType::Ask,
            AISOperationType::Think,
            AISOperationType::Reason,
            AISOperationType::Plan,
            AISOperationType::Reflect,
            AISOperationType::Verify,
            AISOperationType::Inv,
            AISOperationType::Exc,
            AISOperationType::Print,
            AISOperationType::Jump,
            AISOperationType::BranchOnValue,
            AISOperationType::LoopStart,
            AISOperationType::LoopEnd,
            AISOperationType::Return,
            AISOperationType::Switch,
            AISOperationType::FlowCall,
            AISOperationType::Merge,
            AISOperationType::Fence,
            AISOperationType::WaitAll,
            AISOperationType::TryCatch,
            AISOperationType::Err,
            AISOperationType::Communicate,
            AISOperationType::UpdateGoal,
            AISOperationType::Guard,
            AISOperationType::Claim,
            AISOperationType::Pause,
            AISOperationType::Resume,
            AISOperationType::ConstStr,
            AISOperationType::Yield,
        ]
    }
}

// ============================================================================
// Field Specifications
// ============================================================================

/// A field in an operation specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationField {
    /// Field name.
    pub name: &'static str,
    /// Whether the field is required.
    pub required: bool,
    /// Description of the field.
    pub description: &'static str,
}

impl OperationField {
    /// Create a required field.
    pub const fn required(name: &'static str, description: &'static str) -> Self {
        Self {
            name,
            required: true,
            description,
        }
    }

    /// Create an optional field.
    pub const fn optional(name: &'static str, description: &'static str) -> Self {
        Self {
            name,
            required: false,
            description,
        }
    }
}

// ============================================================================
// Latency Classification
// ============================================================================

/// Expected latency tier for an operation, used by the scheduler for
/// critical-path analysis and by agents for cost estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationLatency {
    /// No external I/O — executes in microseconds (control flow, sync).
    None,
    /// Local I/O only — millisecond range (memory, code sandbox).
    Low,
    /// Single LLM call or tool invocation — seconds range.
    Medium,
    /// Extended thinking / multi-step LLM — tens of seconds.
    High,
}

impl OperationLatency {
    /// Returns a human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            OperationLatency::None => "none",
            OperationLatency::Low => "low",
            OperationLatency::Medium => "medium",
            OperationLatency::High => "high",
        }
    }
}

impl fmt::Display for OperationLatency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Operation Specification
// ============================================================================

/// Complete specification of an AIS operation.
///
/// This is the single source of truth for operation semantics. Both the
/// compiler (for validation and code generation) and runtime (for dispatch
/// and execution) use these specifications.
#[derive(Debug, Clone)]
pub struct OperationSpec {
    /// Operation type identifier.
    pub op_type: AISOperationType,
    /// Human-readable name (e.g., "QueryMemory", "Reason").
    pub name: &'static str,
    /// Operation category.
    pub category: OperationCategory,
    /// Short description of what the operation does (one line).
    pub description: &'static str,
    /// Extended description with usage guidance (for agents and docs).
    pub long_description: &'static str,
    /// Expected latency tier.
    pub latency: OperationLatency,
    /// Minimal JSON example showing typical usage (for agents).
    pub example_json: Option<&'static str>,
    /// Required and optional input fields.
    pub fields: &'static [OperationField],
    /// Whether this operation needs async execution (submission to executor).
    pub needs_submission: bool,
    /// Minimum number of input tokens required.
    pub min_inputs: u32,
    /// Whether operation produces output tokens.
    pub produces_output: bool,
}

impl OperationSpec {
    /// Get a field by name.
    pub fn get_field(&self, name: &str) -> Option<&OperationField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Get all required fields.
    pub fn required_fields(&self) -> impl Iterator<Item = &OperationField> {
        self.fields.iter().filter(|f| f.required)
    }
}

// ============================================================================
// Operation Registry
// ============================================================================

/// All AIS operations (metadata + public + internal) with complete specs.
pub static AIS_OPERATIONS: &[OperationSpec] = &[
    // ========== Metadata Operation (1) ==========
    OperationSpec {
        op_type: AISOperationType::Agent,
        name: "Agent",
        category: OperationCategory::Metadata,
        description: "Agent structural declaration (memory, beliefs, goals, capabilities)",
        long_description: "Declares an agent's identity and initial AAM state. Every graph must \
            have exactly one AGENT node. It configures memory tiers (STM/LTM/Episodic), initial \
            beliefs and goals, and the capabilities the agent can invoke. The runtime uses this \
            to initialize the agent's AAM before executing any other node.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 0, "op": "AGENT", "attributes": {"memory": {"stm": true}, "beliefs": {"role": "analyst"}, "goals": ["summarize data"], "capabilities": ["search", "calculate"]}}"#),
        fields: &[
            OperationField::optional("memory", "Memory configuration"),
            OperationField::optional("beliefs", "Initial beliefs"),
            OperationField::optional("goals", "Initial goals"),
            OperationField::optional("capabilities", "Available capabilities"),
        ],
        needs_submission: false,
        min_inputs: 0,
        produces_output: false,
    },
    // ========== Memory Operations (2) ==========
    OperationSpec {
        op_type: AISOperationType::QMem,
        name: "QueryMemory",
        category: OperationCategory::Memory,
        description: "Retrieve data from memory (STM, LTM, or Episodic)",
        long_description: "Reads from the agent's memory system. Supports three tiers: \
            STM (short-term, per-execution scratch), LTM (long-term, persists across runs), \
            and Episodic (execution traces). The query string is matched against stored keys. \
            Returns the stored value or null if not found.",
        latency: OperationLatency::Low,
        example_json: Some(r#"{"id": 2, "op": "QMEM", "attributes": {"query": "user_name", "memory_tier": "stm"}}"#),
        fields: &[
            OperationField::required("query", "Query string or key to search for"),
            OperationField::optional("memory_tier", "Target memory tier: stm, ltm, or episodic"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::UMem,
        name: "UpdateMemory",
        category: OperationCategory::Memory,
        description: "Persist or update data in memory",
        long_description: "Writes a key-value pair to the agent's memory system. If the key \
            already exists, the value is overwritten. Supports the same three memory tiers as \
            QMEM. Use FENCE after UMEM if subsequent QMEM nodes must see the write.",
        latency: OperationLatency::Low,
        example_json: Some(r#"{"id": 3, "op": "UMEM", "attributes": {"key": "summary", "value": "{{node_2}}", "memory_tier": "stm"}}"#),
        fields: &[
            OperationField::required("key", "Key to store the value under"),
            OperationField::required("value", "Value to store"),
            OperationField::optional("memory_tier", "Target memory tier: stm, ltm, or episodic"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== LLM Operations (3) ==========
    OperationSpec {
        op_type: AISOperationType::Ask,
        name: "Ask",
        category: OperationCategory::Reasoning,
        description: "Simple Q&A with LLM (no extended thinking) - LOW latency",
        long_description: "Sends a prompt to the configured LLM and returns the response. \
            The lightest LLM operation — no chain-of-thought or extended thinking. Use for \
            straightforward questions, classifications, extractions, or reformulations. \
            Template strings support {{node_N}} interpolation for dataflow inputs.",
        latency: OperationLatency::Medium,
        example_json: Some(r#"{"id": 1, "op": "ASK", "attributes": {"template_str": "Summarize: {{node_0}}"}}"#),
        fields: &[
            OperationField::required("template_str", "Prompt template for the question"),
            OperationField::optional("temperature", "Sampling temperature (0.0-1.0)"),
            OperationField::optional("model", "LLM model override (uses config default)"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Think,
        name: "Think",
        category: OperationCategory::Reasoning,
        description: "Extended thinking with token_budget - HIGH latency",
        long_description: "Activates extended thinking (chain-of-thought) with a configurable \
            token budget. The LLM produces internal reasoning before the final answer. Use for \
            complex multi-step problems, math, code generation, or planning that benefits from \
            deliberate reasoning. The budget controls how many tokens the model can spend thinking.",
        latency: OperationLatency::High,
        example_json: Some(r#"{"id": 1, "op": "THINK", "attributes": {"template_str": "Solve step by step: {{node_0}}", "budget": 4096}}"#),
        fields: &[
            OperationField::required("template_str", "Prompt template for deep reasoning"),
            OperationField::optional("budget", "Token budget for extended thinking"),
            OperationField::optional("temperature", "Sampling temperature (0.0-1.0)"),
            OperationField::optional("model", "LLM model override (uses config default)"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Reason,
        name: "Reason",
        category: OperationCategory::Reasoning,
        description: "Structured reasoning with belief/goal updates - MEDIUM latency",
        long_description: "Performs structured reasoning that can update the agent's beliefs \
            and goals (AAM state). Unlike ASK, the runtime parses the LLM response for belief \
            and goal mutations. Use when the agent needs to update its internal state based on \
            new information. Supports structured JSON output mode.",
        latency: OperationLatency::Medium,
        example_json: Some(r#"{"id": 1, "op": "REASON", "attributes": {"template_str": "Given {{node_0}}, update your analysis", "structured": true}}"#),
        fields: &[
            OperationField::required("template_str", "Prompt template for structured reasoning"),
            OperationField::optional("temperature", "Sampling temperature (0.0-1.0)"),
            OperationField::optional("model", "LLM model override (uses config default)"),
            OperationField::optional("structured", "Enable structured JSON output"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Planning & Analysis Operations (3) ==========
    OperationSpec {
        op_type: AISOperationType::Plan,
        name: "Plan",
        category: OperationCategory::Reasoning,
        description: "Decompose goal into AIS subgraph",
        long_description: "Uses the LLM to decompose a high-level goal into a sequence of \
            concrete steps. The output is a structured plan that can be used to drive subsequent \
            nodes. Supports optional constraints to bound the plan space.",
        latency: OperationLatency::High,
        example_json: Some(r#"{"id": 1, "op": "PLAN", "attributes": {"goal": "Research and summarize recent AI papers", "constraints": "max 5 steps"}}"#),
        fields: &[
            OperationField::required("goal", "Goal to decompose into steps"),
            OperationField::optional("constraints", "Constraints on the plan"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Reflect,
        name: "Reflect",
        category: OperationCategory::Reasoning,
        description: "Analyze execution trace for self-improvement",
        long_description: "Retrieves past execution traces and asks the LLM to analyze them \
            for patterns, failures, or improvements. Useful for iterative refinement loops \
            where the agent learns from its own execution history.",
        latency: OperationLatency::Medium,
        example_json: Some(r#"{"id": 4, "op": "REFLECT", "attributes": {"trace_query": "last_execution"}}"#),
        fields: &[
            OperationField::required("trace_query", "Query to retrieve trace for reflection"),
            OperationField::optional("reflection_prompt", "Custom prompt for reflection"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Verify,
        name: "Verify",
        category: OperationCategory::Reasoning,
        description: "Fact-check outputs against evidence",
        long_description: "Cross-references a claim against provided evidence using the LLM. \
            Returns a verification result with confidence score. Use after ASK/THINK/REASON \
            nodes to validate outputs before acting on them.",
        latency: OperationLatency::Medium,
        example_json: Some(r#"{"id": 5, "op": "VERIFY", "attributes": {"claim": "{{node_3}}", "evidence": "{{node_4}}"}}"#),
        fields: &[
            OperationField::required("claim", "Claim to verify"),
            OperationField::required("evidence", "Evidence to check against"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Tool Operations (3) ==========
    OperationSpec {
        op_type: AISOperationType::Inv,
        name: "InvokeTool",
        category: OperationCategory::Tools,
        description: "Call external tool with structured params; store result",
        long_description: "Invokes a registered capability (tool or function) by name. \
            The capability must be declared in the AGENT node's capabilities list or \
            registered in the runtime's CapabilityRegistry. Parameters are passed as \
            a JSON object. The tool's return value becomes this node's output token.",
        latency: OperationLatency::Medium,
        example_json: Some(r#"{"id": 2, "op": "INV", "attributes": {"capability": "web_search", "parameters": {"query": "{{node_1}}"}}}"#),
        fields: &[
            OperationField::required("capability", "Name of the capability/tool to invoke"),
            OperationField::optional("parameters", "Parameters to pass to the tool"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Exc,
        name: "ExecuteCode",
        category: OperationCategory::Tools,
        description: "Run code in a sandboxed environment; update Beliefs",
        long_description: "Executes arbitrary code in a sandboxed environment. The sandbox \
            prevents file system access, network calls, and other side effects unless \
            explicitly allowed. The code's stdout/return value becomes the output token. \
            Execution results are also written to the agent's beliefs.",
        latency: OperationLatency::Low,
        example_json: Some(r#"{"id": 3, "op": "EXC", "attributes": {"code": "print(2 + 2)"}}"#),
        fields: &[
            OperationField::required("code", "Code to execute"),
            OperationField::optional("sandbox_config", "Sandbox configuration"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Print,
        name: "PrintOutput",
        category: OperationCategory::Tools,
        description: "Print output to stdout for debugging or user display",
        long_description: "Writes a message to stdout. Supports {{node_N}} template \
            interpolation. Useful for debugging graphs during development or displaying \
            final results to the user. The message is also stored as the output token.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 4, "op": "PRINT", "attributes": {"message": "Result: {{node_3}}"}}"#),
        fields: &[OperationField::required("message", "Message to print")],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Control Flow Operations (7) ==========
    OperationSpec {
        op_type: AISOperationType::Jump,
        name: "Jump",
        category: OperationCategory::ControlFlow,
        description: "Unconditional jump to a labeled instruction",
        long_description: "Transfers control flow unconditionally to a target label. \
            The label must correspond to a node ID in the graph. Edges from this node \
            use Control dependency type.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 5, "op": "JUMP", "attributes": {"label": "7"}}"#),
        fields: &[OperationField::required("label", "Target label to jump to")],
        needs_submission: false,
        min_inputs: 0,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::BranchOnValue,
        name: "BranchOnValue",
        category: OperationCategory::ControlFlow,
        description: "Conditional branch based on token value comparison",
        long_description: "Evaluates an input token against a value and branches to one of \
            two labels. If the token matches the value, control goes to label_true; otherwise \
            to label_false. Used for if/else patterns in agent workflows.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 5, "op": "BRANCH_ON_VALUE", "attributes": {"token": "{{node_4}}", "value": "yes", "label_true": "6", "label_false": "7"}}"#),
        fields: &[
            OperationField::required("token", "Token to evaluate"),
            OperationField::required("value", "Value to compare against"),
            OperationField::required("label_true", "Label if comparison is true"),
            OperationField::required("label_false", "Label if comparison is false"),
        ],
        needs_submission: false,
        min_inputs: 1,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::LoopStart,
        name: "LoopStart",
        category: OperationCategory::ControlFlow,
        description: "Begin bounded loop",
        long_description: "Marks the beginning of a bounded loop. The count_token specifies \
            how many iterations to execute. Must be paired with a LOOP_END node. The compiler \
            verifies loop bounds at compile time to prevent infinite loops.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 3, "op": "LOOP_START", "attributes": {"count_token": "3"}}"#),
        fields: &[OperationField::required(
            "count_token",
            "Token containing iteration count",
        )],
        needs_submission: false,
        min_inputs: 1,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::LoopEnd,
        name: "LoopEnd",
        category: OperationCategory::ControlFlow,
        description: "End bounded loop",
        long_description: "Marks the end of a bounded loop started by LOOP_START. The runtime \
            decrements the loop counter and branches back to LOOP_START if iterations remain.",
        latency: OperationLatency::None,
        example_json: None,
        fields: &[],
        needs_submission: false,
        min_inputs: 0,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::Return,
        name: "Return",
        category: OperationCategory::ControlFlow,
        description: "Return from subgraph with result token",
        long_description: "Returns a value from a subgraph or flow. The token attribute \
            specifies which node's output to return. Used as the terminal node in flows \
            invoked via FLOW_CALL.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 6, "op": "RETURN", "attributes": {"token": "{{node_5}}"}}"#),
        fields: &[OperationField::required("token", "Result token to return")],
        needs_submission: false,
        min_inputs: 1,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Switch,
        name: "Switch",
        category: OperationCategory::ControlFlow,
        description: "Multi-way branch based on string value comparison",
        long_description: "Routes execution to one of several branches based on matching \
            a discriminant token against case labels. Each case specifies a label string \
            and a destination node. If no case matches, the default destination is used. \
            The matched branch's result becomes the output token.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 3, "op": "SWITCH", "attributes": {"discriminant": "{{node_2}}", "cases": [{"label": "math", "node_id": 4}, {"label": "code", "node_id": 5}], "default": "6"}}"#),
        fields: &[
            OperationField::required("discriminant", "Token to match against case labels"),
            OperationField::required("cases", "Array of case label/destination pairs"),
            OperationField::optional("default", "Default destination if no case matches"),
        ],
        needs_submission: true,
        min_inputs: 1,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::FlowCall,
        name: "FlowCall",
        category: OperationCategory::ControlFlow,
        description: "Call a flow on another agent with implicit parallelism",
        long_description: "Invokes a named flow on a target agent. The target agent executes \
            its flow graph independently and returns the result. Multiple FLOW_CALL nodes can \
            execute in parallel if they have no data dependencies between them. This is the \
            primary mechanism for multi-agent composition.",
        latency: OperationLatency::High,
        example_json: Some(r#"{"id": 4, "op": "FLOW_CALL", "attributes": {"agent_name": "researcher", "flow_name": "analyze", "args": {"topic": "{{node_1}}"}}}"#),
        fields: &[
            OperationField::required("agent_name", "Name of the agent to call"),
            OperationField::required("flow_name", "Name of the flow to invoke"),
            OperationField::optional("args", "Arguments to pass to the flow"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Synchronization Operations (3) ==========
    OperationSpec {
        op_type: AISOperationType::Merge,
        name: "Merge",
        category: OperationCategory::Synchronization,
        description: "Sync parallel paths; aggregate tokens into one",
        long_description: "Waits for multiple parallel branches to complete and combines \
            their output tokens into a single aggregated result. Used after parallel \
            FLOW_CALL or fan-out patterns to collect results before further processing.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 6, "op": "MERGE", "attributes": {"tokens": ["{{node_3}}", "{{node_4}}", "{{node_5}}"]}}"#),
        fields: &[OperationField::required(
            "tokens",
            "List of tokens to merge",
        )],
        needs_submission: true,
        min_inputs: 1,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Fence,
        name: "Fence",
        category: OperationCategory::Synchronization,
        description: "Memory barrier; order prior QMEM/UMEM operations",
        long_description: "Ensures all preceding memory operations (UMEM writes) are visible \
            to subsequent QMEM reads. Without a FENCE, the scheduler may reorder memory \
            operations for parallelism. Place between UMEM and QMEM when ordering matters.",
        latency: OperationLatency::None,
        example_json: None,
        fields: &[OperationField::optional(
            "ordering",
            "Memory ordering constraint",
        )],
        needs_submission: true,
        min_inputs: 0,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::WaitAll,
        name: "WaitAll",
        category: OperationCategory::Synchronization,
        description: "Block until all specified tokens are available",
        long_description: "Blocks execution until all listed input tokens are ready. Unlike \
            MERGE, it does not combine the tokens — it simply acts as a synchronization barrier. \
            Commonly used before a node that needs all its inputs but doesn't need them merged.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 5, "op": "WAIT_ALL", "attributes": {"tokens": ["{{node_2}}", "{{node_3}}"]}}"#),
        fields: &[OperationField::required("tokens", "Tokens to wait for")],
        needs_submission: true,
        min_inputs: 1,
        produces_output: true,
    },
    // ========== Error Handling Operations (2) ==========
    OperationSpec {
        op_type: AISOperationType::TryCatch,
        name: "TryCatch",
        category: OperationCategory::ErrorHandling,
        description: "Structured exception handling with recovery subgraph",
        long_description: "Wraps a try subgraph with a catch recovery subgraph. If any node \
            in the try subgraph fails, execution transfers to the catch subgraph. The catch \
            subgraph receives the error context and can attempt recovery or graceful degradation.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 2, "op": "TRY_CATCH", "attributes": {"try_subgraph": "3", "catch_subgraph": "4"}}"#),
        fields: &[
            OperationField::required("try_subgraph", "Subgraph to try executing"),
            OperationField::required("catch_subgraph", "Recovery subgraph on failure"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Err,
        name: "HandleError",
        category: OperationCategory::ErrorHandling,
        description: "Invoke recovery template on failure; update Goals/Beliefs",
        long_description: "Handles an error by invoking a recovery template. The error handler \
            can update the agent's goals and beliefs to reflect the failure and adapt the agent's \
            strategy. Typically used inside TRY_CATCH catch subgraphs.",
        latency: OperationLatency::Medium,
        example_json: Some(r#"{"id": 4, "op": "ERR", "attributes": {"error_handler": "retry_with_fallback"}}"#),
        fields: &[OperationField::required(
            "error_handler",
            "Error handler to invoke",
        )],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Communication Operations (1) ==========
    OperationSpec {
        op_type: AISOperationType::Communicate,
        name: "Communicate",
        category: OperationCategory::Communication,
        description: "Send message to recipient using protocol",
        long_description: "Sends a message from this agent to another agent. The target agent \
            must be reachable in the runtime's agent registry. Supports different communication \
            protocols (direct, broadcast, request-reply). The response from the target agent \
            becomes this node's output token.",
        latency: OperationLatency::Medium,
        example_json: Some(r#"{"id": 3, "op": "COMMUNICATE", "attributes": {"target_agent": "reviewer", "message": "Please review: {{node_2}}"}}"#),
        fields: &[
            OperationField::required("target_agent", "Target agent to communicate with"),
            OperationField::required("message", "Message to send"),
            OperationField::optional("protocol", "Communication protocol to use"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Phase 1 ISA Extensions (5) ==========
    OperationSpec {
        op_type: AISOperationType::UpdateGoal,
        name: "UpdateGoal",
        category: OperationCategory::Memory,
        description: "Modify AAM goals at runtime: set, remove, or clear",
        long_description: "Dynamically modifies the agent's goal set during execution. \
            Supports three actions: 'set' (upsert a goal with priority), 'remove' (delete \
            a specific goal), and 'clear' (remove all goals). Goal changes are visible to \
            subsequent REASON and REFLECT nodes.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 3, "op": "UPDATE_GOAL", "attributes": {"goal_id": "optimize_latency", "action": "set", "priority": 2}}"#),
        fields: &[
            OperationField::required(
                "goal_id",
                "Goal identifier (used as description key for upsert/remove)",
            ),
            OperationField::optional("action", "Action to perform: set (default), remove, clear"),
            OperationField::optional("priority", "Goal priority (u32, default: 1)"),
        ],
        needs_submission: false,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Guard,
        name: "Guard",
        category: OperationCategory::ControlFlow,
        description: "Enforce preconditions: halt or skip based on condition",
        long_description: "Evaluates a condition expression against the input token. If the \
            condition fails, the guard either halts execution with an error or skips the \
            downstream subgraph (configurable via on_fail). Use to enforce invariants like \
            confidence thresholds, non-null checks, or content validation.",
        latency: OperationLatency::None,
        example_json: Some(r#"{"id": 3, "op": "GUARD", "attributes": {"condition": "> 0.8", "on_fail": "skip", "error_message": "Confidence too low"}}"#),
        fields: &[
            OperationField::required(
                "condition",
                "Condition expression: '> 0.8', '!= null', 'not_empty', etc.",
            ),
            OperationField::optional("error_message", "Message on failure"),
            OperationField::optional("on_fail", "Failure mode: halt (default) or skip"),
        ],
        needs_submission: false,
        min_inputs: 1,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Claim,
        name: "Claim",
        category: OperationCategory::Communication,
        description: "Atomically claim a task from a shared work queue via APXM server",
        long_description: "Claims a task from a distributed work queue managed by the APXM \
            server. The claim is atomic — only one agent gets each task. The claimed task \
            is leased for a configurable duration. If the agent doesn't complete within the \
            lease, the task returns to the queue for other agents.",
        latency: OperationLatency::Low,
        example_json: Some(r#"{"id": 2, "op": "CLAIM", "attributes": {"queue": "review_tasks", "lease_ms": 30000}}"#),
        fields: &[
            OperationField::required("queue", "Queue name to claim from"),
            OperationField::optional("lease_ms", "Lease duration in ms (default: 60000)"),
            OperationField::optional("max_wait_ms", "Max time to wait for a task (default: 5000)"),
            OperationField::optional("server_url", "Override APXM_SERVER_URL env var"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Pause,
        name: "Pause",
        category: OperationCategory::Communication,
        description: "Suspend execution pending human-in-the-loop review via checkpoint",
        long_description: "Creates a checkpoint and suspends execution until a human resumes \
            it via the APXM server API. The pause message is displayed to the human reviewer. \
            Optionally sends a webhook notification. The human can provide input that becomes \
            this node's output token when RESUME is called.",
        latency: OperationLatency::High,
        example_json: Some(r#"{"id": 5, "op": "PAUSE", "attributes": {"message": "Please review the analysis before proceeding"}}"#),
        fields: &[
            OperationField::required("message", "Human-readable message explaining the pause"),
            OperationField::optional(
                "checkpoint_id",
                "Stable checkpoint ID (auto-generated if omitted)",
            ),
            OperationField::optional("timeout_ms", "Max wait in ms (0 = indefinite, default: 0)"),
            OperationField::optional("poll_interval_ms", "Polling interval in ms (default: 2000)"),
            OperationField::optional(
                "notification_url",
                "Webhook URL to notify on pause creation",
            ),
            OperationField::optional("server_url", "Override APXM_SERVER_URL env var"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Resume,
        name: "Resume",
        category: OperationCategory::ControlFlow,
        description: "Resume a suspended PAUSE checkpoint; polls server until human resumes; returns human_input",
        long_description: "Polls the APXM server for a specific checkpoint until a human \
            resumes it. When resumed, the human's input (if any) becomes this node's output \
            token. Configurable polling interval and max attempts prevent indefinite blocking.",
        latency: OperationLatency::High,
        example_json: Some(r#"{"id": 6, "op": "RESUME", "attributes": {"checkpoint": "review_checkpoint_1"}}"#),
        fields: &[
            OperationField::required("checkpoint", "Checkpoint ID to resume from"),
            OperationField::optional(
                "poll_max_attempts",
                "Max polling attempts (default 60 × 5s = 5 min)",
            ),
            OperationField::optional(
                "poll_interval_ms",
                "Interval between polls in ms (default 5000)",
            ),
            OperationField::optional("server_url", "Override APXM_SERVER_URL env var"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Internal Operations (2) ==========
    OperationSpec {
        op_type: AISOperationType::ConstStr,
        name: "ConstStr",
        category: OperationCategory::Internal,
        description: "String constant (compiler internal for string literals)",
        long_description: "Compiler-internal operation that produces a constant string value. \
            Not available in the public AIS. The compiler generates CONST_STR nodes when \
            lowering template strings to explicit dataflow.",
        latency: OperationLatency::None,
        example_json: None,
        fields: &[OperationField::required(
            "value",
            "The string constant value",
        )],
        needs_submission: false,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Yield,
        name: "Yield",
        category: OperationCategory::Internal,
        description: "Yield value from switch case region (terminates region)",
        long_description: "Compiler-internal operation that terminates a switch case region \
            and yields a value to the parent SWITCH node. Not available in the public AIS.",
        latency: OperationLatency::None,
        example_json: None,
        fields: &[OperationField::required(
            "value",
            "The value to yield from the region",
        )],
        needs_submission: false,
        min_inputs: 1,
        produces_output: true,
    },
];

// ============================================================================
// Lookup Functions
// ============================================================================

/// Get the specification for an operation type.
pub fn get_operation_spec(op_type: AISOperationType) -> &'static OperationSpec {
    AIS_OPERATIONS
        .iter()
        .find(|s| s.op_type == op_type)
        .unwrap_or_else(|| {
            panic!(
                "Operation {:?} has no specification. This is a bug - all operations must have specs.",
                op_type
            )
        })
}

/// Get all operation specifications.
pub fn get_all_operations() -> impl Iterator<Item = &'static OperationSpec> {
    AIS_OPERATIONS.iter()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_ops_have_specs() {
        for op_type in AISOperationType::all_operations() {
            let spec = get_operation_spec(*op_type);
            assert_eq!(spec.op_type, *op_type);
        }
    }

    #[test]
    fn test_operation_counts() {
        assert_eq!(
            AIS_OPERATIONS.len(),
            32,
            "Expected 32 total operations (1 metadata + 29 public + 2 internal)"
        );
        assert_eq!(
            AISOperationType::all_operations().len(),
            32,
            "Expected 32 total operation types"
        );
    }

    #[test]
    fn test_mlir_mnemonics() {
        assert_eq!(AISOperationType::Agent.mlir_mnemonic(), "agent");
        assert_eq!(AISOperationType::Ask.mlir_mnemonic(), "ask");
        assert_eq!(AISOperationType::Think.mlir_mnemonic(), "think");
        assert_eq!(AISOperationType::Reason.mlir_mnemonic(), "reason");
        assert_eq!(AISOperationType::QMem.mlir_mnemonic(), "qmem");
        assert_eq!(AISOperationType::WaitAll.mlir_mnemonic(), "wait_all");
    }

    #[test]
    fn test_from_wire_index() {
        // Spot-check key indices matching ArtifactEmitter.cpp OperationKind
        assert_eq!(
            AISOperationType::from_wire_index(0),
            Some(AISOperationType::Inv)
        );
        assert_eq!(
            AISOperationType::from_wire_index(1),
            Some(AISOperationType::Ask)
        );
        assert_eq!(
            AISOperationType::from_wire_index(19),
            Some(AISOperationType::ConstStr)
        );
        assert_eq!(
            AISOperationType::from_wire_index(20),
            Some(AISOperationType::Switch)
        );
        assert_eq!(
            AISOperationType::from_wire_index(24),
            Some(AISOperationType::Reason)
        );
        // Out-of-range returns None
        assert_eq!(AISOperationType::from_wire_index(25), None);
        assert_eq!(AISOperationType::from_wire_index(u32::MAX), None);
    }

    #[test]
    fn test_wire_index_round_trip_coverage() {
        let mut seen = std::collections::HashSet::new();
        for i in 0u32..25 {
            let op = AISOperationType::from_wire_index(i);
            assert!(
                op.is_some(),
                "from_wire_index({i}) returned None — gap in wire mapping"
            );
            let op = op.unwrap();
            assert!(
                seen.insert(op),
                "from_wire_index({i}) returned duplicate {op:?}"
            );
        }
        assert_eq!(
            seen.len(),
            25,
            "Expected 25 distinct wire-indexed operations"
        );
        assert_eq!(
            AISOperationType::from_wire_index(25),
            None,
            "Index 25 should be out of range"
        );
    }

    #[test]
    fn test_wire_indexed_ops_subset_of_all_ops() {
        let all_ops: std::collections::HashSet<AISOperationType> =
            AISOperationType::all_operations().iter().copied().collect();
        for i in 0u32..25 {
            let op = AISOperationType::from_wire_index(i).unwrap();
            assert!(
                all_ops.contains(&op),
                "Wire-indexed op {op:?} (index {i}) is not in all_operations()"
            );
        }
    }
}
