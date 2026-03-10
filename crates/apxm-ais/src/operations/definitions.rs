//! AIS Operation Definitions - Single Source of Truth
//!
//! This module contains the complete specification for all 31 AIS operations
//! (28 public + 1 metadata + 2 internal). Both the compiler and runtime use
//! these definitions to ensure consistent semantics.

use super::category::OperationCategory;
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Operation Type Enum
// ============================================================================

/// Represents all AIS operation types.
///
/// This enum is the canonical list of operations (31 total):
/// - 1 metadata operation (AgentOp)
/// - 28 public operations
/// - 2 internal operations (ConstStr, Yield)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AISOperationType {
    // Metadata Operation (1) - matches tablegen ais.agent
    /// Agent metadata declaration (memory, beliefs, goals, capabilities).
    Agent,

    // Memory Operations (2)
    /// Query memory (read from memory system).
    QMem,
    /// Update memory (write to memory system).
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

    /// Returns true if this is a public AIS operation (part of the 28).
    pub fn is_public(&self) -> bool {
        !matches!(
            self,
            AISOperationType::ConstStr | AISOperationType::Yield | AISOperationType::Agent
        )
    }

    /// Returns true if this is a metadata operation.
    pub fn is_metadata(&self) -> bool {
        matches!(self, AISOperationType::Agent)
    }

    /// Returns true if this is an internal operation.
    pub fn is_internal(&self) -> bool {
        matches!(self, AISOperationType::ConstStr | AISOperationType::Yield)
    }

    /// Get all public operation types (28 operations).
    pub fn public_operations() -> &'static [AISOperationType] {
        &[
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
            // Phase 1 ISA extensions
            AISOperationType::UpdateGoal,
            AISOperationType::Guard,
            AISOperationType::Claim,
            AISOperationType::Pause,
            AISOperationType::Resume,
        ]
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
    /// Description of what the operation does.
    pub description: &'static str,
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

    /// Check if a field is required.
    pub fn is_field_required(&self, name: &str) -> bool {
        self.get_field(name).is_some_and(|f| f.required)
    }

    /// Get all required fields.
    pub fn required_fields(&self) -> impl Iterator<Item = &OperationField> {
        self.fields.iter().filter(|f| f.required)
    }

    /// Get all optional fields.
    pub fn optional_fields(&self) -> impl Iterator<Item = &OperationField> {
        self.fields.iter().filter(|f| !f.required)
    }
}

// ============================================================================
// Metadata Operations (1) - Agent structural declaration
// ============================================================================

/// Metadata operations (1 operation).
pub static METADATA_OPERATIONS: &[OperationSpec] = &[OperationSpec {
    op_type: AISOperationType::Agent,
    name: "Agent",
    category: OperationCategory::Metadata,
    description: "Agent structural declaration (memory, beliefs, goals, capabilities)",
    fields: &[
        OperationField::optional("memory", "Memory configuration"),
        OperationField::optional("beliefs", "Initial beliefs"),
        OperationField::optional("goals", "Initial goals"),
        OperationField::optional("capabilities", "Available capabilities"),
    ],
    needs_submission: false,
    min_inputs: 0,
    produces_output: false,
}];

// ============================================================================
// Operation Registry: 28 Public Operations
// ============================================================================

/// All 29 public AIS operations with complete metadata.
pub static AIS_OPERATIONS: &[OperationSpec] = &[
    // ========== Memory Operations (2) ==========
    OperationSpec {
        op_type: AISOperationType::QMem,
        name: "QueryMemory",
        category: OperationCategory::Memory,
        description: "Retrieve data from memory (STM, LTM, or Episodic)",
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
        fields: &[
            OperationField::required("key", "Key to store the value under"),
            OperationField::required("value", "Value to store"),
            OperationField::optional("memory_tier", "Target memory tier: stm, ltm, or episodic"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== LLM Operations (3) - Compiler markers for critical path analysis ==========
    OperationSpec {
        op_type: AISOperationType::Ask,
        name: "Ask",
        category: OperationCategory::Reasoning,
        description: "Simple Q&A with LLM (no extended thinking) - LOW latency",
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
        fields: &[OperationField::required("message", "Message to print")],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Control Flow Operations (5) ==========
    OperationSpec {
        op_type: AISOperationType::Jump,
        name: "Jump",
        category: OperationCategory::ControlFlow,
        description: "Unconditional jump to a labeled instruction",
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
        fields: &[
            OperationField::required("target_agent", "Target agent to communicate with"),
            OperationField::required("message", "Message to send"),
            OperationField::optional("protocol", "Communication protocol to use"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Phase 1 ISA Extensions (4) ==========
    OperationSpec {
        op_type: AISOperationType::UpdateGoal,
        name: "UpdateGoal",
        category: OperationCategory::Memory,
        description: "Modify AAM goals at runtime: set, remove, or clear",
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
];

// ============================================================================
// Internal Operations (not part of public AIS)
// ============================================================================

/// Internal operations used by the compiler but not exposed in the public AIS.
pub static INTERNAL_OPERATIONS: &[OperationSpec] = &[
    OperationSpec {
        op_type: AISOperationType::ConstStr,
        name: "ConstStr",
        category: OperationCategory::Internal,
        description: "String constant (compiler internal for string literals)",
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
///
/// This function never fails - all operation types have corresponding specs.
/// It checks metadata, public, and internal operations.
pub fn get_operation_spec(op_type: AISOperationType) -> &'static OperationSpec {
    // Check metadata operations
    if let Some(spec) = METADATA_OPERATIONS.iter().find(|s| s.op_type == op_type) {
        return spec;
    }

    // Check public operations
    if let Some(spec) = AIS_OPERATIONS.iter().find(|s| s.op_type == op_type) {
        return spec;
    }

    // Check internal operations
    if let Some(spec) = INTERNAL_OPERATIONS.iter().find(|s| s.op_type == op_type) {
        return spec;
    }

    // This should never happen if the operation type enum and specs are in sync
    panic!(
        "Operation {:?} has no specification. This is a bug - all operations must have specs.",
        op_type
    );
}

/// Get all operation specifications (metadata + public + internal).
pub fn get_all_operations() -> impl Iterator<Item = &'static OperationSpec> {
    METADATA_OPERATIONS
        .iter()
        .chain(AIS_OPERATIONS.iter())
        .chain(INTERNAL_OPERATIONS.iter())
}

/// Get only public operation specifications.
pub fn get_public_operations() -> impl Iterator<Item = &'static OperationSpec> {
    AIS_OPERATIONS.iter()
}

/// Find an operation specification by name.
///
/// Searches metadata, public, and internal operations.
/// Returns None if no operation with the given name exists.
pub fn find_operation_by_name(name: &str) -> Option<&'static OperationSpec> {
    get_all_operations().find(|s| s.name == name)
}

/// Find an operation specification by MLIR mnemonic.
pub fn find_operation_by_mnemonic(mnemonic: &str) -> Option<&'static OperationSpec> {
    get_all_operations().find(|s| s.op_type.mlir_mnemonic() == mnemonic)
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
            METADATA_OPERATIONS.len(),
            1,
            "Expected 1 metadata operation"
        );
        assert_eq!(
            AIS_OPERATIONS.len(),
            29,
            "Expected 29 public AIS operations (24 original + 5 phase-1 extensions)"
        );
        assert_eq!(
            INTERNAL_OPERATIONS.len(),
            2,
            "Expected 2 internal operations (ConstStr, Yield)"
        );
        assert_eq!(
            AISOperationType::all_operations().len(),
            32,
            "Expected 32 total operations (27 original + 5 phase-1 extensions)"
        );
    }

    #[test]
    fn test_agent_is_metadata() {
        assert!(AISOperationType::Agent.is_metadata());
        assert!(!AISOperationType::Agent.is_public());
        assert!(!AISOperationType::Agent.is_internal());
    }

    #[test]
    fn test_const_str_is_internal() {
        assert!(AISOperationType::ConstStr.is_internal());
        assert!(!AISOperationType::ConstStr.is_public());
        assert!(!AISOperationType::ConstStr.is_metadata());
    }

    #[test]
    fn test_llm_ops_are_public() {
        // All three LLM ops are public
        assert!(AISOperationType::Ask.is_public());
        assert!(AISOperationType::Think.is_public());
        assert!(AISOperationType::Reason.is_public());
        // None are internal or metadata
        assert!(!AISOperationType::Ask.is_internal());
        assert!(!AISOperationType::Think.is_internal());
        assert!(!AISOperationType::Reason.is_internal());
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
