//! Operation Metadata for Compiler Code Generation
//!
//! This module provides the `OperationMetadata` struct and static instances
//! used by the compiler for code generation. These were previously missing
//! from apxm-core, causing broken imports.

use super::definitions::{AISOperationType, OperationField};

/// Metadata for compiler code generation.
///
/// This struct provides the information needed by the compiler to emit
/// Rust code for each operation.
#[derive(Debug, Clone)]
pub struct OperationMetadata {
    /// Operation name (e.g., "qmem", "rsn").
    pub name: &'static str,
    /// Required and optional fields.
    pub fields: &'static [OperationField],
    /// Whether this operation needs async submission.
    pub needs_submission: bool,
}

impl OperationMetadata {
    /// Create metadata from an operation type.
    pub const fn from_spec(
        name: &'static str,
        fields: &'static [OperationField],
        needs_submission: bool,
    ) -> Self {
        Self {
            name,
            fields,
            needs_submission,
        }
    }

    /// Get a field by name.
    pub fn get_field(&self, name: &str) -> Option<&OperationField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Get all required field names.
    pub fn required_field_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.fields.iter().filter(|f| f.required).map(|f| f.name)
    }
}

/// Emit trait for code generation.
///
/// This trait is implemented by the compiler to emit Rust code for operations.
pub trait OperationEmit {
    /// Emit Rust code for this operation.
    fn emit(
        &self,
        emitter: &mut dyn std::fmt::Write,
        op_id: &str,
        args: &std::collections::HashMap<&'static str, String>,
    ) -> std::fmt::Result;
}

// ============================================================================
// Static Operation Metadata Instances
// ============================================================================

/// Query Memory operation metadata.
pub static QMEM: OperationMetadata = OperationMetadata {
    name: "qmem",
    fields: &[
        OperationField {
            name: "query",
            required: true,
            description: "Query string or key to search for",
        },
        OperationField {
            name: "memory_tier",
            required: false,
            description: "Target memory tier: stm, ltm, or episodic",
        },
    ],
    needs_submission: true,
};

/// Update Memory operation metadata.
pub static UMEM: OperationMetadata = OperationMetadata {
    name: "umem",
    fields: &[
        OperationField {
            name: "key",
            required: true,
            description: "Key to store the value under",
        },
        OperationField {
            name: "value",
            required: true,
            description: "Value to store",
        },
        OperationField {
            name: "memory_tier",
            required: false,
            description: "Target memory tier: stm, ltm, or episodic",
        },
    ],
    needs_submission: true,
};

/// Reason operation metadata.
pub static RSN: OperationMetadata = OperationMetadata {
    name: "rsn",
    fields: &[
        OperationField {
            name: "prompt",
            required: true,
            description: "Prompt template for reasoning",
        },
        OperationField {
            name: "model",
            required: false,
            description: "LLM model to use",
        },
        OperationField {
            name: "context",
            required: false,
            description: "Additional context for reasoning",
        },
    ],
    needs_submission: true,
};

/// Invoke Tool operation metadata.
pub static INV: OperationMetadata = OperationMetadata {
    name: "inv",
    fields: &[
        OperationField {
            name: "capability",
            required: true,
            description: "Name of the capability/tool to invoke",
        },
        OperationField {
            name: "parameters",
            required: false,
            description: "Parameters to pass to the tool",
        },
    ],
    needs_submission: true,
};

/// Merge operation metadata.
pub static MERGE: OperationMetadata = OperationMetadata {
    name: "merge",
    fields: &[OperationField {
        name: "tokens",
        required: true,
        description: "List of tokens to merge",
    }],
    needs_submission: true,
};

/// Wait All operation metadata.
pub static WAIT_ALL: OperationMetadata = OperationMetadata {
    name: "wait_all",
    fields: &[OperationField {
        name: "tokens",
        required: true,
        description: "Tokens to wait for",
    }],
    needs_submission: true,
};

/// Plan operation metadata.
pub static PLAN: OperationMetadata = OperationMetadata {
    name: "plan",
    fields: &[
        OperationField {
            name: "goal",
            required: true,
            description: "Goal to decompose into steps",
        },
        OperationField {
            name: "constraints",
            required: false,
            description: "Constraints on the plan",
        },
    ],
    needs_submission: true,
};

/// Reflect operation metadata.
pub static REFLECT: OperationMetadata = OperationMetadata {
    name: "reflect",
    fields: &[
        OperationField {
            name: "trace_query",
            required: true,
            description: "Query to retrieve trace for reflection",
        },
        OperationField {
            name: "reflection_prompt",
            required: false,
            description: "Custom prompt for reflection",
        },
    ],
    needs_submission: true,
};

/// Verify operation metadata.
pub static VERIFY: OperationMetadata = OperationMetadata {
    name: "verify",
    fields: &[
        OperationField {
            name: "claim",
            required: true,
            description: "Claim to verify",
        },
        OperationField {
            name: "evidence",
            required: true,
            description: "Evidence to check against",
        },
    ],
    needs_submission: true,
};

/// Execute Code operation metadata.
pub static EXC: OperationMetadata = OperationMetadata {
    name: "exc",
    fields: &[
        OperationField {
            name: "code",
            required: true,
            description: "Code to execute",
        },
        OperationField {
            name: "sandbox_config",
            required: false,
            description: "Sandbox configuration",
        },
    ],
    needs_submission: true,
};

/// Fence operation metadata.
pub static FENCE: OperationMetadata = OperationMetadata {
    name: "fence",
    fields: &[OperationField {
        name: "ordering",
        required: false,
        description: "Memory ordering constraint",
    }],
    needs_submission: true,
};

/// Try-Catch operation metadata.
pub static TRY_CATCH: OperationMetadata = OperationMetadata {
    name: "try_catch",
    fields: &[
        OperationField {
            name: "try_subgraph",
            required: true,
            description: "Subgraph to try executing",
        },
        OperationField {
            name: "catch_subgraph",
            required: true,
            description: "Recovery subgraph on failure",
        },
    ],
    needs_submission: true,
};

/// Error Handler operation metadata.
pub static ERR: OperationMetadata = OperationMetadata {
    name: "err",
    fields: &[OperationField {
        name: "error_handler",
        required: true,
        description: "Error handler to invoke",
    }],
    needs_submission: true,
};

/// Communicate operation metadata.
pub static COMMUNICATE: OperationMetadata = OperationMetadata {
    name: "communicate",
    fields: &[
        OperationField {
            name: "target_agent",
            required: true,
            description: "Target agent to communicate with",
        },
        OperationField {
            name: "message",
            required: true,
            description: "Message to send",
        },
        OperationField {
            name: "protocol",
            required: false,
            description: "Communication protocol to use",
        },
    ],
    needs_submission: true,
};

/// Const String operation metadata (internal).
pub static CONST_STR: OperationMetadata = OperationMetadata {
    name: "const_str",
    fields: &[OperationField {
        name: "value",
        required: true,
        description: "The string constant value",
    }],
    needs_submission: false,
};

/// Agent operation metadata (metadata).
pub static AGENT: OperationMetadata = OperationMetadata {
    name: "agent",
    fields: &[
        OperationField {
            name: "memory",
            required: false,
            description: "Memory configuration",
        },
        OperationField {
            name: "beliefs",
            required: false,
            description: "Initial beliefs",
        },
        OperationField {
            name: "goals",
            required: false,
            description: "Initial goals",
        },
        OperationField {
            name: "capabilities",
            required: false,
            description: "Available capabilities",
        },
    ],
    needs_submission: false,
};

// Control flow operations (no async submission needed)

/// Jump operation metadata.
pub static JUMP: OperationMetadata = OperationMetadata {
    name: "jump",
    fields: &[OperationField {
        name: "label",
        required: true,
        description: "Target label to jump to",
    }],
    needs_submission: false,
};

/// Branch on Value operation metadata.
pub static BRANCH_ON_VALUE: OperationMetadata = OperationMetadata {
    name: "branch_on_value",
    fields: &[
        OperationField {
            name: "token",
            required: true,
            description: "Token to evaluate",
        },
        OperationField {
            name: "value",
            required: true,
            description: "Value to compare against",
        },
        OperationField {
            name: "label_true",
            required: true,
            description: "Label if comparison is true",
        },
        OperationField {
            name: "label_false",
            required: true,
            description: "Label if comparison is false",
        },
    ],
    needs_submission: false,
};

/// Loop Start operation metadata.
pub static LOOP_START: OperationMetadata = OperationMetadata {
    name: "loop_start",
    fields: &[OperationField {
        name: "count_token",
        required: true,
        description: "Token containing iteration count",
    }],
    needs_submission: false,
};

/// Loop End operation metadata.
pub static LOOP_END: OperationMetadata = OperationMetadata {
    name: "loop_end",
    fields: &[],
    needs_submission: false,
};

/// Return operation metadata.
pub static RETURN: OperationMetadata = OperationMetadata {
    name: "return",
    fields: &[OperationField {
        name: "token",
        required: true,
        description: "Result token to return",
    }],
    needs_submission: false,
};

/// Switch operation metadata.
pub static SWITCH: OperationMetadata = OperationMetadata {
    name: "switch",
    fields: &[
        OperationField {
            name: "discriminant",
            required: true,
            description: "Token to match against case labels",
        },
        OperationField {
            name: "case_labels",
            required: true,
            description: "Array of case labels",
        },
        OperationField {
            name: "case_destinations",
            required: true,
            description: "Array of case destinations",
        },
        OperationField {
            name: "default_destination",
            required: false,
            description: "Default destination if no case matches",
        },
    ],
    needs_submission: true,
};

/// Flow Call operation metadata.
pub static FLOW_CALL: OperationMetadata = OperationMetadata {
    name: "flow_call",
    fields: &[
        OperationField {
            name: "agent_name",
            required: true,
            description: "Name of the agent to call",
        },
        OperationField {
            name: "flow_name",
            required: true,
            description: "Name of the flow to invoke",
        },
        OperationField {
            name: "args",
            required: false,
            description: "Arguments to pass to the flow",
        },
    ],
    needs_submission: true,
};

/// Get operation metadata by operation type.
pub fn get_operation_metadata(op_type: AISOperationType) -> &'static OperationMetadata {
    match op_type {
        AISOperationType::Agent => &AGENT,
        AISOperationType::QMem => &QMEM,
        AISOperationType::UMem => &UMEM,
        AISOperationType::Rsn => &RSN,
        AISOperationType::Plan => &PLAN,
        AISOperationType::Reflect => &REFLECT,
        AISOperationType::Verify => &VERIFY,
        AISOperationType::Inv => &INV,
        AISOperationType::Exc => &EXC,
        AISOperationType::Jump => &JUMP,
        AISOperationType::BranchOnValue => &BRANCH_ON_VALUE,
        AISOperationType::LoopStart => &LOOP_START,
        AISOperationType::LoopEnd => &LOOP_END,
        AISOperationType::Return => &RETURN,
        AISOperationType::Switch => &SWITCH,
        AISOperationType::FlowCall => &FLOW_CALL,
        AISOperationType::Merge => &MERGE,
        AISOperationType::Fence => &FENCE,
        AISOperationType::WaitAll => &WAIT_ALL,
        AISOperationType::TryCatch => &TRY_CATCH,
        AISOperationType::Err => &ERR,
        AISOperationType::Communicate => &COMMUNICATE,
        AISOperationType::ConstStr => &CONST_STR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::get_operation_spec;

    #[test]
    fn test_all_ops_have_metadata() {
        for op_type in AISOperationType::all_operations() {
            let metadata = get_operation_metadata(*op_type);
            assert_eq!(metadata.name, op_type.mlir_mnemonic());
        }
    }

    #[test]
    fn test_metadata_matches_spec() {
        for op_type in AISOperationType::all_operations() {
            let metadata = get_operation_metadata(*op_type);
            let spec = get_operation_spec(*op_type);
            assert_eq!(
                metadata.needs_submission, spec.needs_submission,
                "needs_submission mismatch for {:?}",
                op_type
            );
        }
    }
}
