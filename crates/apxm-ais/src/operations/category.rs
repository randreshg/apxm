//! Operation categories for AIS operations.

/// Categories for AIS operations, used by the scheduler to determine behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationCategory {
    /// Metadata operations: AGENT
    Metadata,
    /// Memory operations: QMEM, UMEM
    Memory,
    /// Reasoning operations: RSN, PLAN, REFLECT, VERIFY
    Reasoning,
    /// Tool operations: INV, EXC
    Tools,
    /// Control flow operations: JUMP, BRANCH_ON_VALUE, LOOP_START, LOOP_END, RETURN
    ControlFlow,
    /// Synchronization operations: MERGE, FENCE, WAIT_ALL
    Synchronization,
    /// Error handling operations: TRY_CATCH, ERR
    ErrorHandling,
    /// Communication operations: COMMUNICATE
    Communication,
    /// Internal operations: CONST_STR
    Internal,
}

impl OperationCategory {
    /// Returns true if operations in this category require LLM calls.
    pub fn requires_llm(&self) -> bool {
        matches!(self, OperationCategory::Reasoning)
    }

    /// Returns true if operations in this category are I/O bound.
    pub fn is_io_bound(&self) -> bool {
        matches!(
            self,
            OperationCategory::Memory
                | OperationCategory::Reasoning
                | OperationCategory::Tools
                | OperationCategory::Communication
        )
    }

    /// Returns true if operations affect control flow.
    pub fn affects_control_flow(&self) -> bool {
        matches!(
            self,
            OperationCategory::ControlFlow | OperationCategory::ErrorHandling
        )
    }

    /// Returns true if this is a metadata category (no execution).
    pub fn is_metadata(&self) -> bool {
        matches!(self, OperationCategory::Metadata)
    }
}
