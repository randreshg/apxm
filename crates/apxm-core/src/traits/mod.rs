//! Shared traits for APXM.
//!
//! These traits define the contracts implemented by capabilities, LLM backends,
//! and operation executors. Implementations live in higher layers (runtime).

pub mod capability;
pub mod executor;
pub mod llm;

pub use capability::{Capability, CapabilityError};
pub use executor::{ExecutionError, OperationExecutor};
pub use llm::{
    FinishReason, LLMBackend, LLMError, LLMRequest, LLMResponse, LLMResponseStream, TokenUsage,
};
