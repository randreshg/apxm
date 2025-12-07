//! Agent Abstract Machine (AAM) module.
//!
//! This module provides the formal model of agent state, including Beliefs,
//! Goals, Capabilities, and Episodic Memory. All components are thread-safe
//! and support concurrent access from multiple execution components.

pub mod belief;
pub mod capability;
pub mod episodic;
pub mod goal;
pub mod state;

pub use belief::Belief;
pub use capability::{
    CapabilityExecutionProfile, CapabilityMetadata, CapabilitySchemas, Parameter,
};
pub use episodic::{EpisodicLog, GoalChange, StateTransition};
pub use goal::{Goal, GoalId, GoalStatus};
pub use state::AAMState;
