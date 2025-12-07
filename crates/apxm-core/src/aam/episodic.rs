//! Episodic memory for AAM.
//!
//! Episodic memory records state transitions and allows querying past
//! experiences for learning and decision-making.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::aam::goal::{Goal, GoalId, GoalStatus};
use crate::types::{AISOperationType, Value};

/// Represents a change to a goal during a state transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GoalChange {
    /// A new goal was added.
    Added(Goal),
    /// A goal was removed.
    Removed(GoalId),
    /// A goal's status changed.
    StatusChanged(GoalId, GoalStatus, GoalStatus),
    /// A goal's priority changed.
    PriorityChanged(GoalId, u32, u32),
}

/// Represents a state transition in the agent's execution history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateTransition {
    /// Timestamp when the transition occurred.
    pub timestamp: DateTime<Utc>,
    /// The operation that caused this transition.
    pub operation: AISOperationType,
    /// Beliefs before the transition.
    pub before_beliefs: HashMap<String, Value>,
    /// Beliefs after the transition.
    pub after_beliefs: HashMap<String, Value>,
    /// Changes to goals during this transition.
    pub goal_changes: Vec<GoalChange>,
    /// Name of the capability used (if any).
    pub capability_used: Option<String>,
    /// Duration of the operation.
    pub duration: Duration,
    /// Additional metadata.
    pub metadata: HashMap<String, Value>,
}

impl StateTransition {
    /// Creates a new state transition.
    pub fn new(
        operation: AISOperationType,
        before_beliefs: HashMap<String, Value>,
        after_beliefs: HashMap<String, Value>,
        goal_changes: Vec<GoalChange>,
        duration: Duration,
    ) -> Self {
        StateTransition {
            timestamp: Utc::now(),
            operation,
            before_beliefs,
            after_beliefs,
            goal_changes,
            capability_used: None,
            duration,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new state transition with a capability.
    pub fn with_capability(
        operation: AISOperationType,
        before_beliefs: HashMap<String, Value>,
        after_beliefs: HashMap<String, Value>,
        goal_changes: Vec<GoalChange>,
        capability_used: String,
        duration: Duration,
    ) -> Self {
        StateTransition {
            timestamp: Utc::now(),
            operation,
            before_beliefs,
            after_beliefs,
            goal_changes,
            capability_used: Some(capability_used),
            duration,
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to the transition.
    pub fn add_metadata(&mut self, key: String, value: Value) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
}

/// Trait for episodic memory backends.
///
/// This trait allows pluggable backends for storing and querying
/// state transitions, enabling different storage strategies.
pub trait EpisodicLog: Send + Sync {
    /// Appends a state transition to the log.
    fn append(&mut self, transition: StateTransition);

    /// Queries transitions based on a filter.
    ///
    /// The filter is a function that returns true for transitions
    /// that should be included in the results.
    fn query<F>(&self, filter: F) -> Vec<StateTransition>
    where
        F: Fn(&StateTransition) -> bool;

    /// Gets the number of transitions in the log.
    fn len(&self) -> usize;

    /// Checks if the log is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory implementation of EpisodicLog.
///
/// TODO: add support for different backends with persistence and efficient querying.
pub struct InMemoryEpisodicLog {
    transitions: Vec<StateTransition>,
}

impl InMemoryEpisodicLog {
    /// Creates a new in-memory episodic log.
    pub fn new() -> Self {
        InMemoryEpisodicLog {
            transitions: Vec::new(),
        }
    }
}

impl Default for InMemoryEpisodicLog {
    fn default() -> Self {
        Self::new()
    }
}

impl EpisodicLog for InMemoryEpisodicLog {
    fn append(&mut self, transition: StateTransition) {
        self.transitions.push(transition);
    }

    fn query<F>(&self, filter: F) -> Vec<StateTransition>
    where
        F: Fn(&StateTransition) -> bool,
    {
        self.transitions
            .iter()
            .filter(|t| filter(t))
            .cloned()
            .collect()
    }

    fn len(&self) -> usize {
        self.transitions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_transition_new() {
        let before = HashMap::new();
        let after = HashMap::new();
        let transition = StateTransition::new(
            AISOperationType::UMem,
            before.clone(),
            after.clone(),
            Vec::new(),
            Duration::from_millis(100),
        );
        assert_eq!(transition.operation, AISOperationType::UMem);
        assert_eq!(transition.before_beliefs, before);
        assert_eq!(transition.after_beliefs, after);
        assert!(transition.capability_used.is_none());
    }

    #[test]
    fn test_state_transition_with_capability() {
        let before = HashMap::new();
        let after = HashMap::new();
        let transition = StateTransition::with_capability(
            AISOperationType::Inv,
            before,
            after,
            Vec::new(),
            "test_cap".to_string(),
            Duration::from_millis(50),
        );
        assert_eq!(transition.capability_used, Some("test_cap".to_string()));
    }

    #[test]
    fn test_state_transition_metadata() {
        let before = HashMap::new();
        let after = HashMap::new();
        let mut transition = StateTransition::new(
            AISOperationType::Rsn,
            before,
            after,
            Vec::new(),
            Duration::from_millis(200),
        );
        transition.add_metadata("key".to_string(), Value::String("value".to_string()));
        assert_eq!(
            transition.get_metadata("key"),
            Some(&Value::String("value".to_string()))
        );
    }

    #[test]
    fn test_goal_change() {
        let goal = Goal::new(1, "Test".to_string(), 10);
        let change = GoalChange::Added(goal.clone());
        assert!(matches!(change, GoalChange::Added(g) if g.id == 1));

        let change2 = GoalChange::StatusChanged(1, GoalStatus::Pending, GoalStatus::Active);
        assert!(matches!(
            change2,
            GoalChange::StatusChanged(id, GoalStatus::Pending, GoalStatus::Active) if id == 1
        ));
    }

    #[test]
    fn test_in_memory_episodic_log() {
        let mut log = InMemoryEpisodicLog::new();
        assert!(log.is_empty());

        let before = HashMap::new();
        let after = HashMap::new();
        let transition = StateTransition::new(
            AISOperationType::UMem,
            before,
            after,
            Vec::new(),
            Duration::from_millis(100),
        );
        log.append(transition);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_in_memory_episodic_log_query() {
        let mut log = InMemoryEpisodicLog::new();

        let before = HashMap::new();
        let after = HashMap::new();
        let transition1 = StateTransition::new(
            AISOperationType::UMem,
            before.clone(),
            after.clone(),
            Vec::new(),
            Duration::from_millis(100),
        );
        let transition2 = StateTransition::new(
            AISOperationType::Inv,
            before,
            after,
            Vec::new(),
            Duration::from_millis(200),
        );

        log.append(transition1);
        log.append(transition2);

        let results = log.query(|t| t.operation == AISOperationType::UMem);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].operation, AISOperationType::UMem);
    }

    #[test]
    fn test_state_transition_serialization() {
        let before = HashMap::new();
        let after = HashMap::new();
        let transition = StateTransition::new(
            AISOperationType::Rsn,
            before,
            after,
            Vec::new(),
            Duration::from_millis(150),
        );
        let json = serde_json::to_string(&transition).expect("serialize transition");
        let deserialized: StateTransition =
            serde_json::from_str(&json).expect("deserialize transition");
        assert_eq!(transition.operation, deserialized.operation);
    }
}
