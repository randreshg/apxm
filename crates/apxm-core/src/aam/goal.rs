//! Goal management for AAM.
//!
//! Goals represent objectives that the agent is trying to achieve, with
//! priority-based ordering for efficient goal selection.

use std::cmp::Ordering;
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::Value;

/// Type alias for goal identifiers.
pub type GoalId = u64;

/// Represents the status of a goal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    /// Goal is active and being worked on.
    Active,
    /// Goal is pending and waiting to be started.
    Pending,
    /// Goal has been completed successfully.
    Completed,
    /// Goal has failed.
    Failed,
    /// Goal has been cancelled.
    Cancelled,
}

/// Represents a goal that the agent is trying to achieve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Goal {
    /// Unique identifier for this goal.
    pub id: GoalId,
    /// Human-readable description of the goal.
    pub description: String,
    /// Priority level (higher = more important).
    pub priority: u32,
    /// Current status of the goal.
    pub status: GoalStatus,
    /// Timestamp when the goal was created.
    pub created_at: DateTime<Utc>,
    /// Timestamp when the goal was completed (if applicable).
    pub completed_at: Option<DateTime<Utc>>,
    /// Additional metadata as key-value pairs.
    pub metadata: HashMap<String, Value>,
}

impl Goal {
    /// Creates a new goal with the current timestamp.
    pub fn new(id: GoalId, description: String, priority: u32) -> Self {
        Goal {
            id,
            description,
            priority,
            status: GoalStatus::Pending,
            created_at: Utc::now(),
            completed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new goal with a specific status.
    pub fn with_status(id: GoalId, description: String, priority: u32, status: GoalStatus) -> Self {
        Goal {
            id,
            description,
            priority,
            status,
            created_at: Utc::now(),
            completed_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Updates the goal status and sets completed_at if the goal is completed or failed.
    pub fn update_status(&mut self, status: GoalStatus) {
        self.status = status.clone();
        if matches!(status, GoalStatus::Completed | GoalStatus::Failed) {
            self.completed_at = Some(Utc::now());
        }
    }

    /// Updates the goal priority.
    pub fn update_priority(&mut self, priority: u32) {
        self.priority = priority;
    }

    /// Adds metadata to the goal.
    pub fn add_metadata(&mut self, key: String, value: Value) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
}

impl PartialEq for Goal {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Goal {}

impl Ord for Goal {
    /// Compare goals by priority (higher priority = greater).
    /// For equal priorities, compare by ID for consistency.
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => self.id.cmp(&other.id),
            other => other,
        }
    }
}

impl PartialOrd for Goal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_new() {
        let goal = Goal::new(1, "Test goal".to_string(), 10);
        assert_eq!(goal.id, 1);
        assert_eq!(goal.description, "Test goal");
        assert_eq!(goal.priority, 10);
        assert_eq!(goal.status, GoalStatus::Pending);
        assert!(goal.completed_at.is_none());
    }

    #[test]
    fn test_goal_with_status() {
        let goal = Goal::with_status(2, "Active goal".to_string(), 20, GoalStatus::Active);
        assert_eq!(goal.status, GoalStatus::Active);
    }

    #[test]
    fn test_goal_update_status() {
        let mut goal = Goal::new(1, "Test".to_string(), 10);
        goal.update_status(GoalStatus::Completed);
        assert_eq!(goal.status, GoalStatus::Completed);
        assert!(goal.completed_at.is_some());
    }

    #[test]
    fn test_goal_update_priority() {
        let mut goal = Goal::new(1, "Test".to_string(), 10);
        goal.update_priority(20);
        assert_eq!(goal.priority, 20);
    }

    #[test]
    fn test_goal_metadata() {
        let mut goal = Goal::new(1, "Test".to_string(), 10);
        goal.add_metadata("key".to_string(), Value::String("value".to_string()));
        assert_eq!(
            goal.get_metadata("key"),
            Some(&Value::String("value".to_string()))
        );
    }

    #[test]
    fn test_goal_ordering() {
        let goal1 = Goal::new(1, "Low priority".to_string(), 10);
        let goal2 = Goal::new(2, "High priority".to_string(), 20);
        assert!(goal2 > goal1);
    }

    #[test]
    fn test_goal_ordering_same_priority() {
        let goal1 = Goal::new(1, "First".to_string(), 10);
        let goal2 = Goal::new(2, "Second".to_string(), 10);
        assert!(goal2 > goal1); // Higher ID = greater when priority is equal
    }

    #[test]
    fn test_goal_equality() {
        let goal1 = Goal::new(1, "Test".to_string(), 10);
        let goal2 = Goal::new(1, "Different".to_string(), 20);
        assert_eq!(goal1, goal2); // Same ID = equal
    }

    #[test]
    fn test_goal_serialization() {
        let goal = Goal::new(1, "Test goal".to_string(), 10);
        let json = serde_json::to_string(&goal).expect("serialize goal");
        let deserialized: Goal = serde_json::from_str(&json).expect("deserialize goal");
        assert_eq!(goal.id, deserialized.id);
        assert_eq!(goal.description, deserialized.description);
        assert_eq!(goal.priority, deserialized.priority);
    }
}
