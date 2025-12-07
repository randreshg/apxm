//! AAMState - Complete agent state with thread-safe access.
//!
//! This module provides the main AAMState struct that coordinates all
//! components of the Agent Abstract Machine: beliefs, goals, capabilities,
//! and episodic memory.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use dashmap::DashMap;
use priority_queue::PriorityQueue;

use crate::aam::capability::CapabilityMetadata;
use crate::aam::episodic::{GoalChange, StateTransition};
use crate::aam::goal::{Goal, GoalId, GoalStatus};
use crate::error::RuntimeError;
use crate::types::{AISOperationType, Value};

/// Complete agent state with thread-safe access.
///
/// AAMState maintains all components of the Agent Abstract Machine:
/// - Beliefs: Key-value memory for agent knowledge
/// - Goals: Priority-ordered objectives
/// - Capabilities: Registry of available tools
/// - Episodic Memory: Execution history and state transitions
#[derive(Clone, Debug)]
pub struct AAMState {
    /// Thread-safe beliefs storage (K→V memory).
    beliefs: Arc<DashMap<String, Value>>,
    /// Thread-safe priority queue of goal IDs ordered by priority.
    goals: Arc<Mutex<PriorityQueue<GoalId, u32>>>,
    /// Thread-safe goal details indexed by ID.
    goal_details: Arc<DashMap<GoalId, Goal>>,
    /// Thread-safe capabilities registry.
    capabilities: Arc<DashMap<String, CapabilityMetadata>>,
    /// Thread-safe episodic memory (execution traces).
    episodic: Arc<Mutex<Vec<StateTransition>>>,
}

impl AAMState {
    /// Creates a new empty AAMState.
    pub fn new() -> Self {
        AAMState {
            beliefs: Arc::new(DashMap::new()),
            goals: Arc::new(Mutex::new(PriorityQueue::new())),
            goal_details: Arc::new(DashMap::new()),
            capabilities: Arc::new(DashMap::new()),
            episodic: Arc::new(Mutex::new(Vec::new())),
        }
    }

    // ========== Belief Management ==========

    /// Sets a belief (key-value pair).
    pub fn set_belief(&self, key: String, value: Value) {
        self.beliefs.insert(key, value);
    }

    /// Gets a belief by key.
    pub fn get_belief(&self, key: &str) -> Option<Value> {
        self.beliefs.get(key).map(|entry| entry.value().clone())
    }

    /// Updates a belief atomically using a closure.
    ///
    /// The closure receives the current value (or None if the key doesn't exist)
    /// and should return the new value.
    pub fn update_belief<F>(&self, key: &str, f: F) -> Value
    where
        F: FnOnce(Option<Value>) -> Value,
    {
        if let Some(mut entry) = self.beliefs.get_mut(key) {
            let new_value = f(Some(entry.value().clone()));
            *entry = new_value.clone();
            new_value
        } else {
            let new_value = f(None);
            self.beliefs.insert(key.to_string(), new_value.clone());
            new_value
        }
    }

    /// Removes a belief by key.
    pub fn remove_belief(&self, key: &str) -> Option<Value> {
        self.beliefs.remove(key).map(|(_, v)| v)
    }

    /// Lists all beliefs as a vector of (key, value) pairs.
    pub fn list_beliefs(&self) -> Vec<(String, Value)> {
        self.beliefs
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Gets the number of beliefs.
    pub fn belief_count(&self) -> usize {
        self.beliefs.len()
    }

    // ========== Goal Management ==========

    /// Adds a new goal to the state.
    pub fn add_goal(&self, goal: Goal) -> Result<(), RuntimeError> {
        let priority = goal.priority;
        let goal_id = goal.id;
        self.goal_queue()?.push(goal_id, priority);
        self.goal_details.insert(goal_id, goal);
        Ok(())
    }

    /// Gets the next goal (highest priority) without removing it.
    pub fn peek_next_goal(&self) -> Result<Option<Goal>, RuntimeError> {
        let goals = self.goal_queue()?;
        let next = goals
            .peek()
            .and_then(|(goal_id, _)| self.goal_details.get(goal_id).map(|g| g.value().clone()));
        Ok(next)
    }

    /// Gets and removes the next goal (highest priority).
    pub fn get_next_goal(&self) -> Result<Option<Goal>, RuntimeError> {
        let mut goals = self.goal_queue()?;
        if let Some((goal_id, _)) = goals.pop() {
            drop(goals);
            Ok(self.goal_details.remove(&goal_id).map(|(_, g)| g))
        } else {
            Ok(None)
        }
    }

    /// Gets a goal by ID.
    pub fn get_goal(&self, goal_id: GoalId) -> Option<Goal> {
        self.goal_details.get(&goal_id).map(|g| g.value().clone())
    }

    /// Updates the status of a goal.
    pub fn update_goal_status(
        &self,
        goal_id: GoalId,
        status: GoalStatus,
    ) -> Result<(), RuntimeError> {
        if let Some(mut goal) = self.goal_details.get_mut(&goal_id) {
            goal.update_status(status);
            Ok(())
        } else {
            Err(RuntimeError::Memory {
                message: format!("Goal not found: {}", goal_id),
                space: Some("AAM".to_string()),
            })
        }
    }

    /// Updates the priority of a goal.
    ///
    /// This requires removing and re-adding the goal to the priority queue.
    pub fn update_goal_priority(
        &self,
        goal_id: GoalId,
        new_priority: u32,
    ) -> Result<(), RuntimeError> {
        if let Some(mut goal) = self.goal_details.get_mut(&goal_id) {
            goal.update_priority(new_priority);

            let mut goals = self.goal_queue()?;
            goals.change_priority(&goal_id, new_priority);
            Ok(())
        } else {
            Err(RuntimeError::Memory {
                message: format!("Goal not found: {}", goal_id),
                space: Some("AAM".to_string()),
            })
        }
    }

    /// Removes a goal by ID.
    pub fn remove_goal(&self, goal_id: GoalId) -> Result<Option<Goal>, RuntimeError> {
        self.goal_queue()?.remove(&goal_id);
        Ok(self.goal_details.remove(&goal_id).map(|(_, g)| g))
    }

    /// Lists all goals.
    pub fn list_goals(&self) -> Vec<Goal> {
        self.goal_details
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Gets the number of goals.
    pub fn goal_count(&self) -> usize {
        self.goal_details.len()
    }

    // ========== Capability Management ==========

    /// Registers a capability in the state.
    pub fn register_capability(&self, metadata: CapabilityMetadata) {
        self.capabilities.insert(metadata.name.clone(), metadata);
    }

    /// Gets capability metadata by name.
    pub fn get_capability(&self, name: &str) -> Option<CapabilityMetadata> {
        self.capabilities.get(name).map(|m| m.value().clone())
    }

    /// Lists all registered capabilities.
    pub fn list_capabilities(&self) -> Vec<CapabilityMetadata> {
        self.capabilities
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Checks if a capability is registered.
    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.contains_key(name)
    }

    /// Gets the number of registered capabilities.
    pub fn capability_count(&self) -> usize {
        self.capabilities.len()
    }

    // ========== Episodic Memory Management ==========

    /// Records a state transition in episodic memory.
    pub fn record_transition(
        &self,
        operation: AISOperationType,
        before_beliefs: HashMap<String, Value>,
        goal_changes: Vec<GoalChange>,
        duration: Duration,
    ) -> Result<(), RuntimeError> {
        let after_beliefs: HashMap<String, Value> = self.list_beliefs().into_iter().collect();
        let transition = StateTransition::new(
            operation,
            before_beliefs,
            after_beliefs,
            goal_changes,
            duration,
        );
        self.episodic_memory()?.push(transition);
        Ok(())
    }

    /// Records a state transition with a capability.
    pub fn record_transition_with_capability(
        &self,
        operation: AISOperationType,
        before_beliefs: HashMap<String, Value>,
        goal_changes: Vec<GoalChange>,
        capability_used: String,
        duration: Duration,
    ) -> Result<(), RuntimeError> {
        let after_beliefs: HashMap<String, Value> = self.list_beliefs().into_iter().collect();
        let transition = StateTransition::with_capability(
            operation,
            before_beliefs,
            after_beliefs,
            goal_changes,
            capability_used,
            duration,
        );
        self.episodic_memory()?.push(transition);
        Ok(())
    }

    /// Gets the most recent transitions.
    pub fn get_recent_transitions(
        &self,
        limit: usize,
    ) -> Result<Vec<StateTransition>, RuntimeError> {
        let episodic = self.episodic_memory()?;
        let len = episodic.len();
        let start = len.saturating_sub(limit);
        Ok(episodic[start..].to_vec())
    }

    /// Gets transitions related to a specific goal.
    pub fn get_transitions_for_goal(
        &self,
        goal_id: GoalId,
    ) -> Result<Vec<StateTransition>, RuntimeError> {
        let episodic = self.episodic_memory()?;
        let transitions = episodic
            .iter()
            .filter(|t| {
                t.goal_changes.iter().any(|gc| match gc {
                    GoalChange::Added(g) => g.id == goal_id,
                    GoalChange::Removed(id) => *id == goal_id,
                    GoalChange::StatusChanged(id, _, _) => *id == goal_id,
                    GoalChange::PriorityChanged(id, _, _) => *id == goal_id,
                })
            })
            .cloned()
            .collect();
        Ok(transitions)
    }

    /// Gets the number of recorded transitions.
    pub fn transition_count(&self) -> Result<usize, RuntimeError> {
        Ok(self.episodic_memory()?.len())
    }

    fn goal_queue(&self) -> Result<MutexGuard<'_, PriorityQueue<GoalId, u32>>, RuntimeError> {
        Self::lock_mutex(&self.goals, "goal queue")
    }

    fn episodic_memory(&self) -> Result<MutexGuard<'_, Vec<StateTransition>>, RuntimeError> {
        Self::lock_mutex(&self.episodic, "episodic memory")
    }

    fn lock_mutex<'a, T>(
        mutex: &'a Arc<Mutex<T>>,
        resource: &str,
    ) -> Result<MutexGuard<'a, T>, RuntimeError> {
        mutex.lock().map_err(|_| RuntimeError::Memory {
            message: format!("Failed to lock {}", resource),
            space: Some("AAM".to_string()),
        })
    }
}

impl Default for AAMState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aam_state_new() {
        let state = AAMState::new();
        assert_eq!(state.belief_count(), 0);
        assert_eq!(state.goal_count(), 0);
        assert_eq!(state.capability_count(), 0);
        assert_eq!(state.transition_count().expect("transition count"), 0);
    }

    #[test]
    fn test_belief_operations() {
        let state = AAMState::new();
        state.set_belief("key1".to_string(), Value::String("value1".to_string()));
        assert_eq!(
            state.get_belief("key1"),
            Some(Value::String("value1".to_string()))
        );
        assert_eq!(state.belief_count(), 1);

        state.update_belief("key1", |value| match value {
            Some(Value::String(existing)) => Value::String(format!("updated_{}", existing)),
            other => panic!("expected string belief, got {:?}", other),
        });
        let updated = state.get_belief("key1").expect("updated belief present");
        let updated_str = updated
            .as_string()
            .expect("updated belief remains a string");
        assert!(updated_str.contains("updated_"));

        state.remove_belief("key1");
        assert_eq!(state.belief_count(), 0);
    }

    #[test]
    fn test_goal_operations() {
        let state = AAMState::new();
        let goal1 = Goal::new(1, "Goal 1".to_string(), 10);
        let goal2 = Goal::new(2, "Goal 2".to_string(), 20);

        state.add_goal(goal1).expect("goal1 inserted");
        state.add_goal(goal2).expect("goal2 inserted");

        assert_eq!(state.goal_count(), 2);

        // Higher priority goal should be returned first
        let next = state
            .get_next_goal()
            .expect("goal queue accessible")
            .expect("goal present");
        assert_eq!(next.priority, 20);

        assert_eq!(state.goal_count(), 1);
    }

    #[test]
    fn test_goal_status_update() {
        let state = AAMState::new();
        let goal = Goal::new(1, "Test".to_string(), 10);
        state.add_goal(goal).expect("goal inserted");

        state
            .update_goal_status(1, GoalStatus::Active)
            .expect("status updated");
        let updated = state.get_goal(1).expect("goal exists");
        assert_eq!(updated.status, GoalStatus::Active);

        let result = state.update_goal_status(999, GoalStatus::Active);
        assert!(result.is_err());
    }

    #[test]
    fn test_capability_operations() {
        let state = AAMState::new();
        let schema = serde_json::json!({});
        let capability = CapabilityMetadata::new(
            "test_cap".to_string(),
            "Test capability".to_string(),
            "1.0.0".to_string(),
            schema.clone(),
            schema.clone(),
        );

        state.register_capability(capability);
        assert!(state.has_capability("test_cap"));
        assert_eq!(state.capability_count(), 1);

        let retrieved = state
            .get_capability("test_cap")
            .expect("capability present");
        assert_eq!(retrieved.name, "test_cap");
    }

    #[test]
    fn test_episodic_memory() {
        let state = AAMState::new();
        state.set_belief("key".to_string(), Value::String("value".to_string()));

        let before = HashMap::new();
        state
            .record_transition(
                AISOperationType::UMem,
                before,
                Vec::new(),
                Duration::from_millis(100),
            )
            .expect("recorded transition");

        assert_eq!(state.transition_count().expect("transition count"), 1);
        let transitions = state
            .get_recent_transitions(10)
            .expect("transitions fetched");
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].operation, AISOperationType::UMem);
    }

    #[test]
    fn test_transitions_for_goal() {
        let state = AAMState::new();
        let goal = Goal::new(1, "Test".to_string(), 10);
        state.add_goal(goal.clone()).expect("goal inserted");

        let before = HashMap::new();
        let goal_changes = vec![GoalChange::Added(goal)];
        state
            .record_transition(
                AISOperationType::Plan,
                before,
                goal_changes,
                Duration::from_millis(50),
            )
            .expect("recorded transition");

        let transitions = state
            .get_transitions_for_goal(1)
            .expect("transitions for goal");
        assert_eq!(transitions.len(), 1);
    }
}
