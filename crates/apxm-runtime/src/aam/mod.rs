//! Agent Abstract Machine (AAM) state management.
//!
//! This module provides the first pass of the AAM used by the runtime. It tracks
//! beliefs (key/value map), goals (priority queue), registered capabilities, and
//! episodic state transitions. The interface is intentionally conservative so we
//! can evolve it alongside the rest of the runtime.

pub mod effects;

use apxm_core::types::values::Value;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Prefix for staged belief keys (used by QMEM).
pub const STAGED_BELIEF_PREFIX: &str = "_stage:";

/// Shared handle to the Agent Abstract Machine state.
#[derive(Clone, Default)]
pub struct Aam {
    inner: Arc<RwLock<AamState>>,
}

impl Aam {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(AamState::new())),
        }
    }

    /// Apply a mutation to the AAM state.
    pub fn apply_transition<F>(&self, label: TransitionLabel, f: F) -> TransitionRecord
    where
        F: FnOnce(&mut AamState) -> TransitionDelta,
    {
        let mut state = self.inner.write();
        state.apply_transition(label, f)
    }

    /// Read-only snapshot of beliefs.
    pub fn beliefs(&self) -> HashMap<String, Value> {
        self.inner.read().beliefs.clone()
    }

    pub fn get_belief(&self, key: &str) -> Option<Value> {
        self.inner.read().beliefs.get(key).cloned()
    }

    pub fn set_belief(
        &self,
        key: String,
        value: Value,
        label: TransitionLabel,
    ) -> TransitionRecord {
        self.apply_transition(label, move |state| {
            let mut delta = TransitionDelta::default();
            let before = state.beliefs.insert(key.clone(), value.clone());
            delta.belief_changes.insert(key, (before, Some(value)));
            delta
        })
    }

    pub fn add_goal(&self, goal: Goal, label: TransitionLabel) -> TransitionRecord {
        self.apply_transition(label, move |state| state.add_goal(goal))
    }

    pub fn register_capability(
        &self,
        name: String,
        metadata: CapabilityRecord,
        label: TransitionLabel,
    ) -> TransitionRecord {
        self.apply_transition(label, move |state| {
            state.capabilities.insert(name.clone(), metadata.clone());
            let mut delta = TransitionDelta::default();
            delta
                .capability_changes
                .push(CapabilityChange::Registered { name, metadata });
            delta
        })
    }

    pub fn recent_transitions(&self, last_n: usize) -> Vec<TransitionRecord> {
        let state = self.inner.read();
        let start = state.transitions.len().saturating_sub(last_n);
        state.transitions[start..].to_vec()
    }

    pub fn checkpoint(&self) -> AamCheckpoint {
        self.inner.read().snapshot()
    }

    pub fn restore(&self, checkpoint: &AamCheckpoint) {
        self.inner.write().restore(checkpoint);
    }

    pub fn enter_operation(&self, op_id: u64) {
        let mut state = self.inner.write();
        state.call_stack.push(CallFrame {
            op_id,
            entered_at: Utc::now(),
        });
    }

    pub fn exit_operation(&self) -> Option<u64> {
        self.inner.write().call_stack.pop().map(|frame| frame.op_id)
    }

    pub fn register_exception_handler(&self, op_id: u64, handler_id: u64) {
        self.inner
            .write()
            .exception_handlers
            .insert(op_id, handler_id);
    }

    pub fn current_exception_handler(&self) -> Option<u64> {
        let state = self.inner.read();
        state
            .call_stack
            .last()
            .and_then(|frame| state.exception_handlers.get(&frame.op_id))
            .copied()
    }

    pub fn goals(&self) -> Vec<Goal> {
        self.inner.read().goal_details.values().cloned().collect()
    }

    pub fn top_goal(&self) -> Option<Goal> {
        let state = self.inner.read();
        state
            .goals
            .peek()
            .and_then(|(id, _)| state.goal_details.get(id).cloned())
    }

    pub fn update_goal_status(
        &self,
        goal_id: GoalId,
        new_status: GoalStatus,
        label: TransitionLabel,
    ) -> Option<TransitionRecord> {
        let mut state = self.inner.write();
        if !state.goal_details.contains_key(&goal_id) {
            return None;
        }
        Some(state.apply_transition(label, move |state| {
            state.update_goal_status(goal_id, new_status)
        }))
    }

    pub fn remove_goal(&self, goal_id: GoalId, label: TransitionLabel) -> Option<TransitionRecord> {
        let mut state = self.inner.write();
        if !state.goal_details.contains_key(&goal_id) {
            return None;
        }
        Some(state.apply_transition(label, move |state| state.remove_goal(goal_id)))
    }

    pub fn has_capability(&self, name: &str) -> bool {
        self.inner.read().capabilities.contains_key(name)
    }

    pub fn capabilities(&self) -> HashMap<String, CapabilityRecord> {
        self.inner.read().capabilities.clone()
    }
}

/// Internal AAM state.
pub struct AamState {
    pub beliefs: HashMap<String, Value>,
    pub goals: PriorityQueue<GoalId, u32>,
    pub goal_details: HashMap<GoalId, Goal>,
    pub capabilities: HashMap<String, CapabilityRecord>,
    pub transitions: Vec<TransitionRecord>,
    pub call_stack: Vec<CallFrame>,
    pub exception_handlers: HashMap<u64, u64>,
}

impl Default for AamState {
    fn default() -> Self {
        Self::new()
    }
}

impl AamState {
    pub fn new() -> Self {
        Self {
            beliefs: HashMap::new(),
            goals: PriorityQueue::new(),
            goal_details: HashMap::new(),
            capabilities: HashMap::new(),
            transitions: Vec::new(),
            call_stack: Vec::new(),
            exception_handlers: HashMap::new(),
        }
    }

    fn apply_transition<F>(&mut self, label: TransitionLabel, f: F) -> TransitionRecord
    where
        F: FnOnce(&mut Self) -> TransitionDelta,
    {
        let before = self.beliefs.clone();
        let delta = f(self);
        let belief_changes = Self::diff_beliefs(&before, &self.beliefs, delta.belief_changes);

        let record = TransitionRecord {
            timestamp: Utc::now(),
            label,
            belief_changes,
            goal_changes: delta.goal_changes,
            capability_changes: delta.capability_changes,
        };
        self.transitions.push(record.clone());
        record
    }

    fn diff_beliefs(
        before_snapshot: &HashMap<String, Value>,
        after: &HashMap<String, Value>,
        mut explicit_changes: HashMap<String, (Option<Value>, Option<Value>)>,
    ) -> HashMap<String, (Option<Value>, Option<Value>)> {
        for key in before_snapshot.keys().chain(after.keys()) {
            if explicit_changes.contains_key(key) {
                continue;
            }
            let before = before_snapshot.get(key).cloned();
            let after = after.get(key).cloned();
            if before != after {
                explicit_changes.insert(key.clone(), (before, after));
            }
        }
        explicit_changes
    }

    fn add_goal(&mut self, goal: Goal) -> TransitionDelta {
        let priority = goal.priority;
        let id = goal.id;
        self.goals.push(id, priority);
        self.goal_details.insert(id, goal.clone());

        let mut delta = TransitionDelta::default();
        delta.goal_changes.push(GoalChange::Added(goal));
        delta
    }

    fn update_goal_status(&mut self, goal_id: GoalId, new_status: GoalStatus) -> TransitionDelta {
        let mut delta = TransitionDelta::default();
        if let Some(goal) = self.goal_details.get_mut(&goal_id) {
            let old = goal.status.clone();
            goal.status = new_status.clone();
            delta.goal_changes.push(GoalChange::StatusChanged {
                id: goal_id,
                from: old,
                to: new_status,
            });
        }
        delta
    }

    fn remove_goal(&mut self, goal_id: GoalId) -> TransitionDelta {
        let mut delta = TransitionDelta::default();
        self.goals.remove(&goal_id);
        if self.goal_details.remove(&goal_id).is_some() {
            delta.goal_changes.push(GoalChange::Removed(goal_id));
        }
        delta
    }

    pub fn snapshot(&self) -> AamCheckpoint {
        AamCheckpoint {
            beliefs: self.beliefs.clone(),
            goals: self.goal_details.values().cloned().collect(),
            timestamp: Utc::now(),
        }
    }

    fn restore(&mut self, checkpoint: &AamCheckpoint) {
        self.beliefs = checkpoint.beliefs.clone();
        self.goals.clear();
        self.goal_details.clear();
        for goal in &checkpoint.goals {
            self.goals.push(goal.id, goal.priority);
            self.goal_details.insert(goal.id, goal.clone());
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransitionRecord {
    pub timestamp: DateTime<Utc>,
    pub label: TransitionLabel,
    pub belief_changes: HashMap<String, (Option<Value>, Option<Value>)>,
    pub goal_changes: Vec<GoalChange>,
    pub capability_changes: Vec<CapabilityChange>,
}

#[derive(Debug, Clone, Default)]
pub struct TransitionDelta {
    pub belief_changes: HashMap<String, (Option<Value>, Option<Value>)>,
    pub goal_changes: Vec<GoalChange>,
    pub capability_changes: Vec<CapabilityChange>,
}

/// Labels for transitions recorded in episodic memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionLabel {
    Operation { op_id: u64, op_type: Option<String> },
    Custom(String),
}

impl TransitionLabel {
    pub fn custom(label: impl Into<String>) -> Self {
        TransitionLabel::Custom(label.into())
    }

    pub fn operation(op_id: u64, op_type: impl Into<String>) -> Self {
        TransitionLabel::Operation {
            op_id,
            op_type: Some(op_type.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub description: String,
    pub priority: u32,
    pub status: GoalStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GoalId(Uuid);

impl GoalId {
    pub fn new() -> Self {
        GoalId(Uuid::now_v7())
    }
}

impl std::fmt::Display for GoalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GoalStatus {
    Active,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum GoalChange {
    Added(Goal),
    Removed(GoalId),
    StatusChanged {
        id: GoalId,
        from: GoalStatus,
        to: GoalStatus,
    },
}

#[derive(Debug, Clone)]
pub enum CapabilityChange {
    Registered {
        name: String,
        metadata: CapabilityRecord,
    },
}

#[derive(Debug, Clone)]
pub struct CallFrame {
    pub op_id: u64,
    pub entered_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AamCheckpoint {
    pub beliefs: HashMap<String, Value>,
    pub goals: Vec<Goal>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityRecord {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
    pub cost_estimate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn belief_updates_recorded() {
        let aam = Aam::new();
        let label = TransitionLabel::custom("test");
        aam.set_belief("user".into(), Value::String("Alice".into()), label.clone());
        aam.set_belief("user".into(), Value::String("Bob".into()), label.clone());

        let recent = aam.recent_transitions(2);
        assert_eq!(recent.len(), 2);
        assert!(recent[1].belief_changes.get("user").is_some());
    }

    #[test]
    fn register_goal_updates_priority() {
        let aam = Aam::new();
        let goal = Goal {
            id: GoalId::new(),
            description: "test".into(),
            priority: 90,
            status: GoalStatus::Active,
        };
        let record = aam.add_goal(goal.clone(), TransitionLabel::custom("goal"));
        assert!(
            matches!(record.goal_changes.get(0), Some(GoalChange::Added(g)) if g.id == goal.id)
        );
    }
}
