//! Belief management for AAM.
//!
//! Beliefs represent the agent's knowledge base as key-value pairs with timestamps.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::Value;

/// Represents a single belief in the agent's knowledge base.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Belief {
    /// The key identifying this belief.
    pub key: String,
    /// The value of this belief.
    pub value: Value,
    /// Timestamp when this belief was created or last updated.
    pub timestamp: DateTime<Utc>,
}

impl Belief {
    /// Creates a new belief with the current timestamp
    pub fn new(key: String, value: Value) -> Self {
        Belief {
            key,
            value,
            timestamp: Utc::now(),
        }
    }

    /// Creates a new belief with a specific timestamp.
    pub fn with_timestamp(key: String, value: Value, timestamp: DateTime<Utc>) -> Self {
        Belief {
            key,
            value,
            timestamp,
        }
    }

    /// Updates the belief value and timestamp.
    pub fn update(&mut self, value: Value) {
        self.value = value;
        self.timestamp = Utc::now();
    }

    /// Gets the belief key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Gets the belief value.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Gets the timestamp.
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_new() {
        let belief = Belief::new(
            "test_key".to_string(),
            Value::String("test_value".to_string()),
        );
        assert_eq!(belief.key(), "test_key");
        assert!(matches!(belief.value(), Value::String(s) if s == "test_value"));
        assert!(belief.timestamp() <= Utc::now());
    }

    #[test]
    fn test_belief_with_timestamp() {
        let timestamp = Utc::now();
        let belief = Belief::with_timestamp(
            "key".to_string(),
            Value::Number(crate::types::Number::Integer(42)),
            timestamp,
        );
        assert_eq!(belief.timestamp(), timestamp);
    }

    #[test]
    fn test_belief_update() {
        let mut belief = Belief::new("key".to_string(), Value::Bool(true));
        let old_timestamp = belief.timestamp();
        std::thread::sleep(std::time::Duration::from_millis(10));
        belief.update(Value::Bool(false));
        assert!(matches!(belief.value(), Value::Bool(false)));
        assert!(belief.timestamp() > old_timestamp);
    }

    #[test]
    fn test_belief_serialization() {
        let belief = Belief::new("key".to_string(), Value::String("value".to_string()));
        let json = serde_json::to_string(&belief).expect("serialize belief");
        let deserialized: Belief = serde_json::from_str(&json).expect("deserialize belief");
        assert_eq!(belief.key(), deserialized.key());
    }
}
