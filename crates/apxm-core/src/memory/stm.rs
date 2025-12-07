//! Short-term memory (STM) types.
//!
//! STM entries are time-bounded key/value pairs that live in a small, fast
//! cache. Configuration keeps the tier size and default TTL compact and
//! predictable.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::types::Value;

/// Represents a single STM entry with an absolute timestamp and TTL.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct STMEntry {
    /// Unique key for the entry.
    pub key: String,
    /// Stored value.
    pub value: Value,
    /// Creation timestamp.
    pub timestamp: DateTime<Utc>,
    /// Time-to-live for the entry.
    pub ttl: Duration,
}

impl STMEntry {
    /// Creates a new STM entry using the current time.
    pub fn new(key: impl Into<String>, value: Value, ttl: Duration) -> Self {
        STMEntry {
            key: key.into(),
            value,
            timestamp: Utc::now(),
            ttl,
        }
    }

    /// Creates an entry with an explicit timestamp (useful for testing).
    pub fn with_timestamp(
        key: impl Into<String>,
        value: Value,
        timestamp: DateTime<Utc>,
        ttl: Duration,
    ) -> Self {
        STMEntry {
            key: key.into(),
            value,
            timestamp,
            ttl,
        }
    }

    /// Returns the instant when the entry should expire.
    pub fn expires_at(&self) -> Option<DateTime<Utc>> {
        ChronoDuration::from_std(self.ttl)
            .ok()
            .map(|delta| self.timestamp + delta)
    }

    /// Returns true if the entry is expired at the provided time.
    pub fn is_expired_at(&self, now: DateTime<Utc>) -> bool {
        self.expires_at()
            .map(|deadline| now >= deadline)
            .unwrap_or(true)
    }

    /// Returns the remaining TTL relative to the provided time, clamped at zero.
    pub fn remaining_ttl(&self, now: DateTime<Utc>) -> Option<Duration> {
        let deadline = self.expires_at()?;
        if now >= deadline {
            return Some(Duration::from_secs(0));
        }

        (deadline - now).to_std().ok()
    }
}

/// Configuration for STM behaviour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct STMConfig {
    /// Maximum number of entries to retain.
    pub max_size: usize,
    /// Default TTL for new entries.
    pub ttl: Duration,
}

impl STMConfig {
    /// Default maximum entries for STM.
    pub const DEFAULT_MAX_SIZE: usize = 512;
    /// Default TTL of five minutes for STM entries.
    pub const DEFAULT_TTL: Duration = Duration::from_secs(300);

    /// Creates a new STM configuration, clamping the max size to at least 1.
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        STMConfig {
            max_size: max_size.max(1),
            ttl,
        }
    }

    /// Validates that configuration values are usable.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_size == 0 {
            return Err("STM max_size must be greater than zero".to_string());
        }
        if self.ttl.is_zero() {
            return Err("STM ttl must be greater than zero".to_string());
        }
        Ok(())
    }
}

impl Default for STMConfig {
    fn default() -> Self {
        STMConfig {
            max_size: Self::DEFAULT_MAX_SIZE,
            ttl: Self::DEFAULT_TTL,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_expiry_logic() {
        let ttl = Duration::from_secs(5);
        let created_at = Utc::now() - ChronoDuration::seconds(10);
        let entry = STMEntry::with_timestamp("k", Value::Bool(true), created_at, ttl);

        assert!(entry.is_expired_at(Utc::now()));
        let before_expiry = created_at + ChronoDuration::seconds(3);
        assert!(!entry.is_expired_at(before_expiry));
    }

    #[test]
    fn test_expires_at_and_remaining() {
        let ttl = Duration::from_secs(4);
        let created_at = Utc::now();
        let entry = STMEntry::with_timestamp("id", Value::String("v".into()), created_at, ttl);

        let deadline = entry.expires_at().expect("deadline");
        assert!(deadline > created_at);

        let remaining = entry
            .remaining_ttl(created_at + ChronoDuration::seconds(2))
            .expect("remaining ttl");
        assert!(remaining.as_secs() <= 2);
    }

    #[test]
    fn test_entry_serialization_round_trip() {
        let ttl = Duration::from_secs(1);
        let entry = STMEntry::new("k", Value::Number(crate::types::Number::Integer(1)), ttl);
        let json = serde_json::to_string(&entry).expect("serialize");
        let deserialized: STMEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(entry.key, deserialized.key);
        assert_eq!(entry.value, deserialized.value);
    }

    #[test]
    fn test_config_defaults_and_validation() {
        let default = STMConfig::default();
        assert_eq!(default.max_size, STMConfig::DEFAULT_MAX_SIZE);
        assert_eq!(default.ttl, STMConfig::DEFAULT_TTL);
        assert!(default.validate().is_ok());

        let invalid = STMConfig::new(0, Duration::from_secs(0));
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_config_custom() {
        let cfg = STMConfig::new(1, Duration::from_secs(2));
        assert_eq!(cfg.max_size, 1);
        assert_eq!(cfg.ttl, Duration::from_secs(2));

        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: STMConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, restored);
    }
}
