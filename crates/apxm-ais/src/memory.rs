//! Memory Tier Definitions
//!
//! A-PXM uses a three-tier memory hierarchy:
//!
//! 1. **STM (Short-Term Memory)**: Fast access to recent context and tool output
//! 2. **LTM (Long-Term Memory)**: Persistent knowledge store
//! 3. **Episodic**: Execution traces for reflection and debugging

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Error type for memory tier parsing.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MemoryTierParseError {
    #[error("Unknown memory tier: {0}")]
    UnknownTier(String),
}

/// Memory tier in the A-PXM memory hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum MemoryTier {
    /// Short-Term Memory: Fast access to recent context.
    /// Typically implemented as an LRU cache with O(1) access.
    #[serde(alias = "STM")]
    #[default]
    Stm,

    /// Long-Term Memory: Persistent knowledge store.
    /// Typically backed by a database with vector similarity search.
    #[serde(alias = "LTM")]
    Ltm,

    /// Episodic Memory: Execution traces for reflection.
    /// Append-only log of state transitions and events.
    Episodic,
}

impl MemoryTier {
    /// Get the tier name as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryTier::Stm => "stm",
            MemoryTier::Ltm => "ltm",
            MemoryTier::Episodic => "episodic",
        }
    }
}

impl FromStr for MemoryTier {
    type Err = MemoryTierParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "stm" | "short-term" | "short_term" => Ok(MemoryTier::Stm),
            "ltm" | "long-term" | "long_term" => Ok(MemoryTier::Ltm),
            "episodic" | "trace" | "log" => Ok(MemoryTier::Episodic),
            _ => Err(MemoryTierParseError::UnknownTier(s.to_string())),
        }
    }
}

impl MemoryTier {
    /// Get typical access latency characteristics.
    pub fn latency_class(&self) -> LatencyClass {
        match self {
            MemoryTier::Stm => LatencyClass::Microseconds,
            MemoryTier::Ltm => LatencyClass::Milliseconds,
            MemoryTier::Episodic => LatencyClass::Milliseconds,
        }
    }

    /// Check if tier supports semantic search.
    pub fn supports_semantic_search(&self) -> bool {
        match self {
            MemoryTier::Stm => false,
            MemoryTier::Ltm => true,
            MemoryTier::Episodic => false,
        }
    }

    /// Check if tier is append-only.
    pub fn is_append_only(&self) -> bool {
        matches!(self, MemoryTier::Episodic)
    }
}

impl fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Latency class for memory operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyClass {
    /// Sub-millisecond latency (microseconds).
    Microseconds,
    /// Millisecond-scale latency.
    Milliseconds,
}

impl LatencyClass {
    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            LatencyClass::Microseconds => "~1μs (in-memory)",
            LatencyClass::Milliseconds => "~1ms (persistent)",
        }
    }
}

/// Memory operation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryOperation {
    /// Read from memory.
    Read,
    /// Write to memory.
    Write,
    /// Search memory (semantic or keyword).
    Search,
    /// Delete from memory.
    Delete,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_from_str() {
        assert_eq!("stm".parse(), Ok(MemoryTier::Stm));
        assert_eq!("STM".parse(), Ok(MemoryTier::Stm));
        assert_eq!("ltm".parse(), Ok(MemoryTier::Ltm));
        assert_eq!("episodic".parse(), Ok(MemoryTier::Episodic));
        assert!("unknown".parse::<MemoryTier>().is_err());
    }

    #[test]
    fn test_tier_properties() {
        assert!(MemoryTier::Ltm.supports_semantic_search());
        assert!(!MemoryTier::Stm.supports_semantic_search());
        assert!(MemoryTier::Episodic.is_append_only());
        assert!(!MemoryTier::Stm.is_append_only());
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(format!("{}", MemoryTier::Stm), "stm");
        assert_eq!(format!("{}", MemoryTier::Ltm), "ltm");
        assert_eq!(format!("{}", MemoryTier::Episodic), "episodic");
    }
}
