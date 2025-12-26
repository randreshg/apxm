//! Memory Tier Definitions
//!
//! A-PXM uses a three-tier memory hierarchy:
//!
//! 1. **STM (Short-Term Memory)**: Fast access to recent context and tool output
//! 2. **LTM (Long-Term Memory)**: Persistent knowledge store
//! 3. **Episodic**: Execution traces for reflection and debugging

use serde::{Deserialize, Serialize};
use std::fmt;

/// Memory tier in the A-PXM memory hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryTier {
    /// Short-Term Memory: Fast access to recent context.
    /// Typically implemented as an LRU cache with O(1) access.
    #[serde(alias = "STM")]
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

    /// Parse tier from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "stm" | "short-term" | "short_term" => Some(MemoryTier::Stm),
            "ltm" | "long-term" | "long_term" => Some(MemoryTier::Ltm),
            "episodic" | "trace" | "log" => Some(MemoryTier::Episodic),
            _ => None,
        }
    }

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

impl Default for MemoryTier {
    fn default() -> Self {
        MemoryTier::Stm
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
        assert_eq!(MemoryTier::from_str("stm"), Some(MemoryTier::Stm));
        assert_eq!(MemoryTier::from_str("STM"), Some(MemoryTier::Stm));
        assert_eq!(MemoryTier::from_str("ltm"), Some(MemoryTier::Ltm));
        assert_eq!(MemoryTier::from_str("episodic"), Some(MemoryTier::Episodic));
        assert_eq!(MemoryTier::from_str("unknown"), None);
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
