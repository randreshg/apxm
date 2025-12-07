//! Memory space identifiers.
//!
//! The memory system is divided into short-term memory (STM), long-term memory
//! (LTM), and episodic storage. This enum provides a type-safe way to refer to
//! each tier.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifies a specific memory tier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MemorySpace {
    /// Short-term memory (fast, volatile).
    Stm,
    /// Long-term memory (persistent, searchable).
    Ltm,
    /// Episodic memory (append-only traces).
    Episodic,
}

impl MemorySpace {
    /// Returns true when the space is STM.
    pub fn is_stm(&self) -> bool {
        matches!(self, MemorySpace::Stm)
    }

    /// Returns true when the space is LTM.
    pub fn is_ltm(&self) -> bool {
        matches!(self, MemorySpace::Ltm)
    }

    /// Returns true when the space is Episodic.
    pub fn is_episodic(&self) -> bool {
        matches!(self, MemorySpace::Episodic)
    }
}

impl fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemorySpace::Stm => write!(f, "STM"),
            MemorySpace::Ltm => write!(f, "LTM"),
            MemorySpace::Episodic => write!(f, "EPISODIC"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_and_helpers() {
        assert_eq!(MemorySpace::Stm.to_string(), "STM");
        assert!(MemorySpace::Stm.is_stm());
        assert!(!MemorySpace::Stm.is_ltm());
        assert!(!MemorySpace::Stm.is_episodic());

        assert_eq!(MemorySpace::Ltm.to_string(), "LTM");
        assert!(MemorySpace::Ltm.is_ltm());
        assert!(!MemorySpace::Ltm.is_stm());

        assert_eq!(MemorySpace::Episodic.to_string(), "EPISODIC");
        assert!(MemorySpace::Episodic.is_episodic());
    }

    #[test]
    fn test_serialization_round_trip() {
        let json = serde_json::to_string(&MemorySpace::Stm).expect("serialize");
        assert_eq!(json, "\"STM\"");

        let deserialized: MemorySpace =
            serde_json::from_str("\"EPISODIC\"").expect("deserialize episodic");
        assert_eq!(deserialized, MemorySpace::Episodic);
    }
}
