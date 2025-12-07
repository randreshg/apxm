//! Memory system type definitions for APXM.
//!
//! This module exposes the memory tiers (STM, LTM, Episodic) and their
//! associated types used across the system. Implementations live in runtime;
//! here we only define the shared data structures and traits.

pub mod episodic;
pub mod ltm;
pub mod space;
pub mod stm;

pub use episodic::{EpisodicEntry, EpisodicQuery};
pub use ltm::{LTMBackend, LTMQuery, LTMResult};
pub use space::MemorySpace;
pub use stm::{STMConfig, STMEntry};
