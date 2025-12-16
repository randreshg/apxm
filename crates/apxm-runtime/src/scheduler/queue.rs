//! Priority queue for ready operations.
//!
//! This module provides a 4-level priority queue for scheduling operations.
//! Operations are enqueued by priority and dequeued FIFO within each priority level.

use std::sync::Arc;

use apxm_core::types::NodeId;
use crossbeam_deque::Injector;

/// Priority levels for operations.
///
/// Higher numbers = higher priority (executed first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Priority {
    /// Lowest priority (0-29).
    Low = 0,
    /// Normal priority (30-59).
    Normal = 1,
    /// High priority (60-89).
    High = 2,
    /// Critical priority (90-100).
    Critical = 3,
}

impl Priority {
    /// Convert a numeric priority (0-100) to a Priority level.
    pub fn from_u8(priority: u8) -> Self {
        match priority {
            x if x >= 90 => Priority::Critical,
            x if x >= 60 => Priority::High,
            x if x >= 30 => Priority::Normal,
            _ => Priority::Low,
        }
    }

    /// Get the priority as a usize index (for array indexing).
    #[inline]
    pub fn as_index(self) -> usize {
        self as usize
    }

    /// Number of priority levels.
    pub const COUNT: usize = 4;
}

/// Priority queue for ready operations.
///
/// Maintains 4 separate FIFO queues (one per priority level).
/// Operations are dequeued from the highest-priority non-empty queue.
pub struct PriorityQueue {
    /// Injectors for each priority level.
    ///
    /// Index 0 = Low, 1 = Normal, 2 = High, 3 = Critical
    injectors: [Arc<Injector<NodeId>>; Priority::COUNT],
}

impl PriorityQueue {
    /// Create a new priority queue.
    pub fn new() -> Self {
        Self {
            injectors: [
                Arc::new(Injector::new()),
                Arc::new(Injector::new()),
                Arc::new(Injector::new()),
                Arc::new(Injector::new()),
            ],
        }
    }

    /// Push an operation to the queue at the given priority.
    #[inline]
    pub fn push(&self, node_id: NodeId, priority: Priority) {
        self.injectors[priority.as_index()].push(node_id);
    }

    /// Get a reference to the injector at the given priority level.
    ///
    /// Useful for work-stealing.
    #[inline]
    pub fn injector(&self, priority: Priority) -> &Arc<Injector<NodeId>> {
        &self.injectors[priority.as_index()]
    }

    /// Get references to all injectors.
    ///
    /// Returned in order: [Low, Normal, High, Critical]
    pub fn injectors(&self) -> &[Arc<Injector<NodeId>>; Priority::COUNT] {
        &self.injectors
    }

    /// Check if all queues are empty.
    pub fn is_empty(&self) -> bool {
        self.injectors.iter().all(|inj| inj.is_empty())
    }

    /// Get approximate total length across all queues.
    ///
    /// Note: This is an estimate due to concurrent modifications.
    pub fn len(&self) -> usize {
        self.injectors.iter().map(|inj| inj.len()).sum()
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_from_u8() {
        assert_eq!(Priority::from_u8(0), Priority::Low);
        assert_eq!(Priority::from_u8(29), Priority::Low);
        assert_eq!(Priority::from_u8(30), Priority::Normal);
        assert_eq!(Priority::from_u8(59), Priority::Normal);
        assert_eq!(Priority::from_u8(60), Priority::High);
        assert_eq!(Priority::from_u8(89), Priority::High);
        assert_eq!(Priority::from_u8(90), Priority::Critical);
        assert_eq!(Priority::from_u8(100), Priority::Critical);
    }

    #[test]
    fn test_priority_as_index() {
        assert_eq!(Priority::Low.as_index(), 0);
        assert_eq!(Priority::Normal.as_index(), 1);
        assert_eq!(Priority::High.as_index(), 2);
        assert_eq!(Priority::Critical.as_index(), 3);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_queue_creation() {
        let queue = PriorityQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_queue_push() {
        let queue = PriorityQueue::new();

        queue.push(1, Priority::Low);
        queue.push(2, Priority::High);
        queue.push(3, Priority::Critical);

        assert!(!queue.is_empty());
        // Note: len() is approximate, but should be at least 3
        assert!(queue.len() >= 3);
    }

    #[test]
    fn test_queue_injector_access() {
        let queue = PriorityQueue::new();

        let injector = queue.injector(Priority::High);
        injector.push(42);

        assert!(!queue.is_empty());
    }

    #[test]
    fn test_queue_injectors_access() {
        let queue = PriorityQueue::new();

        let injectors = queue.injectors();
        assert_eq!(injectors.len(), Priority::COUNT);

        injectors[Priority::Critical.as_index()].push(100);
        assert!(!queue.is_empty());
    }
}
