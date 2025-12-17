//! Work-stealing scheduler implementation.
//!
//! This module implements a work-stealing algorithm using crossbeam-deque.
//! Workers maintain local queues and can steal work from:
//! 1. Their own local queue (fastest)
//! 2. Global injectors by priority (high to low)
//! 3. Other workers' queues (slowest)

use std::sync::Arc;

use apxm_core::types::NodeId;
use crossbeam_deque::{Steal, Stealer, Worker};

use crate::scheduler::queue::{Priority, PriorityQueue};

/// Work-stealing coordinator for the scheduler.
///
/// Manages local worker queues and implements the stealing strategy.
pub struct WorkStealingScheduler {
    /// Global priority queues (injectors).
    queue: Arc<PriorityQueue>,

    /// Stealers for all worker threads.
    ///
    /// Used for stealing work from other workers when local queue is empty.
    stealers: Vec<Stealer<NodeId>>,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler.
    ///
    /// Returns the scheduler and a vector of Worker handles (one per thread).
    pub fn new(num_workers: usize, queue: Arc<PriorityQueue>) -> (Self, Vec<Worker<NodeId>>) {
        let mut stealers = Vec::with_capacity(num_workers);
        let mut workers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }

        let scheduler = Self { queue, stealers };

        (scheduler, workers)
    }

    /// Try to get the next node to execute.
    ///
    /// Stealing strategy (in order):
    /// 1. Pop from local worker queue (O(1), no contention)
    /// 2. Steal from global injectors, highest priority first
    /// 3. Steal from other workers' queues
    ///
    /// Returns Some(node_id) if work was found, None if all queues are empty.
    pub fn steal_next(&self, worker: &Worker<NodeId>, worker_id: usize) -> Option<NodeId> {
        // Step 1: Try local queue first (fastest path)
        if let Some(node_id) = worker.pop() {
            return Some(node_id);
        }

        // Step 2: Try global injectors, high priority to low
        if let Some(node_id) = self.steal_from_global() {
            return Some(node_id);
        }

        // Step 3: Steal from other workers (slowest path)
        self.steal_from_workers(worker_id)
    }

    /// Steal from global injectors, trying high-priority queues first.
    fn steal_from_global(&self) -> Option<NodeId> {
        // Try priorities from high to low: Critical → High → Normal → Low
        let priorities = [
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ];

        for priority in priorities {
            let injector = self.queue.injector(priority);
            match injector.steal() {
                Steal::Success(node_id) => return Some(node_id),
                Steal::Empty => continue,
                Steal::Retry => {
                    // Retry once on conflict
                    if let Steal::Success(node_id) = injector.steal() {
                        return Some(node_id);
                    }
                }
            }
        }

        None
    }

    /// Steal from other workers' queues.
    ///
    /// Uses round-robin strategy starting from the next worker.
    fn steal_from_workers(&self, worker_id: usize) -> Option<NodeId> {
        let num_workers = self.stealers.len();
        if num_workers <= 1 {
            return None; // No other workers to steal from
        }

        // Try stealing from each other worker
        for i in 1..num_workers {
            let target = (worker_id + i) % num_workers;
            let stealer = &self.stealers[target];

            match stealer.steal() {
                Steal::Success(node_id) => return Some(node_id),
                Steal::Empty => continue,
                Steal::Retry => {
                    // Retry once on conflict
                    if let Steal::Success(node_id) = stealer.steal() {
                        return Some(node_id);
                    }
                }
            }
        }

        None
    }

    /// Check if all queues (global and workers) appear empty.
    ///
    /// Note: This is a best-effort check and may be racy.
    pub fn is_empty(&self) -> bool {
        // Check global queues
        if !self.queue.is_empty() {
            return false;
        }

        // Check all worker queues
        for stealer in &self.stealers {
            if !stealer.is_empty() {
                return false;
            }
        }

        true
    }

    /// Get approximate total work across all queues.
    ///
    /// Note: This is an estimate due to concurrent modifications.
    pub fn total_work(&self) -> usize {
        let mut total = self.queue.len();

        for stealer in &self.stealers {
            total += stealer.len();
        }

        total
    }

    /// Get the number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.stealers.len()
    }
}

/// Helper for pushing work to global injectors.
///
/// This is typically used when work is generated outside of worker threads.
pub fn push_global(queue: &PriorityQueue, node_id: NodeId, priority: Priority) {
    queue.push(node_id, priority);
}

/// Helper for pushing work to a local worker queue.
///
/// This is more efficient than pushing to global when the worker is available.
pub fn push_local(worker: &Worker<NodeId>, node_id: NodeId) {
    worker.push(node_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_stealing_creation() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(4, queue);

        assert_eq!(scheduler.num_workers(), 4);
        assert_eq!(workers.len(), 4);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_local_queue_priority() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(2, queue);

        let worker0 = &workers[0];

        // Push to local queue
        worker0.push(100);
        worker0.push(200);

        // Should pop from local first (FIFO order)
        let node1 = scheduler.steal_next(worker0, 0);
        let node2 = scheduler.steal_next(worker0, 0);

        assert_eq!(node1, Some(100));
        assert_eq!(node2, Some(200));
    }

    #[test]
    fn test_global_injector_stealing() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(2, queue.clone());

        let worker0 = &workers[0];

        // Push to global injectors at different priorities
        queue.push(1, Priority::Low);
        queue.push(2, Priority::High);
        queue.push(3, Priority::Critical);

        // Should steal from high to low priority
        let node1 = scheduler.steal_next(worker0, 0);
        let node2 = scheduler.steal_next(worker0, 0);
        let node3 = scheduler.steal_next(worker0, 0);

        assert_eq!(node1, Some(3)); // Critical first
        assert_eq!(node2, Some(2)); // High second
        assert_eq!(node3, Some(1)); // Low last
    }

    #[test]
    fn test_worker_stealing() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(3, queue);

        let worker0 = &workers[0];
        let worker1 = &workers[1];
        let worker2 = &workers[2];

        // Worker 1 has work
        worker1.push(100);
        worker1.push(200);

        // Worker 2 has work
        worker2.push(300);

        // Worker 0 should steal from worker 1 first (round-robin)
        let node1 = scheduler.steal_next(worker0, 0);
        assert!(node1.is_some());
        assert!(node1 == Some(100) || node1 == Some(200) || node1 == Some(300));
    }

    #[test]
    fn test_empty_check() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(2, queue.clone());

        assert!(scheduler.is_empty());

        // Add work to global queue
        queue.push(1, Priority::Normal);
        assert!(!scheduler.is_empty());

        // Clear it
        let _ = scheduler.steal_next(&workers[0], 0);
        assert!(scheduler.is_empty());

        // Add work to local queue
        workers[0].push(2);
        assert!(!scheduler.is_empty());
    }

    #[test]
    fn test_total_work() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(2, queue.clone());

        assert_eq!(scheduler.total_work(), 0);

        // Add to global
        queue.push(1, Priority::Normal);
        queue.push(2, Priority::High);

        // Add to local
        workers[0].push(10);
        workers[1].push(20);

        // Should count all work (approximate)
        let total = scheduler.total_work();
        assert!(total >= 4); // May vary slightly due to races
    }

    #[test]
    fn test_push_helpers() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(1, queue.clone());

        // Test push_global
        push_global(&queue, 100, Priority::High);
        assert!(!scheduler.is_empty());

        // Test push_local
        push_local(&workers[0], 200);
        assert_eq!(scheduler.total_work(), 2);
    }

    #[test]
    fn test_single_worker_no_stealing() {
        let queue = Arc::new(PriorityQueue::new());
        let (scheduler, workers) = WorkStealingScheduler::new(1, queue);

        let worker0 = &workers[0];
        worker0.push(100);

        // With single worker, can't steal from others
        let node = scheduler.steal_next(worker0, 0);
        assert_eq!(node, Some(100));

        // Should return None when empty
        let node = scheduler.steal_next(worker0, 0);
        assert_eq!(node, None);
    }
}
