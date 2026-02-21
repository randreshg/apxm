//! Per-session execution lane guard.
//!
//! Ensures requests for the same session execute serially while allowing
//! different sessions to run concurrently.

use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{Mutex, OwnedMutexGuard};

/// Session lane coordinator.
#[derive(Default)]
pub struct SessionLaneGuard {
    lanes: DashMap<String, Arc<Mutex<()>>>,
}

impl SessionLaneGuard {
    /// Create an empty lane guard.
    pub fn new() -> Self {
        Self {
            lanes: DashMap::new(),
        }
    }

    /// Acquire the lane for a session.
    pub async fn acquire(&self, session_id: &str) -> SessionLanePermit {
        let lane = self
            .lanes
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let guard = lane.lock_owned().await;
        SessionLanePermit {
            _guard: guard,
            session_id: session_id.to_string(),
        }
    }
}

/// Held while a session lane lock is active.
pub struct SessionLanePermit {
    _guard: OwnedMutexGuard<()>,
    #[allow(dead_code)]
    session_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::{Duration, Instant, sleep};

    #[tokio::test]
    async fn same_session_is_serialized() {
        let lanes = Arc::new(SessionLaneGuard::new());
        let active = Arc::new(AtomicUsize::new(0));
        let peaks = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..3 {
            let lanes = Arc::clone(&lanes);
            let active = Arc::clone(&active);
            let peaks = Arc::clone(&peaks);
            handles.push(tokio::spawn(async move {
                let _permit = lanes.acquire("same").await;
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peaks.fetch_max(now, Ordering::SeqCst);
                sleep(Duration::from_millis(20)).await;
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for handle in handles {
            handle.await.expect("task should complete");
        }
        assert_eq!(peaks.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn different_sessions_can_run_in_parallel() {
        let lanes = Arc::new(SessionLaneGuard::new());
        let start = Instant::now();
        let la = Arc::clone(&lanes);
        let lb = Arc::clone(&lanes);
        let t1 = tokio::spawn(async move {
            let _permit = la.acquire("x").await;
            sleep(Duration::from_millis(80)).await;
        });
        let t2 = tokio::spawn(async move {
            let _permit = lb.acquire("y").await;
            sleep(Duration::from_millis(80)).await;
        });
        let _ = t1.await;
        let _ = t2.await;
        assert!(start.elapsed() < Duration::from_millis(150));
    }
}
