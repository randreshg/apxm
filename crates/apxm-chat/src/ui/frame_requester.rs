//! Frame draw scheduling utilities inspired by Codex's FrameRequester pattern.
//!
//! Widgets and background tasks can hold a [`FrameRequester`] handle to request
//! a redraw without tightly coupling to the TUI event loop. Requests are
//! coalesced so a burst of updates still results in a single draw call.

use std::time::{Duration, Instant};

use tokio::sync::{broadcast, mpsc};

/// Lightweight handle that schedules future draws on the TUI loop.
#[derive(Clone, Debug)]
pub struct FrameRequester {
    tx: mpsc::UnboundedSender<Instant>,
}

impl FrameRequester {
    /// Create a requester tied to the provided draw notification channel.
    pub fn new(draw_tx: broadcast::Sender<()>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(FrameScheduler::new(rx, draw_tx).run());
        Self { tx }
    }

    /// Request the UI to draw as soon as possible.
    pub fn schedule_frame(&self) {
        let _ = self.tx.send(Instant::now());
    }
}

struct FrameScheduler {
    rx: mpsc::UnboundedReceiver<Instant>,
    draw_tx: broadcast::Sender<()>,
}

impl FrameScheduler {
    fn new(rx: mpsc::UnboundedReceiver<Instant>, draw_tx: broadcast::Sender<()>) -> Self {
        Self { rx, draw_tx }
    }

    async fn run(mut self) {
        const FALLBACK: Duration = Duration::from_secs(30 * 24 * 60 * 60);
        let mut next_deadline: Option<Instant> = None;

        loop {
            let target = next_deadline.unwrap_or_else(|| Instant::now() + FALLBACK);
            let sleep = tokio::time::sleep_until(target.into());
            tokio::pin!(sleep);

            tokio::select! {
                maybe_time = self.rx.recv() => {
                    let Some(requested) = maybe_time else { break };
                    next_deadline = Some(next_deadline.map_or(requested, |existing| existing.min(requested)));
                }
                _ = &mut sleep => {
                    if next_deadline.take().is_some() {
                        let _ = self.draw_tx.send(());
                    }
                }
            }
        }
    }
}
