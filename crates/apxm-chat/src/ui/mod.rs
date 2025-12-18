//! Terminal UI with a three-pane layout inspired by Codex/Goose chat apps.

mod app;
mod frame_requester;
mod input;
mod render;

use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::{
    event::{Event as CrosstermEvent, EventStream},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::select;
use tokio::sync::{broadcast, mpsc::UnboundedReceiver};
use tokio::task::LocalSet;
use tokio::time::interval;

use crate::{ChatSession, ChatStreamEvent, error::ChatResult};

use app::{AppAction, AppState, FocusTarget};
use frame_requester::FrameRequester;
use render::render;

/// Launch the interactive chat UI.
pub async fn run_chat_ui(session: ChatSession) -> ChatResult<()> {
    let mut terminal = init_terminal()?;
    let local = LocalSet::new();
    let result = local
        .run_until(run_event_loop(&mut terminal, session))
        .await;
    restore_terminal(&mut terminal)?;
    result
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    session: ChatSession,
) -> ChatResult<()> {
    let (draw_tx, _) = broadcast::channel(16);
    let mut draw_rx = draw_tx.subscribe();
    let frame_requester = FrameRequester::new(draw_tx.clone());
    let mut app = AppState::new(session, frame_requester.clone()).await?;

    let mut reader = EventStream::new();
    let mut ticker = interval(Duration::from_millis(120));
    let mut stream_rx: Option<UnboundedReceiver<ChatStreamEvent>> = None;

    frame_requester.schedule_frame();

    loop {
        select! {
            _ = ticker.tick() => {
                if app.on_tick() {
                    frame_requester.schedule_frame();
                }
            }
            maybe_event = reader.next() => {
                if let Some(event) = maybe_event {
                    let event = event?;
                    match process_event(event, &mut app).await? {
                        EventOutcome::Continue => {}
                        EventOutcome::Quit => break,
                        EventOutcome::StartStream(rx) => stream_rx = Some(rx),
                    }
                } else {
                    break;
                }
            }
            res = draw_rx.recv() => {
                if res.is_ok() {
                    terminal.draw(|f| render(f, &app))?;
                } else {
                    break;
                }
            }
            maybe_stream = async {
                if let Some(rx) = stream_rx.as_mut() {
                    rx.recv().await
                } else {
                    None
                }
            }, if stream_rx.is_some() => {
                if let Some(event) = maybe_stream {
                    app.handle_stream_event(event);
                } else {
                    stream_rx = None;
                }
            }
        }

        if app.take_history_reload() {
            app.reload_history().await?;
        }
    }

    Ok(())
}

enum EventOutcome {
    Continue,
    Quit,
    StartStream(UnboundedReceiver<ChatStreamEvent>),
}

async fn process_event(event: CrosstermEvent, app: &mut AppState) -> ChatResult<EventOutcome> {
    match event {
        CrosstermEvent::Key(key) => match app.handle_key(key).await? {
            AppAction::None => Ok(EventOutcome::Continue),
            AppAction::Quit => Ok(EventOutcome::Quit),
            AppAction::StartStreaming(rx) => Ok(EventOutcome::StartStream(rx)),
        },
        CrosstermEvent::Paste(data) => {
            if matches!(app.focus, FocusTarget::Chat) {
                app.input.insert_str(&data);
                app.request_frame.schedule_frame();
            }
            Ok(EventOutcome::Continue)
        }
        CrosstermEvent::Resize(_, _) => {
            app.request_frame.schedule_frame();
            Ok(EventOutcome::Continue)
        }
        _ => Ok(EventOutcome::Continue),
    }
}

fn init_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ui_module_compiles() {
        // smoke test for cfg(test)
    }
}
