use std::sync::Arc;

use chrono::Utc;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::{
    sync::{Mutex, mpsc::UnboundedReceiver},
    task,
};

use crate::{
    ChatResponse, ChatSession, ChatStreamEvent,
    error::{ChatError, ChatResult},
    storage::Message,
};

use super::{frame_requester::FrameRequester, input::InputBuffer};

/// Application-level result of handling a key press.
pub enum AppAction {
    None,
    StartStreaming(UnboundedReceiver<ChatStreamEvent>),
    Quit,
}

// Removed: FocusTarget and ViewMode enums - no longer using sidebars/tabs
// Chat is always focused, specialist data viewed via slash commands

#[derive(Debug, Default, Clone)]
pub struct StreamingState {
    pub buffer: String,
    pub last_status: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct SpecializedState {
    pub plan: Option<apxm_core::Plan>,
    pub compiler_log: Vec<String>,
    pub execution_results: Vec<String>,
    pub dsl_code: Option<String>,
}

impl SpecializedState {
    pub fn apply_response(&mut self, response: &ChatResponse) {
        if let Some(plan) = &response.plan {
            self.plan = Some(plan.clone());
        }
        self.compiler_log = response.compiler_messages.clone();
        self.execution_results = response.execution_results.clone();
        self.dsl_code = response.dsl_code.clone();
    }
}

fn sanitize_compiler_message(msg: &str) -> String {
    // Minimal sanitizer for compiler messages shown in the UI:
    // - Trim leading/trailing whitespace
    // - Replace control characters with spaces
    // - Truncate extremely long messages to keep UI responsive
    let mut s = msg.trim().to_string();

    // Replace control characters (except newline) with spaces
    s = s
        .chars()
        .map(|c| {
            if (c as u32) < 0x20 && c != '\n' && c != '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();

    // Collapse repeated whitespace into single spaces for compactness
    let mut out = String::with_capacity(s.len());
    let mut last_was_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }

    // Truncate to a safe length (keep front and end summary)
    const MAX_LEN: usize = 1024;
    if out.len() > MAX_LEN {
        let prefix = &out[..800];
        let suffix = &out[out.len().saturating_sub(200)..];
        format!("{} ... {}", prefix.trim_end(), suffix.trim_start())
    } else {
        out.trim().to_string()
    }
}

pub struct AppState {
    session: Arc<Mutex<ChatSession>>,
    pub messages: Vec<Message>,
    // Removed: sessions list, selected_session, session_scroll (no left sidebar)
    // Removed: view_mode (no tabs)
    // Removed: focus (always on chat)
    // Removed: left_sidebar_visible, right_sidebar_visible (no sidebars)
    pub chat_scroll: u16,
    pub input: InputBuffer,
    pub streaming: Option<StreamingState>,
    pub specialized: SpecializedState, // For slash commands like /plan, /compiler
    pub is_processing: bool,
    pub status_line: Option<String>,
    pub request_frame: FrameRequester,
    pub pending_history_reload: bool,
    pub current_session_id: String,
    tick: u16,
}

impl AppState {
    pub async fn new(session: ChatSession, request_frame: FrameRequester) -> ChatResult<Self> {
        let current_session_id = session.id().to_string();
        let messages = session.messages().to_vec();

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            messages,
            chat_scroll: 0,
            input: InputBuffer::default(),
            streaming: None,
            specialized: SpecializedState::default(),
            is_processing: false,
            status_line: None,
            request_frame,
            pending_history_reload: false,
            current_session_id,
            tick: 0,
        })
    }

    pub fn set_status(&mut self, message: impl Into<String>) {
        self.status_line = Some(message.into());
        self.request_frame.schedule_frame();
    }

    pub async fn handle_key(&mut self, key: KeyEvent) -> ChatResult<AppAction> {
        // Quit on Ctrl+C, Ctrl+D, or Ctrl+Q
        if Self::is_quit_combo(&key) {
            return Ok(AppAction::Quit);
        }

        // Ctrl+L - Clear input buffer
        if key.code == KeyCode::Char('l') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.input.clear();
            self.request_frame.schedule_frame();
            return Ok(AppAction::None);
        }

        // Handle input keys
        match key.code {
            KeyCode::Enter => {
                // Shift+Enter or Alt+Enter - Insert newline (multi-line input)
                if key.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) {
                    self.input.insert_newline();
                    self.request_frame.schedule_frame();
                } else if self.input.as_str().trim_start().starts_with('/') {
                    // Slash command autocomplete: if suggestions exist, accept selected one
                    let suggestions = crate::commands::slash_suggestions(self.input.as_str());
                    if !suggestions.is_empty() {
                        let sel = self.input.slash_selected_index().unwrap_or(0);
                        let sel = sel.min(suggestions.len() - 1);
                        let chosen = suggestions[sel].usage;
                        self.input.set_text(&format!("{} ", chosen));
                        self.input.reset_slash_selection();
                        self.request_frame.schedule_frame();
                        return Ok(AppAction::None);
                    }
                    // No suggestions - submit as normal
                    match self.submit_input().await {
                        Ok(Some(rx)) => return Ok(AppAction::StartStreaming(rx)),
                        Ok(None) => {}
                        Err(ChatError::Command(cmd)) if cmd == "exit" => {
                            return Ok(AppAction::Quit);
                        }
                        Err(err) => {
                            self.set_status(format!("Error: {err}"));
                        }
                    }
                } else {
                    // Normal message submission
                    match self.submit_input().await {
                        Ok(Some(rx)) => return Ok(AppAction::StartStreaming(rx)),
                        Ok(None) => {}
                        Err(ChatError::Command(cmd)) if cmd == "exit" => {
                            return Ok(AppAction::Quit);
                        }
                        Err(err) => {
                            self.set_status(format!("Error: {err}"));
                        }
                    }
                }
            }
            KeyCode::Backspace => {
                self.input.backspace();
                self.request_frame.schedule_frame();
            }
            KeyCode::Delete => {
                self.input.delete();
                self.request_frame.schedule_frame();
            }
            KeyCode::Left => {
                self.input.move_left();
                self.request_frame.schedule_frame();
            }
            KeyCode::Right => {
                self.input.move_right();
                self.request_frame.schedule_frame();
            }
            KeyCode::Up => {
                // Slash command suggestion navigation
                if self.input.as_str().trim_start().starts_with('/') {
                    let suggestions = crate::commands::slash_suggestions(self.input.as_str());
                    if !suggestions.is_empty() {
                        self.input.slash_select_prev(suggestions.len());
                        self.request_frame.schedule_frame();
                    } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_chat(-1);
                    } else {
                        self.input.move_up();
                        self.request_frame.schedule_frame();
                    }
                } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.scroll_chat(-1);
                } else {
                    self.input.move_up();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::Down => {
                // Slash command suggestion navigation
                if self.input.as_str().trim_start().starts_with('/') {
                    let suggestions = crate::commands::slash_suggestions(self.input.as_str());
                    if !suggestions.is_empty() {
                        self.input.slash_select_next(suggestions.len());
                        self.request_frame.schedule_frame();
                    } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_chat(1);
                    } else {
                        self.input.move_down();
                        self.request_frame.schedule_frame();
                    }
                } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.scroll_chat(1);
                } else {
                    self.input.move_down();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::Home => {
                self.input.move_to_start_of_line();
                self.request_frame.schedule_frame();
            }
            KeyCode::End => {
                self.input.move_to_end_of_line();
                self.request_frame.schedule_frame();
            }
            KeyCode::Char(ch) => {
                if !key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.input.insert_char(ch);
                    self.request_frame.schedule_frame();
                }
            }
            _ => {}
        }

        Ok(AppAction::None)
    }

    pub fn handle_stream_event(&mut self, event: ChatStreamEvent) {
        match event {
            ChatStreamEvent::Status(msg) => {
                if let Some(stream) = self.streaming.as_mut() {
                    stream.last_status = Some(msg.clone());
                }
                self.status_line = Some(msg);
            }
            ChatStreamEvent::Plan(plan) => {
                self.specialized.plan = Some(plan.clone());
                // AAM state will be populated from execution results, not plan
            }
            ChatStreamEvent::CompilerMessage(msg) => {
                // Sanitize compiler message first.
                let clean = sanitize_compiler_message(&msg);

                // Truncate long lines and limit number of lines per message to avoid
                // overlapping the chat input or overflowing the UI.
                const MAX_LINE_LEN: usize = 120;
                const MAX_LINES_PER_MSG: usize = 20;

                let mut truncated = String::new();
                for (i, line) in clean.lines().enumerate() {
                    if i >= MAX_LINES_PER_MSG {
                        truncated.push_str("... (truncated)\n");
                        break;
                    }
                    if line.len() > MAX_LINE_LEN {
                        // Truncate long lines while preserving start and ellipsis.
                        truncated.push_str(&format!("{}...\n", &line[..MAX_LINE_LEN]));
                    } else {
                        truncated.push_str(line);
                        truncated.push('\n');
                    }
                }

                let entry = truncated.trim_end().to_string();
                self.specialized.compiler_log.push(entry);

                // Limit total log entries to prevent memory growth and excessive UI height.
                const MAX_LOG_ENTRIES: usize = 100;
                if self.specialized.compiler_log.len() > MAX_LOG_ENTRIES {
                    self.specialized
                        .compiler_log
                        .drain(0..(self.specialized.compiler_log.len() - MAX_LOG_ENTRIES));
                }
            }
            ChatStreamEvent::Chunk(chunk) => {
                let stream = self.streaming.get_or_insert_with(StreamingState::default);
                stream.buffer.push_str(&chunk);
            }
            ChatStreamEvent::Finished(response) => {
                self.specialized.apply_response(&response);
                self.streaming = None;
                self.is_processing = false;
                self.pending_history_reload = true;
                self.set_status("Response ready");
            }
            ChatStreamEvent::Error(msg) => {
                // Limit exposure of error details in the main UI and surface only a short,
                // sanitized summary. Push the full diagnostic (sanitized) into the
                // compiler log where users can inspect it in the diagnostics panel.
                self.streaming = None;
                self.is_processing = false;

                // Full details go to compiler_log (sanitized for UI safety).
                self.specialized
                    .compiler_log
                    .push(format!("✗ Full error: {}", sanitize_compiler_message(&msg)));

                // Short summary shown in status line to avoid leaking sensitive details.
                let summary = msg.lines().next().unwrap_or("").trim();
                let display = if summary.is_empty() {
                    "Request failed".to_string()
                } else if summary.len() > 120 {
                    format!("{}...", &summary[..120])
                } else {
                    summary.to_string()
                };

                self.set_status(format!("Error: {} (see diagnostics)", display));
            }
        }
        self.request_frame.schedule_frame();
    }

    pub fn on_tick(&mut self) -> bool {
        self.tick = self.tick.wrapping_add(1);
        self.streaming.is_some() && self.tick.is_multiple_of(4)
    }

    pub fn tick_count(&self) -> u16 {
        self.tick
    }

    pub fn take_history_reload(&mut self) -> bool {
        if self.pending_history_reload {
            self.pending_history_reload = false;
            true
        } else {
            false
        }
    }

    pub async fn reload_history(&mut self) -> ChatResult<()> {
        let session = self.session.lock().await;
        self.current_session_id = session.id().to_string();
        self.messages = session.messages().to_vec();
        Ok(())
    }

    async fn submit_input(&mut self) -> ChatResult<Option<UnboundedReceiver<ChatStreamEvent>>> {
        if self.is_processing {
            self.set_status("Assistant is still thinking...");
            return Ok(None);
        }

        if self.input.is_empty() {
            return Ok(None);
        }

        let text = self.input.take();

        if text.trim_start().starts_with('/') {
            let output = {
                let mut session = self.session.lock().await;
                session.process_input(&text).await
            };

            match output {
                Ok(response) => {
                    self.append_system_message(response);
                    self.request_frame.schedule_frame();
                }
                Err(err) => return Err(err),
            }
            return Ok(None);
        }

        self.append_user_message(&text);
        self.streaming = Some(StreamingState::default());
        self.is_processing = true;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let session = self.session.clone();
        let frame = self.request_frame.clone();
        let prompt = text.clone();
        task::spawn_local(async move {
            let mut guard = session.lock().await;
            if let Err(err) = guard
                .process_input_streaming(&prompt, Some(tx.clone()))
                .await
            {
                let _ = tx.send(ChatStreamEvent::Error(err.to_string()));
            }
            frame.schedule_frame();
        });

        Ok(Some(rx))
    }

    fn append_user_message(&mut self, text: &str) {
        self.messages.push(Message {
            role: "user".to_string(),
            content: text.to_string(),
            timestamp: Utc::now(),
        });
        self.request_frame.schedule_frame();
    }

    fn append_system_message(&mut self, text: String) {
        self.messages.push(Message {
            role: "system".to_string(),
            content: text,
            timestamp: Utc::now(),
        });
        self.request_frame.schedule_frame();
    }

    fn scroll_chat(&mut self, delta: i16) {
        if delta.is_negative() {
            let amount = delta.unsigned_abs();
            self.chat_scroll = self.chat_scroll.saturating_sub(amount);
        } else {
            self.chat_scroll = self.chat_scroll.saturating_add(delta as u16);
        }
        self.request_frame.schedule_frame();
    }

    fn is_quit_combo(key: &KeyEvent) -> bool {
        key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(
                key.code,
                KeyCode::Char('c') | KeyCode::Char('d') | KeyCode::Char('q')
            )
    }
}
