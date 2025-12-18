use std::sync::Arc;

use chrono::Utc;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::{
    sync::{Mutex, mpsc::UnboundedReceiver},
    task,
};

use crate::{
    ChatResponse, ChatSession, ChatStreamEvent,
    config::ChatConfig,
    error::{ChatError, ChatResult},
    storage::{Message, SessionInfo},
};

use super::{frame_requester::FrameRequester, input::InputBuffer};

/// Application-level result of handling a key press.
pub enum AppAction {
    None,
    StartStreaming(UnboundedReceiver<ChatStreamEvent>),
    Quit,
}

/// Focus targets for keyboard navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusTarget {
    Sessions,
    Chat,
    Specialist,
}

impl FocusTarget {
    pub fn next(self) -> Self {
        match self {
            FocusTarget::Sessions => FocusTarget::Chat,
            FocusTarget::Chat => FocusTarget::Specialist,
            FocusTarget::Specialist => FocusTarget::Sessions,
        }
    }
}

/// Specialized views for the right sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Minimal,
    Compiler,
    Dag,
    Aam,
    Full,
}

#[derive(Debug, Clone, Default)]
pub struct AamState {
    pub goals: Vec<String>,
    pub beliefs: Vec<String>,
    pub episodic_memory: Vec<String>,
    pub capabilities: Vec<String>,
}

impl AamState {
    fn from_plan(plan: &crate::translator::OuterPlan) -> Self {
        let goals = vec![plan.result.clone()];
        let beliefs = plan
            .plan
            .iter()
            .map(|step| format!("Dependency: {}", step.description))
            .collect();
        let capabilities = plan
            .plan
            .iter()
            .map(|step| {
                format!(
                    "Capability: {} (priority {})",
                    step.description, step.priority
                )
            })
            .collect();
        let episodic_memory = vec!["Awaiting execution traces".to_string()];
        Self {
            goals,
            beliefs,
            episodic_memory,
            capabilities,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct StreamingState {
    pub buffer: String,
    pub last_status: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct SpecializedState {
    pub plan: Option<crate::translator::OuterPlan>,
    pub compiler_log: Vec<String>,
    pub execution_results: Vec<String>,
    pub aam: AamState,
    pub dsl_code: Option<String>,
}

impl SpecializedState {
    pub fn apply_response(&mut self, response: &ChatResponse) {
        if let Some(plan) = &response.outer_plan {
            self.plan = Some(plan.clone());
            self.aam = AamState::from_plan(plan);
        }
        self.compiler_log = response.compiler_messages.clone();
        self.execution_results = response.execution_results.clone();
        self.dsl_code = response.dsl_code.clone();
    }
}

pub struct AppState {
    session: Arc<Mutex<ChatSession>>,
    pub config: ChatConfig,
    pub messages: Vec<Message>,
    pub sessions: Vec<SessionInfo>,
    pub selected_session: usize,
    pub view_mode: ViewMode,
    pub focus: FocusTarget,
    pub left_sidebar_visible: bool,
    pub right_sidebar_visible: bool,
    pub chat_scroll: u16,
    pub session_scroll: usize,
    pub input: InputBuffer,
    pub streaming: Option<StreamingState>,
    pub specialized: SpecializedState,
    pub is_processing: bool,
    pub status_line: Option<String>,
    pub request_frame: FrameRequester,
    pub pending_history_reload: bool,
    pub current_session_id: String,
    tick: u16,
}

impl AppState {
    pub async fn new(session: ChatSession, request_frame: FrameRequester) -> ChatResult<Self> {
        let config = session.config().clone();
        let current_session_id = session.id().to_string();
        let messages = session.messages().to_vec();
        let mut sessions = Vec::new();
        let mut status_line = None;
        match ChatSession::list_sessions(&config.session_storage_path).await {
            Ok(list) => sessions = list,
            Err(err) => status_line = Some(format!("Failed to list sessions: {err}")),
        }
        let selected_session = sessions
            .iter()
            .position(|info| info.id == current_session_id)
            .unwrap_or(0);

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            config,
            messages,
            sessions,
            selected_session,
            view_mode: ViewMode::Full,
            focus: FocusTarget::Chat,
            left_sidebar_visible: true,
            right_sidebar_visible: true,
            chat_scroll: 0,
            session_scroll: selected_session.saturating_sub(1),
            input: InputBuffer::default(),
            streaming: None,
            specialized: SpecializedState::default(),
            is_processing: false,
            status_line,
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
        if Self::is_quit_combo(&key) {
            return Ok(AppAction::Quit);
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('b') => {
                    self.toggle_left_sidebar();
                    return Ok(AppAction::None);
                }
                KeyCode::Char('p') => {
                    self.toggle_right_sidebar();
                    return Ok(AppAction::None);
                }
                KeyCode::Char('n') => {
                    self.create_new_session().await?;
                    return Ok(AppAction::None);
                }
                KeyCode::Char('r') => {
                    self.refresh_sessions().await?;
                    return Ok(AppAction::None);
                }
                KeyCode::Char('l') => {
                    self.input.clear();
                    self.request_frame.schedule_frame();
                    return Ok(AppAction::None);
                }
                KeyCode::Char('1') => {
                    self.view_mode = ViewMode::Minimal;
                    self.request_frame.schedule_frame();
                    return Ok(AppAction::None);
                }
                KeyCode::Char('2') => {
                    self.view_mode = ViewMode::Compiler;
                    self.request_frame.schedule_frame();
                    return Ok(AppAction::None);
                }
                KeyCode::Char('3') => {
                    self.view_mode = ViewMode::Dag;
                    self.request_frame.schedule_frame();
                    return Ok(AppAction::None);
                }
                KeyCode::Char('4') => {
                    self.view_mode = ViewMode::Aam;
                    self.request_frame.schedule_frame();
                    return Ok(AppAction::None);
                }
                KeyCode::Char('5') => {
                    self.view_mode = ViewMode::Full;
                    self.request_frame.schedule_frame();
                    return Ok(AppAction::None);
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Tab => {
                self.focus = self.focus.next();
                self.request_frame.schedule_frame();
            }
            KeyCode::Enter => {
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                {
                    self.input.insert_newline();
                    self.request_frame.schedule_frame();
                } else if matches!(self.focus, FocusTarget::Sessions) {
                    self.load_selected_session().await?;
                } else {
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
                if matches!(self.focus, FocusTarget::Chat) {
                    self.input.backspace();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::Delete => {
                if matches!(self.focus, FocusTarget::Chat) {
                    self.input.delete();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::Left => {
                if matches!(self.focus, FocusTarget::Chat) {
                    self.input.move_left();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::Right => {
                if matches!(self.focus, FocusTarget::Chat) {
                    self.input.move_right();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::Up => match self.focus {
                FocusTarget::Sessions => {
                    self.move_session_selection(-1);
                }
                FocusTarget::Chat => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_chat(-1);
                    } else {
                        self.input.move_up();
                        self.request_frame.schedule_frame();
                    }
                }
                FocusTarget::Specialist => {
                    self.scroll_chat(-1);
                }
            },
            KeyCode::Down => match self.focus {
                FocusTarget::Sessions => {
                    self.move_session_selection(1);
                }
                FocusTarget::Chat => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_chat(1);
                    } else {
                        self.input.move_down();
                        self.request_frame.schedule_frame();
                    }
                }
                FocusTarget::Specialist => {
                    self.scroll_chat(1);
                }
            },
            KeyCode::Home => {
                if matches!(self.focus, FocusTarget::Chat) {
                    self.input.move_to_start_of_line();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::End => {
                if matches!(self.focus, FocusTarget::Chat) {
                    self.input.move_to_end_of_line();
                    self.request_frame.schedule_frame();
                }
            }
            KeyCode::Char(ch) => {
                if matches!(self.focus, FocusTarget::Chat)
                    && !key.modifiers.contains(KeyModifiers::CONTROL)
                {
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
                self.specialized.aam = AamState::from_plan(&plan);
            }
            ChatStreamEvent::CompilerMessage(msg) => {
                self.specialized.compiler_log.push(msg);
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
                self.streaming = None;
                self.is_processing = false;
                self.set_status(format!("Error: {msg}"));
            }
        }
        self.request_frame.schedule_frame();
    }

    pub fn on_tick(&mut self) -> bool {
        self.tick = self.tick.wrapping_add(1);
        self.streaming.is_some() && self.tick % 4 == 0
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

    pub async fn refresh_sessions(&mut self) -> ChatResult<()> {
        match ChatSession::list_sessions(&self.config.session_storage_path).await {
            Ok(list) => self.sessions = list,
            Err(err) => self.set_status(format!("Failed to list sessions: {err}")),
        }
        if let Some(idx) = self
            .sessions
            .iter()
            .position(|info| info.id == self.current_session_id)
        {
            self.selected_session = idx;
        }
        self.request_frame.schedule_frame();
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

    async fn create_new_session(&mut self) -> ChatResult<()> {
        if self.is_processing {
            self.set_status("Finish the current request before starting a new session");
            return Ok(());
        }

        let session = ChatSession::new(self.config.clone()).await?;
        self.current_session_id = session.id().to_string();
        self.messages = session.messages().to_vec();
        self.session = Arc::new(Mutex::new(session));
        self.refresh_sessions().await?;
        self.set_status("Started new session");
        Ok(())
    }

    async fn load_selected_session(&mut self) -> ChatResult<()> {
        if self.sessions.is_empty() {
            return Ok(());
        }
        let idx = self.selected_session.min(self.sessions.len() - 1);
        let id = self.sessions[idx].id.clone();
        self.load_session(&id).await
    }

    async fn load_session(&mut self, id: &str) -> ChatResult<()> {
        if self.is_processing {
            self.set_status("Please wait for the current response to finish");
            return Ok(());
        }
        let session = ChatSession::load(id, self.config.clone()).await?;
        self.current_session_id = session.id().to_string();
        self.messages = session.messages().to_vec();
        self.session = Arc::new(Mutex::new(session));
        self.refresh_sessions().await?;
        self.set_status(format!("Loaded session {}", &id[..id.len().min(8)]));
        Ok(())
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

    fn toggle_left_sidebar(&mut self) {
        self.left_sidebar_visible = !self.left_sidebar_visible;
        self.request_frame.schedule_frame();
    }

    fn toggle_right_sidebar(&mut self) {
        self.right_sidebar_visible = !self.right_sidebar_visible;
        self.request_frame.schedule_frame();
    }

    fn move_session_selection(&mut self, delta: i32) {
        if self.sessions.is_empty() {
            return;
        }
        let len = self.sessions.len() as i32;
        let mut new_idx = self.selected_session as i32 + delta;
        new_idx = new_idx.clamp(0, len - 1);
        self.selected_session = new_idx as usize;
        if self.selected_session < self.session_scroll {
            self.session_scroll = self.selected_session;
        } else if self.selected_session > self.session_scroll + 5 {
            self.session_scroll = self.selected_session - 5;
        }
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
