//! Chat session management with execution

use crate::{
    commands::SlashCommand,
    config::ChatConfig,
    error::{ChatError, ChatResult},
    storage::{Message, SessionInfo, SessionStorage},
    translator::{OuterPlan, Translator},
};
use apxm_linker::{Linker, LinkerConfig, error::LinkerError};
use chrono::Utc;
use std::path::Path;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

/// Chat session with execution capabilities
pub struct ChatSession {
    /// Session ID
    pub id: String,

    /// Storage backend
    storage: SessionStorage,

    /// Linker for DSL compilation and execution
    linker: Linker,

    /// Translator for NL → DSL
    translator: Translator,

    /// Configuration
    config: ChatConfig,

    /// In-memory message history
    messages: Vec<Message>,

    /// System prompt
    system_prompt: String,
}

/// Rich response returned from the chat pipeline.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Final assistant message content
    pub content: String,
    /// Outer plan emitted by the translator
    pub outer_plan: Option<OuterPlan>,
    /// AIS/DSL source that was executed
    pub dsl_code: Option<String>,
    /// Execution outputs rendered to strings
    pub execution_results: Vec<String>,
    /// Compiler/linker diagnostics gathered during execution
    pub compiler_messages: Vec<String>,
}

impl ChatResponse {
    fn from_text(content: String) -> Self {
        Self {
            content,
            outer_plan: None,
            dsl_code: None,
            execution_results: Vec::new(),
            compiler_messages: Vec::new(),
        }
    }
}

/// Events emitted while a response is being streamed to the UI.
#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    /// High-level status message for progress updates
    Status(String),
    /// Updated planning graph emitted by the translator
    Plan(OuterPlan),
    /// Compiler or linker message for the diagnostics pane
    CompilerMessage(String),
    /// Chunk of assistant output text for incremental rendering
    Chunk(String),
    /// Response completed successfully with diagnostic payload
    Finished(ChatResponse),
    /// Non-recoverable error while generating the response
    Error(String),
}

impl ChatSession {
    /// Create a new chat session
    pub async fn new(config: ChatConfig) -> ChatResult<Self> {
        let id = Uuid::now_v7().to_string();

        // Create storage
        let storage_path = config.session_storage_path.join("sessions.db");
        let storage = SessionStorage::new(&storage_path).await?;

        // Create linker
        let linker_config = LinkerConfig::from_apxm_config(config.apxm_config.clone());
        let linker = Linker::new(linker_config).await?;

        // Create translator with LLM registry from linker
        let registry = linker.runtime_llm_registry();
        let planning_model = config
            .planning_model
            .clone()
            .unwrap_or_else(|| config.default_model.clone());
        let translator =
            Translator::new(registry, planning_model, Some(config.default_model.clone()));

        let system_prompt = config.system_prompt.clone();
        let messages = Vec::new();

        // Create session info
        let _session_info = SessionInfo {
            id: id.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            message_count: 0,
            model: config.default_model.clone(),
        };

        storage.create_session(&_session_info).await?;

        Ok(Self {
            id,
            storage,
            linker,
            translator,
            config,
            messages,
            system_prompt,
        })
    }

    /// Load existing session
    pub async fn load(session_id: &str, config: ChatConfig) -> ChatResult<Self> {
        // Create storage
        let storage_path = config.session_storage_path.join("sessions.db");
        let storage = SessionStorage::new(&storage_path).await?;

        // Get session info
        let _session_info = storage
            .get_session(session_id)
            .await?
            .ok_or_else(|| ChatError::Config(format!("Session not found: {}", session_id)))?;

        // Load messages
        let messages = storage.get_messages(session_id).await?;

        // Create linker
        let linker_config = LinkerConfig::from_apxm_config(config.apxm_config.clone());
        let linker = Linker::new(linker_config).await?;

        // Create translator
        let registry = linker.runtime_llm_registry();
        let planning_model = config
            .planning_model
            .clone()
            .unwrap_or_else(|| config.default_model.clone());
        let translator =
            Translator::new(registry, planning_model, Some(config.default_model.clone()));

        let system_prompt = config.system_prompt.clone();

        Ok(Self {
            id: session_id.to_string(),
            storage,
            linker,
            translator,
            config,
            messages,
            system_prompt,
        })
    }

    /// Process user input (checks for slash commands)
    pub async fn process_input(&mut self, input: &str) -> ChatResult<String> {
        if let Some(command) = SlashCommand::parse(input) {
            return self.process_command(command).await;
        }

        let response = self.process_chat_request(input, None).await?;
        Ok(response.content)
    }

    /// Process user input and return structured diagnostics, without streaming.
    pub async fn process_input_with_diagnostics(
        &mut self,
        input: &str,
    ) -> ChatResult<ChatResponse> {
        if let Some(command) = SlashCommand::parse(input) {
            let text = self.process_command(command).await?;
            return Ok(ChatResponse::from_text(text));
        }

        self.process_chat_request(input, None).await
    }

    /// Process user input and emit streaming events while running the pipeline.
    pub async fn process_input_streaming(
        &mut self,
        input: &str,
        stream: Option<UnboundedSender<ChatStreamEvent>>,
    ) -> ChatResult<ChatResponse> {
        if let Some(command) = SlashCommand::parse(input) {
            let text = self.process_command(command).await?;
            if let Some(ref tx) = stream {
                let _ = tx.send(ChatStreamEvent::Status("Command executed".to_string()));
                let _ = tx.send(ChatStreamEvent::Chunk(text.clone()));
                let _ = tx.send(ChatStreamEvent::Finished(ChatResponse::from_text(
                    text.clone(),
                )));
            }
            return Ok(ChatResponse::from_text(text));
        }

        self.process_chat_request(input, stream.as_ref()).await
    }

    /// Process natural language input (NL → DSL → execute) with optional streaming updates
    async fn process_chat_request(
        &mut self,
        input: &str,
        stream: Option<&UnboundedSender<ChatStreamEvent>>,
    ) -> ChatResult<ChatResponse> {
        emit_stream(
            stream,
            ChatStreamEvent::Status("Planning request".to_string()),
        );

        let context_with_system = self.context_with_system_prompt();
        let translation = match self.translator.translate(input, &context_with_system).await {
            Ok(plan) => plan,
            Err(err) => {
                emit_stream(stream, ChatStreamEvent::Error(err.to_string()));
                return Err(err);
            }
        };

        emit_stream(
            stream,
            ChatStreamEvent::Plan(translation.outer_plan.clone()),
        );

        let plan_snapshot = translation.outer_plan.clone();
        let mut current_dsl = translation.dsl_code.clone();
        let max_attempts = 3;

        for attempt in 0..max_attempts {
            emit_stream(
                stream,
                ChatStreamEvent::CompilerMessage("Writing DSL artifact".to_string()),
            );
            let tmp_dir = self.config.session_storage_path.join("tmp");
            std::fs::create_dir_all(&tmp_dir)?;
            let timestamp = Utc::now().timestamp();
            let temp_path = tmp_dir.join(format!("chat_{}_{}_{}.ais", self.id, timestamp, attempt));
            std::fs::write(&temp_path, &current_dsl)?;

            emit_stream(
                stream,
                ChatStreamEvent::Status("Executing plan".to_string()),
            );

            match self.linker.run(&temp_path, false).await {
                Ok(result) => {
                    let response = if result.execution.results.is_empty() {
                        format!(
                            "Plan: {}

DSL executed successfully ({} nodes in {}ms)",
                            plan_snapshot.result.as_str(),
                            result.execution.stats.executed_nodes,
                            result.execution.stats.duration_ms
                        )
                    } else {
                        let mut output = format!(
                            "Plan: {}

",
                            plan_snapshot.result.as_str()
                        );
                        output.push_str(
                            "Results:
",
                        );
                        let mut entries: Vec<_> = result.execution.results.iter().collect();
                        entries.sort_by_key(|(token_id, _)| *token_id);
                        for (token_id, value) in entries {
                            let rendered = match value.to_json() {
                                Ok(json) => json.to_string(),
                                Err(_) => value.to_string(),
                            };
                            output.push_str(&format!(
                                "  [token {}] {}
",
                                token_id, rendered
                            ));
                        }
                        output
                    };

                    let mut execution_entries = Vec::new();
                    if !result.execution.results.is_empty() {
                        let mut entries: Vec<_> = result.execution.results.iter().collect();
                        entries.sort_by_key(|(token_id, _)| *token_id);
                        for (token_id, value) in entries {
                            let rendered = match value.to_json() {
                                Ok(json) => json.to_string(),
                                Err(_) => value.to_string(),
                            };
                            execution_entries.push(format!("[token {}] {}", token_id, rendered));
                        }
                    }

                    let mut compiler_messages = vec![format!(
                        "Executed {} nodes in {}ms",
                        result.execution.stats.executed_nodes, result.execution.stats.duration_ms
                    )];
                    compiler_messages
                        .push(format!("Temporary DSL saved to {}", temp_path.display()));

                    let _ = std::fs::remove_file(&temp_path);

                    emit_stream(
                        stream,
                        ChatStreamEvent::CompilerMessage("Execution finished".to_string()),
                    );

                    self.add_message("user", input).await?;
                    self.add_message("assistant", &response).await?;

                    let chat_response = ChatResponse {
                        content: response.clone(),
                        outer_plan: Some(plan_snapshot.clone()),
                        dsl_code: Some(current_dsl.clone()),
                        execution_results: execution_entries,
                        compiler_messages,
                    };

                    emit_stream(
                        stream,
                        ChatStreamEvent::Status("Streaming response".to_string()),
                    );
                    for chunk in response.split_inclusive(|c: char| c == ' ' || c == '\n') {
                        if !chunk.is_empty() {
                            emit_stream(stream, ChatStreamEvent::Chunk(chunk.to_string()));
                        }
                    }
                    emit_stream(stream, ChatStreamEvent::Finished(chat_response.clone()));

                    return Ok(chat_response);
                }
                Err(err) => {
                    let _ = std::fs::remove_file(&temp_path);
                    tracing::warn!(attempt = attempt + 1, error = %err, "Linker run failed");
                    if attempt + 1 < max_attempts && matches!(err, LinkerError::Compiler(_)) {
                        emit_stream(
                            stream,
                            ChatStreamEvent::Status(
                                "Compiler rejected DSL. Retrying...".to_string(),
                            ),
                        );
                        let feedback = err.to_string();
                        current_dsl = self
                            .translator
                            .regenerate_dsl_with_feedback(input, &plan_snapshot, &feedback)
                            .await?;
                        continue;
                    } else {
                        emit_stream(stream, ChatStreamEvent::Error(err.to_string()));
                        return Err(err.into());
                    }
                }
            }
        }

        Err(ChatError::Model {
            message: "Failed to compile DSL after retries".to_string(),
            source: None,
        })
    }

    /// Process slash command    /// Process slash command
    pub async fn process_command(&mut self, command: SlashCommand) -> ChatResult<String> {
        command.execute(self).await
    }

    /// Add a message to the conversation
    async fn add_message(&mut self, role: &str, content: &str) -> ChatResult<()> {
        let message = Message {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
        };

        self.messages.push(message.clone());
        self.storage.add_message(&self.id, &message).await?;

        // Update session info
        let mut session_info = self
            .storage
            .get_session(&self.id)
            .await?
            .ok_or_else(|| ChatError::Config("Session not found".to_string()))?;

        session_info.updated_at = Utc::now();
        session_info.message_count = self.messages.len();

        self.storage.update_session(&session_info).await?;

        Ok(())
    }

    /// Save session to storage
    pub async fn save(&self) -> ChatResult<()> {
        let session_info = self
            .storage
            .get_session(&self.id)
            .await?
            .ok_or_else(|| ChatError::Config("Session not found".to_string()))?;

        self.storage.update_session(&session_info).await?;

        Ok(())
    }

    /// List all sessions
    pub async fn list_sessions(storage_path: &Path) -> ChatResult<Vec<SessionInfo>> {
        let storage_path = storage_path.join("sessions.db");
        let storage = SessionStorage::new(&storage_path).await?;
        storage.list_sessions().await
    }

    /// Get conversation messages
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear conversation history
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    /// Get session ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get configuration
    pub fn config(&self) -> &ChatConfig {
        &self.config
    }

    /// Get mutable configuration (for slash commands)
    pub fn config_mut(&mut self) -> &mut ChatConfig {
        &mut self.config
    }

    /// Build conversation context, ensuring the system prompt is first
    fn context_with_system_prompt(&self) -> Vec<Message> {
        if self.system_prompt.trim().is_empty() {
            return self.messages.clone();
        }

        if self
            .messages
            .first()
            .map(|msg| msg.role == "system")
            .unwrap_or(false)
        {
            return self.messages.clone();
        }

        let mut context = Vec::with_capacity(self.messages.len() + 1);
        context.push(Message {
            role: "system".to_string(),
            content: self.system_prompt.clone(),
            timestamp: self
                .messages
                .first()
                .map(|msg| msg.timestamp)
                .unwrap_or_else(Utc::now),
        });
        context.extend(self.messages.iter().cloned());
        context
    }

    /// Reinitialize linker (after config change)
    pub async fn reinit_linker(&mut self) -> ChatResult<()> {
        let linker_config = LinkerConfig::from_apxm_config(self.config.apxm_config.clone());
        self.linker = Linker::new(linker_config).await?;

        // Also recreate translator with new registry
        let registry = self.linker.runtime_llm_registry();
        let planning_model = self
            .config
            .planning_model
            .clone()
            .unwrap_or_else(|| self.config.default_model.clone());
        self.translator = Translator::new(
            registry,
            planning_model,
            Some(self.config.default_model.clone()),
        );

        Ok(())
    }
}

fn emit_stream(stream: Option<&UnboundedSender<ChatStreamEvent>>, event: ChatStreamEvent) {
    if let Some(tx) = stream {
        let _ = tx.send(event);
    }
}
