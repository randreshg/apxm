//! Chat session management with execution

use crate::{
    commands::SlashCommand,
    config::ChatConfig,
    error::{ChatError, ChatResult},
    storage::{Message, SessionInfo, SessionStorage},
    translator::Translator,
};
use apxm_core::Plan;
use apxm_linker::{Linker, LinkerConfig, error::LinkerError};
use chrono::Utc;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

/// Chat session with execution capabilities
pub struct ChatSession {
    /// Session ID
    pub id: String,

    /// Storage backend
    storage: SessionStorage,

    /// Linker for DSL compilation and execution
    linker: Arc<Linker>,

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
    /// Plan emitted by the translator
    pub plan: Option<Plan>,
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
            plan: None,
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
    Plan(Plan),
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
        let linker = Arc::new(Linker::new(linker_config).await?);

        // Create translator with LLM registry from linker
        let registry = linker.runtime_llm_registry();
        let planning_model = config
            .planning_model
            .clone()
            .unwrap_or_else(|| config.default_model.clone());

        // Get available capabilities from runtime for validation
        let available_capabilities = linker.runtime_capabilities();
        tracing::debug!(
            capability_count = available_capabilities.len(),
            capabilities = ?available_capabilities,
            "Passing runtime capabilities to translator for DSL validation"
        );

        let translator = Translator::new(
            registry,
            planning_model,
            Some(config.default_model.clone()),
            Some(available_capabilities),
            Some(Arc::clone(&linker)),
        );

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
        let linker = Arc::new(Linker::new(linker_config).await?);

        // Create translator
        let registry = linker.runtime_llm_registry();
        let planning_model = config
            .planning_model
            .clone()
            .unwrap_or_else(|| config.default_model.clone());

        // Get available capabilities from runtime for validation
        let available_capabilities = linker.runtime_capabilities();
        tracing::debug!(
            capability_count = available_capabilities.len(),
            capabilities = ?available_capabilities,
            "Passing runtime capabilities to translator for DSL validation (session load)"
        );

        let translator = Translator::new(
            registry,
            planning_model,
            Some(config.default_model.clone()),
            Some(available_capabilities),
            Some(Arc::clone(&linker)),
        );

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
        // Validate input length to prevent resource exhaustion
        const MAX_INPUT_LENGTH: usize = 100_000; // 100KB limit
        if input.len() > MAX_INPUT_LENGTH {
            let error = ChatError::Config(format!(
                "Input exceeds maximum allowed size ({} bytes > {} bytes)",
                input.len(),
                MAX_INPUT_LENGTH
            ));
            emit_stream(stream, ChatStreamEvent::Error(error.to_string()));
            return Err(error);
        }

        if input.trim().is_empty() {
            let error = ChatError::Config("Input cannot be empty".to_string());
            emit_stream(stream, ChatStreamEvent::Error(error.to_string()));
            return Err(error);
        }

        emit_stream(
            stream,
            ChatStreamEvent::Status("Planning request...".to_string()),
        );

        let context_with_system = self.context_with_system_prompt();

        emit_stream(
            stream,
            ChatStreamEvent::CompilerMessage("Generating structured plan from request".to_string()),
        );

        let translation = match self.translator.translate(input, &context_with_system).await {
            Ok(plan) => plan,
            Err(err) => {
                let error_msg = format!(
                    "Translation failed: {}\n\nThis usually means:\n\
                     1. The LLM returned invalid JSON for the plan, or\n\
                     2. The DSL model couldn't generate valid syntax.\n\
                     Check your LLM backend configuration and try again.",
                    err
                );
                emit_stream(stream, ChatStreamEvent::Error(error_msg.clone()));
                emit_stream(stream, ChatStreamEvent::CompilerMessage(error_msg.clone()));
                return Err(err);
            }
        };

        emit_stream(
            stream,
            ChatStreamEvent::CompilerMessage(format!(
                "Plan generated successfully: {}",
                translation.plan.result
            )),
        );

        emit_stream(
            stream,
            ChatStreamEvent::Plan(translation.plan.clone()),
        );

        let plan_snapshot = translation.plan.clone();
        let mut current_dsl = translation.dsl_code.clone();
        let max_attempts = 3;

        emit_stream(
            stream,
            ChatStreamEvent::Status("Compiling and executing DSL...".to_string()),
        );

        for attempt in 0..max_attempts {
            let attempt_msg = if attempt == 0 {
                "Compiling DSL to executable artifact".to_string()
            } else {
                format!(
                    "Retry {}/{}: Recompiling with fixes",
                    attempt + 1,
                    max_attempts
                )
            };

            emit_stream(stream, ChatStreamEvent::CompilerMessage(attempt_msg));

            // Create secure temporary file for DSL compilation
            let tmp_dir = self.config.session_storage_path.join("tmp");
            std::fs::create_dir_all(&tmp_dir)?;

            // Use timestamp and random component for uniqueness and avoid collisions
            let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or(0);
            let random_suffix = uuid::Uuid::now_v7();
            let temp_path = tmp_dir.join(format!(
                "chat_{}_{}_{}_{}.ais",
                self.id, timestamp, attempt, random_suffix
            ));

            std::fs::write(&temp_path, &current_dsl)?;

            emit_stream(
                stream,
                ChatStreamEvent::Status(format!(
                    "Executing plan (attempt {}/{})",
                    attempt + 1,
                    max_attempts
                )),
            );

            match self.linker.run(&temp_path, false).await {
                Ok(result) => {
                    // Build execution entries (stringified) from runtime results (if any)
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

                    // Build diagnostic messages for the specialist panel
                    let mut compiler_messages = vec![
                        format!("✓ Compilation successful"),
                        format!(
                            "✓ Executed {} nodes in {}ms",
                            result.execution.stats.executed_nodes,
                            result.execution.stats.duration_ms
                        ),
                    ];

                    if attempt > 0 {
                        compiler_messages
                            .push(format!("ℹ Succeeded after {} retry(ies)", attempt + 1));
                    }

                    // Construct the user-facing response:
                    // - If runtime produced exit outputs, present them as before.
                    // - If no exit outputs were produced, present a diagnostic-rich payload:
                    //   include plan summary, generated DSL, execution stats, partial outputs,
                    //   and compiler/linker messages so the user can inspect what happened.
                    let response = if !result.execution.results.is_empty() {
                        let mut output = String::new();
                        let mut entries: Vec<_> = result.execution.results.iter().collect();
                        entries.sort_by_key(|(token_id, _)| *token_id);

                        // Only show the actual results, not token IDs (compact)
                        for (_, value) in entries {
                            let rendered = match value.to_json() {
                                Ok(json) => {
                                    // Try to pretty-print JSON
                                    if let Ok(pretty) = serde_json::to_string_pretty(&json) {
                                        pretty
                                    } else {
                                        json.to_string()
                                    }
                                }
                                Err(_) => value.to_string(),
                            };
                            if !output.is_empty() {
                                output.push_str("\n\n");
                            }
                            output.push_str(&rendered);
                        }
                        output
                    } else {
                        // Diagnostic mode: no exit tokens produced
                        let mut out = String::new();

                        // Plan summary
                        let plan_result = plan_snapshot.result.trim();
                        if plan_result.is_empty() {
                            out.push_str("Plan summary:\n\n");
                        } else {
                            out.push_str(&format!("Plan: {}\n\n", plan_result));
                        }

                        // Show generated DSL so the user can inspect what was run
                        out.push_str("=== Generated DSL ===\n");
                        out.push_str(&current_dsl);
                        out.push_str("\n\n");

                        // Execution summary
                        out.push_str("=== Execution ===\n");
                        out.push_str("No exit tokens were produced by the program.\n\n");
                        out.push_str(&format!(
                            "Executed nodes: {}\nDuration: {}ms\n\n",
                            result.execution.stats.executed_nodes,
                            result.execution.stats.duration_ms
                        ));

                        // If there are any partial outputs (node-level values), show them
                        if !execution_entries.is_empty() {
                            out.push_str("=== Partial / Unmapped Outputs ===\n");
                            for e in &execution_entries {
                                out.push_str(&format!("- {}\n", e));
                            }
                            out.push_str("\n");
                        }

                        // Compiler / linker messages
                        if !compiler_messages.is_empty() {
                            out.push_str("=== Compiler / Linker Messages ===\n");
                            out.push_str(&compiler_messages.join("\n"));
                            out.push_str("\n");
                        }

                        // Guidance for the user
                        out.push_str(
                            "\nHint: The program may have produced values that were not mapped\n",
                        );
                        out.push_str(
                            "to the outer plan's exit tokens, or inner-plan linking may be\n",
                        );
                        out.push_str(
                            "unsupported in this runtime. Inspect the DSL and compiler messages\n",
                        );
                        out.push_str("above to determine the correct next step.\n");

                        out
                    };

                    let _ = std::fs::remove_file(&temp_path);

                    // Emit a clear compiler message for the specialist panel.
                    // If there were no exit tokens, this helps surface the condition.
                    if result.execution.results.is_empty() {
                        emit_stream(
                            stream,
                            ChatStreamEvent::CompilerMessage(
                                "✗ No exit tokens produced by program (see diagnostics)"
                                    .to_string(),
                            ),
                        );
                    } else {
                        emit_stream(
                            stream,
                            ChatStreamEvent::CompilerMessage(
                                "✓ Execution completed successfully".to_string(),
                            ),
                        );
                    }

                    // Save messages to conversation history
                    self.add_message("user", input).await?;
                    self.add_message("assistant", &response).await?;

                    let chat_response = ChatResponse {
                        content: response.clone(),
                        plan: Some(plan_snapshot.clone()),
                        dsl_code: Some(current_dsl.clone()),
                        execution_results: execution_entries,
                        compiler_messages,
                    };

                    // Stream the clean response to the chat
                    emit_stream(stream, ChatStreamEvent::Status("Ready".to_string()));

                    // Stream response naturally (word by word for better UX)
                    for chunk in response.split_inclusive(|c: char| [' ', '\n'].contains(&c)) {
                        if !chunk.is_empty() {
                            emit_stream(stream, ChatStreamEvent::Chunk(chunk.to_string()));
                        }
                    }

                    emit_stream(stream, ChatStreamEvent::Finished(chat_response.clone()));

                    return Ok(chat_response);
                }
                Err(err) => {
                    // Clean up temp file even on error (prevent accumulation)
                    let _ = std::fs::remove_file(&temp_path);
                    tracing::warn!(attempt = attempt + 1, error = %err, "Linker run failed");

                    // Preserve full error details from compiler
                    let error_details = err.to_string();

                    // Emit detailed compiler error (no information loss)
                    emit_stream(
                        stream,
                        ChatStreamEvent::CompilerMessage(format!(
                            "✗ Compilation error:\n{}\n\nTip: Check DSL syntax and dependencies",
                            error_details
                        )),
                    );

                    if attempt + 1 < max_attempts && matches!(err, LinkerError::Compiler(_)) {
                        emit_stream(
                            stream,
                            ChatStreamEvent::Status(format!(
                                "Fixing errors (attempt {}/{})...",
                                attempt + 2,
                                max_attempts
                            )),
                        );
                        emit_stream(
                            stream,
                            ChatStreamEvent::CompilerMessage(format!(
                                "ℹ Retrying with error feedback..."
                            )),
                        );

                        // Provide full compiler error to LLM for fixing
                        let detailed_feedback = format!(
                            "The generated DSL failed to compile/execute with this error:\n\n{}\n\n\
                            Please regenerate the DSL addressing this error while maintaining:\n\
                            1. Correct syntax (semicolons, parentheses in if conditions)\n\
                            2. Linear flow with no circular dependencies\n\
                            3. Valid capability names and operation types\n\
                            4. Simple, minimal implementation",
                            error_details
                        );

                        current_dsl = self
                            .translator
                            .regenerate_dsl_with_feedback(input, &plan_snapshot, &detailed_feedback)
                            .await?;
                        continue;
                    } else {
                        // Final failure - provide clear error with full details
                        let final_error = format!(
                            "Failed to generate valid DSL after {} attempts.\n\n\
                            Compiler error:\n{}\n\n\
                            Suggestions:\n\
                            • Simplify your request\n\
                            • Break into smaller steps\n\
                            • Check Compiler panel for full details",
                            attempt + 1,
                            error_details
                        );

                        emit_stream(stream, ChatStreamEvent::Error(final_error.clone()));

                        // Full error details with generated DSL in compiler panel
                        emit_stream(
                            stream,
                            ChatStreamEvent::CompilerMessage(format!(
                                "✗ Complete error log:\n{}\n\n=== Generated DSL ===\n{}",
                                error_details,
                                current_dsl
                            )),
                        );

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
        self.linker = Arc::new(Linker::new(linker_config).await?);

        // Also recreate translator with new registry
        let registry = self.linker.runtime_llm_registry();
        let planning_model = self
            .config
            .planning_model
            .clone()
            .unwrap_or_else(|| self.config.default_model.clone());

        // Get available capabilities from runtime for validation
        let available_capabilities = self.linker.runtime_capabilities();
        tracing::debug!(
            capability_count = available_capabilities.len(),
            capabilities = ?available_capabilities,
            "Passing runtime capabilities to translator for DSL validation (reinit_linker)"
        );

        self.translator = Translator::new(
            registry,
            planning_model,
            Some(self.config.default_model.clone()),
            Some(available_capabilities),
            Some(Arc::clone(&self.linker)),
        );

        Ok(())
    }
}

fn emit_stream(stream: Option<&UnboundedSender<ChatStreamEvent>>, event: ChatStreamEvent) {
    if let Some(tx) = stream {
        let _ = tx.send(event);
    }
}
