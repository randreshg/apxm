//! LLM registry configuration for the runtime.

use crate::config::{ApXmConfig, LlmBackendConfig};
use crate::error::DriverError;
use apxm_backends::{LLMRegistry, Provider, ProviderId};
use serde_json::{Map, Value as JsonValue, json};
use std::env;

pub async fn configure_llm_registry(
    registry: &LLMRegistry,
    config: &ApXmConfig,
) -> Result<(), DriverError> {
    let allowed_backends = if config.chat.providers.is_empty() {
        None
    } else {
        Some(
            config
                .chat
                .providers
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>(),
        )
    };

    for backend in &config.llm_backends {
        if let Some(allowed) = &allowed_backends
            && !allowed.contains(&backend.name)
        {
            continue;
        }

        let provider_name = backend.provider.as_deref().unwrap_or("openai").to_string();
        let provider_id = provider_name.parse::<ProviderId>().map_err(|e| {
            DriverError::Driver(format!("Unknown provider '{}': {e}", provider_name))
        })?;

        let api_key = resolve_api_key(provider_id.clone(), backend)?;
        let backend_config: Option<JsonValue> = build_backend_config(backend)?;

        let provider = Provider::new(provider_id, &api_key, backend_config)
            .await
            .map_err(|e| {
                DriverError::Driver(format!("Failed to init backend '{}': {e}", backend.name))
            })?;

        registry
            .register(backend.name.clone(), provider)
            .map_err(|e| {
                DriverError::Driver(format!(
                    "Failed to register backend '{}': {e}",
                    backend.name
                ))
            })?;
    }

    if let Some(default) = config.chat.providers.first() {
        registry.set_default(default.clone()).map_err(|e| {
            DriverError::Driver(format!("Failed to set default backend '{}': {e}", default))
        })?;
    }

    Ok(())
}

fn resolve_api_key(provider: ProviderId, config: &LlmBackendConfig) -> Result<String, DriverError> {
    match config.api_key.as_deref() {
        Some(key) if key.starts_with("env:") => {
            let env_name = &key["env:".len()..];
            env::var(env_name).map_err(|_| {
                DriverError::Driver(format!(
                    "Environment variable '{}' not set for backend '{}'",
                    env_name, config.name
                ))
            })
        }
        Some(key) if !key.is_empty() => Ok(key.to_string()),
        _ if matches!(provider, ProviderId::Ollama) => Ok(String::new()),
        _ => Err(DriverError::Driver(format!(
            "Missing API key for backend '{}'. Set `api_key` or use `env:VAR`.",
            config.name
        ))),
    }
}

fn build_backend_config(config: &LlmBackendConfig) -> Result<Option<JsonValue>, DriverError> {
    let mut map = Map::new();

    if let Some(model) = &config.model {
        map.insert("model".to_string(), json!(model));
    }
    if let Some(endpoint) = &config.endpoint {
        map.insert("base_url".to_string(), json!(endpoint));
    }
    if !config.options.is_empty() {
        for (key, value) in &config.options {
            map.insert(key.clone(), json!(value));
        }
    }

    Ok(if map.is_empty() {
        None
    } else {
        Some(JsonValue::Object(map))
    })
}
