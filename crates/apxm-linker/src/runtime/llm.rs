use crate::error::LinkerError;
use apxm_config::{ApXmConfig, LlmBackendConfig};
use apxm_models::{
    provider::{Provider, ProviderId},
    registry::LLMRegistry,
};
use serde_json::{Map, Value as JsonValue, json};
use std::env;

pub async fn configure_llm_registry(
    registry: &LLMRegistry,
    config: &ApXmConfig,
) -> Result<(), LinkerError> {
    for backend in &config.llm_backends {
        let provider_name = backend.provider.as_deref().unwrap_or("openai").to_string();
        let provider_id = provider_name.parse::<ProviderId>().map_err(|e| {
            LinkerError::Config(format!("Unknown provider '{}': {e}", provider_name))
        })?;

        let api_key = resolve_api_key(backend)?;
        let backend_config: Option<JsonValue> = build_backend_config(backend)?;

        let provider = Provider::new(provider_id, &api_key, backend_config)
            .await
            .map_err(|e| {
                LinkerError::Config(format!("Failed to init backend '{}': {e}", backend.name))
            })?;

        registry
            .register(backend.name.clone(), provider)
            .map_err(|e| {
                LinkerError::Config(format!(
                    "Failed to register backend '{}': {e}",
                    backend.name
                ))
            })?;
    }

    if let Some(default) = config.chat.providers.first() {
        registry.set_default(default.clone()).map_err(|e| {
            LinkerError::Config(format!("Failed to set default backend '{}': {e}", default))
        })?;
    }

    Ok(())
}

fn resolve_api_key(config: &LlmBackendConfig) -> Result<String, LinkerError> {
    match config.api_key.as_deref() {
        Some(key) if key.starts_with("env:") => {
            let env_name = &key["env:".len()..];
            env::var(env_name).map_err(|_| {
                LinkerError::Config(format!(
                    "Environment variable '{}' not set for backend '{}'",
                    env_name, config.name
                ))
            })
        }
        Some(key) if !key.is_empty() => Ok(key.to_string()),
        _ => Err(LinkerError::Config(format!(
            "Missing API key for backend '{}'. Set `api_key` or use `env:VAR`.",
            config.name
        ))),
    }
}

fn build_backend_config(config: &LlmBackendConfig) -> Result<Option<JsonValue>, LinkerError> {
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
