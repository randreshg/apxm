//! LLM registry configuration for the runtime.

use crate::config::{ApXmConfig, LlmBackendConfig};
use crate::error::DriverError;
use apxm_backends::{LLMRegistry, Provider, ProviderId};
use apxm_core::constants::graph::attrs::{BASE_URL, MODEL};
use apxm_core::types::resolve_builtin_provider;
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

    // Primary source: registered credentials from ~/.apxm/credentials.toml
    let mut loaded_from_credentials = false;
    if let Ok(store) = apxm_credentials::CredentialStore::open() {
        if let Ok(credentials) = store.list_all() {
            if !credentials.is_empty() {
                loaded_from_credentials = true;
                for (name, credential) in &credentials {
                    if let Some(allowed) = &allowed_backends {
                        if !allowed.contains(name) {
                            continue;
                        }
                    }

                    let backend_config = credential_to_backend_config(name, credential);
                    let provider_name = &credential.provider;

                    let provider_id = resolve_provider_id(provider_name)?;
                    let api_key = credential.api_key.clone().unwrap_or_default();
                    let backend_json = build_backend_config(&backend_config)?;

                    let provider = Provider::new(provider_id, &api_key, backend_json)
                        .await
                        .map_err(|e| {
                            DriverError::Driver(format!("Failed to init backend '{}': {e}", name))
                        })?;

                    registry.register(name.clone(), provider).map_err(|e| {
                        DriverError::Driver(format!("Failed to register backend '{}': {e}", name))
                    })?;
                }
            }
        }
    }

    // Fallback: load from config.toml's [[llm_backends]] if no credentials were loaded
    if !loaded_from_credentials {
        for backend in &config.llm_backends {
            if let Some(allowed) = &allowed_backends
                && !allowed.contains(&backend.name)
            {
                continue;
            }

            let provider_name = backend.provider.as_deref().unwrap_or("openai").to_string();
            let provider_id = resolve_provider_id(&provider_name)?;
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
    }

    if let Some(default) = config.chat.providers.first() {
        registry.set_default(default.clone()).map_err(|e| {
            DriverError::Driver(format!("Failed to set default backend '{}': {e}", default))
        })?;
    }

    Ok(())
}

/// Resolve a provider name to a ProviderId, using builtin specs as fallback.
fn resolve_provider_id(provider_name: &str) -> Result<ProviderId, DriverError> {
    match provider_name.parse::<ProviderId>() {
        Ok(id) => Ok(id),
        Err(_) => {
            let spec = resolve_builtin_provider(provider_name).ok_or_else(|| {
                DriverError::Driver(format!("Unknown provider '{}'", provider_name))
            })?;
            Ok(match spec.protocol {
                apxm_core::types::ProviderProtocol::OpenAI => ProviderId::OpenAI,
                apxm_core::types::ProviderProtocol::Anthropic => ProviderId::Anthropic,
                apxm_core::types::ProviderProtocol::Google => ProviderId::Google,
                apxm_core::types::ProviderProtocol::Ollama => ProviderId::Ollama,
            })
        }
    }
}

/// Convert a credential from the credential store to an LlmBackendConfig.
fn credential_to_backend_config(
    name: &str,
    credential: &apxm_credentials::credential::Credential,
) -> LlmBackendConfig {
    LlmBackendConfig {
        name: name.to_string(),
        provider: Some(credential.provider.clone()),
        model: credential.model.clone(),
        api_key: credential.api_key.clone(),
        endpoint: credential.base_url.clone(),
        options: std::collections::HashMap::new(),
        extra_headers: credential
            .headers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    }
}

fn resolve_api_key(provider: ProviderId, config: &LlmBackendConfig) -> Result<String, DriverError> {
    match config.api_key.as_deref() {
        Some(key) if key.starts_with("env:") => {
            let env_name = key.strip_prefix("env:").unwrap();
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

/// Resolve a value that may be prefixed with `env:` to read from an environment variable.
///
/// Returns an error when an `env:` prefix is present but the variable is not set,
/// ensuring misconfiguration is caught at startup rather than at request time.
fn resolve_env_value(value: &str, field: &str, backend: &str) -> Result<String, DriverError> {
    if let Some(var_name) = value.strip_prefix("env:") {
        env::var(var_name).map_err(|_| {
            DriverError::Driver(format!(
                "Environment variable '{}' not set for field '{}' in backend '{}'",
                var_name, field, backend
            ))
        })
    } else {
        Ok(value.to_string())
    }
}

fn build_backend_config(config: &LlmBackendConfig) -> Result<Option<JsonValue>, DriverError> {
    let mut map = Map::new();

    if let Some(model) = &config.model {
        let resolved = resolve_env_value(model, MODEL, &config.name)?;
        map.insert(MODEL.to_string(), json!(resolved));
    }
    if let Some(endpoint) = &config.endpoint {
        let resolved = resolve_env_value(endpoint, "endpoint", &config.name)?;
        map.insert(BASE_URL.to_string(), json!(resolved));
    }
    if !config.options.is_empty() {
        for (key, value) in &config.options {
            let resolved = resolve_env_value(value, key, &config.name)?;
            map.insert(key.clone(), json!(resolved));
        }
    }

    // Resolve and forward extra_headers so the backend can inject custom HTTP
    // headers (e.g. Ocp-Apim-Subscription-Key for on-premises LLM gateways).
    // Values are resolved here at startup so missing env vars fail fast.
    if !config.extra_headers.is_empty() {
        let mut headers = Map::new();
        for (k, v) in &config.extra_headers {
            let resolved = resolve_env_value(v, &format!("extra_headers.{}", k), &config.name)?;
            headers.insert(k.clone(), json!(resolved));
        }
        map.insert("extra_headers".to_string(), JsonValue::Object(headers));
    }

    Ok(if map.is_empty() {
        None
    } else {
        Some(JsonValue::Object(map))
    })
}
