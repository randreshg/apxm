use crate::CredentialError;
use crate::credential::Credential;

/// Validate a credential by making a minimal API call.
pub async fn validate_credential(name: &str, cred: &Credential) -> Result<String, CredentialError> {
    let client = reqwest::Client::new();

    match cred.provider.as_str() {
        "openai" | "openrouter" => {
            let base = cred
                .base_url
                .as_deref()
                .unwrap_or(if cred.provider == "openrouter" {
                    "https://openrouter.ai/api"
                } else {
                    "https://api.openai.com"
                });
            let base_trimmed = base.trim_end_matches('/');
            let api_key = cred
                .api_key
                .as_deref()
                .ok_or_else(|| CredentialError::Validation {
                    name: name.to_string(),
                    reason: "No API key set".to_string(),
                })?;

            // Try GET /v1/models first (standard OpenAI)
            let models_url = format!("{}/v1/models", base_trimmed);
            let mut req = client.get(&models_url).bearer_auth(api_key);
            for (k, v) in &cred.headers {
                req = req.header(k.as_str(), v.as_str());
            }

            let resp = req.send().await.map_err(|e| CredentialError::Validation {
                name: name.to_string(),
                reason: format!("Request failed: {e}"),
            })?;

            if resp.status().is_success() {
                return Ok(format!("OK ({})", resp.status()));
            }

            // Fallback: minimal chat completion (for on-premises/custom gateways
            // that don't expose /v1/models but do serve /chat/completions)
            let model = cred.model.as_deref().unwrap_or("gpt-4o-mini");
            let chat_url = format!("{}/chat/completions", base_trimmed);
            let mut req = client
                .post(&chat_url)
                .bearer_auth(api_key)
                .header("content-type", "application/json")
                .body(format!(
                    r#"{{"model":"{}","max_completion_tokens":1,"messages":[{{"role":"user","content":"hi"}}]}}"#,
                    model
                ));
            for (k, v) in &cred.headers {
                req = req.header(k.as_str(), v.as_str());
            }

            let resp = req.send().await.map_err(|e| CredentialError::Validation {
                name: name.to_string(),
                reason: format!("Chat completions request failed: {e}"),
            })?;

            if resp.status().is_success() || resp.status().as_u16() == 400 {
                Ok(format!("OK ({})", resp.status()))
            } else {
                Err(CredentialError::Validation {
                    name: name.to_string(),
                    reason: format!("HTTP {}", resp.status()),
                })
            }
        }
        "anthropic" => {
            let base = cred
                .base_url
                .as_deref()
                .unwrap_or("https://api.anthropic.com");
            let url = format!("{}/v1/messages", base.trim_end_matches('/'));
            let api_key = cred
                .api_key
                .as_deref()
                .ok_or_else(|| CredentialError::Validation {
                    name: name.to_string(),
                    reason: "No API key set".to_string(),
                })?;

            let resp = client
                .post(&url)
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .body(r#"{"model":"claude-3-haiku-20240307","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}"#)
                .send()
                .await
                .map_err(|e| CredentialError::Validation {
                    name: name.to_string(),
                    reason: format!("Request failed: {e}"),
                })?;

            if resp.status().is_success() || resp.status().as_u16() == 400 {
                // 400 means auth worked but request was bad - key is valid
                Ok(format!("OK ({})", resp.status()))
            } else {
                Err(CredentialError::Validation {
                    name: name.to_string(),
                    reason: format!("HTTP {}", resp.status()),
                })
            }
        }
        "google" => {
            let base = cred
                .base_url
                .as_deref()
                .unwrap_or("https://generativelanguage.googleapis.com");
            let api_key = cred
                .api_key
                .as_deref()
                .ok_or_else(|| CredentialError::Validation {
                    name: name.to_string(),
                    reason: "No API key set".to_string(),
                })?;
            let url = format!("{}/v1/models?key={}", base.trim_end_matches('/'), api_key);

            let resp = client
                .get(&url)
                .send()
                .await
                .map_err(|e| CredentialError::Validation {
                    name: name.to_string(),
                    reason: format!("Request failed: {e}"),
                })?;

            if resp.status().is_success() {
                Ok(format!("OK ({})", resp.status()))
            } else {
                Err(CredentialError::Validation {
                    name: name.to_string(),
                    reason: format!("HTTP {}", resp.status()),
                })
            }
        }
        "ollama" => {
            let base = cred.base_url.as_deref().unwrap_or("http://localhost:11434");
            let url = format!("{}/api/tags", base.trim_end_matches('/'));

            let resp = client
                .get(&url)
                .send()
                .await
                .map_err(|e| CredentialError::Validation {
                    name: name.to_string(),
                    reason: format!("Connection failed: {e}"),
                })?;

            if resp.status().is_success() {
                Ok(format!("OK ({})", resp.status()))
            } else {
                Err(CredentialError::Validation {
                    name: name.to_string(),
                    reason: format!("HTTP {}", resp.status()),
                })
            }
        }
        other => Err(CredentialError::Validation {
            name: name.to_string(),
            reason: format!("Unknown provider '{}' - cannot validate", other),
        }),
    }
}
