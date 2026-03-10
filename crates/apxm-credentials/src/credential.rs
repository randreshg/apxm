use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single registered credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credential {
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

/// The credentials file structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CredentialsFile {
    #[serde(default)]
    pub credentials: BTreeMap<String, Credential>,
}

/// Summary info for listing (with masked key).
#[derive(Debug, Clone)]
pub struct CredentialSummary {
    pub provider: String,
    pub masked_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub header_count: usize,
}
