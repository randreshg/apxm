use apxm_core::{error::RuntimeError, types::Value};
use apxm_runtime::capability::{
    executor::{CapabilityExecutor, CapabilityResult},
    metadata::CapabilityMetadata,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchDepth {
    #[default]
    Basic,
    Advanced,
}

impl SearchDepth {
    fn as_tavily_value(self) -> &'static str {
        match self {
            SearchDepth::Basic => "basic",
            SearchDepth::Advanced => "advanced",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchWebConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default)]
    pub blocked_queries: Vec<String>,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    #[serde(default)]
    pub safe_search: bool,
    #[serde(default)]
    pub search_depth: SearchDepth,
    #[serde(default = "default_tavily_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_true")]
    pub include_answer: bool,
}

fn default_true() -> bool {
    true
}

fn default_max_results() -> usize {
    5
}

fn default_tavily_endpoint() -> String {
    "https://api.tavily.com/search".to_string()
}

impl Default for SearchWebConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_domains: None,
            blocked_domains: Vec::new(),
            blocked_queries: Vec::new(),
            max_results: default_max_results(),
            safe_search: false,
            search_depth: SearchDepth::Basic,
            endpoint: default_tavily_endpoint(),
            include_answer: true,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    #[serde(default)]
    results: Vec<TavilyResult>,
    #[serde(default)]
    answer: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
}

pub struct SearchWebCapability {
    metadata: CapabilityMetadata,
    config: SearchWebConfig,
    client: Client,
}

impl SearchWebCapability {
    pub fn new() -> Self {
        Self::with_config(SearchWebConfig::default())
    }

    pub fn with_config(config: SearchWebConfig) -> Self {
        Self {
            metadata: CapabilityMetadata::new(
                "search_web",
                "Search the web via Tavily API",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results"
                        }
                    },
                    "required": ["query"]
                }),
            )
            .with_returns("string")
            .with_latency(450),
            config,
            client: Client::new(),
        }
    }

    pub fn docs() -> Self {
        Self::with_config(SearchWebConfig {
            allowed_domains: Some(
                vec![
                    "docs.rs",
                    "doc.rust-lang.org",
                    "crates.io",
                    "docs.python.org",
                    "pypi.org",
                    "developer.mozilla.org",
                    "nodejs.org",
                    "pkg.go.dev",
                    "learn.microsoft.com",
                    "docs.github.com",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            max_results: 10,
            safe_search: true,
            search_depth: SearchDepth::Basic,
            ..Default::default()
        })
    }

    pub fn research() -> Self {
        Self::with_config(SearchWebConfig {
            max_results: 15,
            safe_search: true,
            search_depth: SearchDepth::Advanced,
            ..Default::default()
        })
    }

    fn check_query_policy(&self, query: &str) -> CapabilityResult<()> {
        let lowercase_query = query.to_lowercase();
        if let Some(blocked_term) = self
            .config
            .blocked_queries
            .iter()
            .find(|term| lowercase_query.contains(term.to_lowercase().as_str()))
        {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!("Query contains blocked term '{blocked_term}'"),
            });
        }
        Ok(())
    }

    fn is_url_allowed(&self, url: &str) -> bool {
        let domain = url
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .split('/')
            .next()
            .unwrap_or_default();

        if self
            .config
            .blocked_domains
            .iter()
            .any(|blocked| domain.contains(blocked))
        {
            return false;
        }

        if let Some(allowed_domains) = &self.config.allowed_domains {
            return allowed_domains
                .iter()
                .any(|allowed| domain.contains(allowed));
        }

        true
    }
}

impl Default for SearchWebCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CapabilityExecutor for SearchWebCapability {
    async fn execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value> {
        let query = args
            .get("query")
            .or_else(|| args.get("arg_query"))
            .or_else(|| args.get("arg0"))
            .and_then(|value| value.as_string())
            .ok_or_else(|| RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: "Missing required 'query' argument".to_string(),
            })?
            .to_string();

        self.check_query_policy(&query)?;

        let max_results = args
            .get("max_results")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .unwrap_or(self.config.max_results);

        let api_key = std::env::var("TAVILY_API_KEY").map_err(|_| RuntimeError::Capability {
            capability: self.metadata.name.clone(),
            message: "TAVILY_API_KEY environment variable is not set".to_string(),
        })?;

        let mut request_body = json!({
            "api_key": api_key,
            "query": query,
            "search_depth": self.config.search_depth.as_tavily_value(),
            "include_answer": self.config.include_answer,
            "max_results": max_results,
            "safe_search": self.config.safe_search,
        });

        if let Some(allowed_domains) = &self.config.allowed_domains {
            request_body["include_domains"] = json!(allowed_domains);
        }
        if !self.config.blocked_domains.is_empty() {
            request_body["exclude_domains"] = json!(self.config.blocked_domains);
        }

        let response = self
            .client
            .post(&self.config.endpoint)
            .json(&request_body)
            .send()
            .await
            .map_err(|error| RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!("Search request failed: {error}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!("Search API responded with status {status}: {body}"),
            });
        }

        let tavily_response =
            response
                .json::<TavilyResponse>()
                .await
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!("Unable to parse search response: {error}"),
                })?;

        let filtered_results = tavily_response
            .results
            .into_iter()
            .filter(|result| self.is_url_allowed(&result.url))
            .collect::<Vec<_>>();

        let mut output = String::new();
        if let Some(answer) = tavily_response.answer
            && !answer.trim().is_empty()
        {
            output.push_str(&format!("Summary: {answer}\n\n"));
        }

        for (index, result) in filtered_results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} ({})\n{}\n\n",
                index + 1,
                result.title,
                result.url,
                result.content
            ));
        }

        if output.trim().is_empty() {
            output = "No results found.".to_string();
        }

        Ok(Value::String(output))
    }

    fn metadata(&self) -> &CapabilityMetadata {
        &self.metadata
    }
}
