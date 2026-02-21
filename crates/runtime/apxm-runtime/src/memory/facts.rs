//! Structured facts API on top of LTM.

use super::MemorySystem;
use apxm_core::error::RuntimeError;
use apxm_core::types::values::Value;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

type Result<T> = std::result::Result<T, RuntimeError>;

const FACT_KEY_PREFIX: &str = "fact:";
const DEFAULT_HALF_LIFE_DAYS: f64 = 30.0;
const DEFAULT_MMR_LAMBDA: f64 = 0.7;

/// Stored fact record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub session_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Returned fact search result with ranking score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactResult {
    pub fact: Fact,
    pub score: f64,
}

/// Optional list filtering criteria.
#[derive(Debug, Clone, Default)]
pub struct FactFilter {
    pub tag: Option<String>,
    pub source: Option<String>,
}

impl MemorySystem {
    /// Store a fact and return its ID.
    pub async fn store_fact(
        &self,
        text: &str,
        tags: &[String],
        source: &str,
        session_id: Option<String>,
    ) -> Result<String> {
        let id = uuid::Uuid::now_v7().to_string();
        let now = Utc::now();
        let fact = Fact {
            id: id.clone(),
            text: text.to_string(),
            tags: tags.to_vec(),
            source: source.to_string(),
            session_id,
            created_at: now,
            updated_at: now,
        };
        self.write(super::MemorySpace::Ltm, fact_key(&id), fact_to_value(&fact))
            .await?;
        Ok(id)
    }

    /// Delete a fact by ID.
    pub async fn delete_fact(&self, id: &str) -> Result<()> {
        self.delete(super::MemorySpace::Ltm, &fact_key(id)).await
    }

    /// List facts with optional filtering.
    pub async fn list_facts(&self, filter: FactFilter) -> Result<Vec<Fact>> {
        let keys = self.ltm().list_keys().await?;
        let mut facts = Vec::new();
        for key in keys {
            if !key.starts_with(FACT_KEY_PREFIX) {
                continue;
            }
            let Some(value) = self.ltm().get(&key).await? else {
                continue;
            };
            let Some(fact) = value_to_fact(&value) else {
                continue;
            };
            if let Some(tag) = filter.tag.as_ref()
                && !fact.tags.iter().any(|t| t == tag)
            {
                continue;
            }
            if let Some(source) = filter.source.as_ref()
                && &fact.source != source
            {
                continue;
            }
            facts.push(fact);
        }
        facts.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(facts)
    }

    /// Search facts using hybrid search (FTS5 BM25 + optional vector) with
    /// temporal decay + MMR diversification on top.
    ///
    /// The search pipeline:
    /// 1. Backend hybrid search (FTS5 BM25 + cosine similarity if embedder is set)
    /// 2. Temporal decay (30-day half-life, recent facts score higher)
    /// 3. MMR re-ranking (lambda=0.7, reduces redundancy)
    pub async fn search_facts(&self, query: &str, limit: usize) -> Result<Vec<FactResult>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Use backend's hybrid search (FTS5 BM25 + optional vector) for initial retrieval
        let fetch_limit = limit * 3; // Over-fetch for MMR re-ranking
        let backend_results = self.ltm().search_semantic(query, fetch_limit).await?;

        // Filter to facts and apply temporal decay
        let mut scored = Vec::new();
        for result in backend_results {
            if !result.key.starts_with(FACT_KEY_PREFIX) {
                continue;
            }
            let Some(fact) = value_to_fact(&result.value) else {
                continue;
            };
            let decayed =
                apply_temporal_decay(result.score, fact.created_at, DEFAULT_HALF_LIFE_DAYS);
            scored.push(FactResult {
                fact,
                score: decayed,
            });
        }

        // If backend search returned nothing for facts, fall back to lexical scan
        if scored.is_empty() {
            let facts = self.list_facts(FactFilter::default()).await?;
            let query_terms = tokenize(query);
            for fact in facts {
                let relevance = lexical_relevance(&query_terms, &fact);
                if relevance <= 0.0 {
                    continue;
                }
                let decayed =
                    apply_temporal_decay(relevance, fact.created_at, DEFAULT_HALF_LIFE_DAYS);
                scored.push(FactResult {
                    fact,
                    score: decayed,
                });
            }
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(mmr_rerank(scored, limit, DEFAULT_MMR_LAMBDA))
    }
}

fn fact_key(id: &str) -> String {
    format!("{FACT_KEY_PREFIX}{id}")
}

fn fact_to_value(fact: &Fact) -> Value {
    let mut obj = HashMap::new();
    obj.insert("id".to_string(), Value::String(fact.id.clone()));
    obj.insert("text".to_string(), Value::String(fact.text.clone()));
    obj.insert(
        "tags".to_string(),
        Value::Array(fact.tags.iter().cloned().map(Value::String).collect()),
    );
    obj.insert("source".to_string(), Value::String(fact.source.clone()));
    if let Some(session) = &fact.session_id {
        obj.insert("session_id".to_string(), Value::String(session.clone()));
    } else {
        obj.insert("session_id".to_string(), Value::Null);
    }
    obj.insert(
        "created_at".to_string(),
        Value::String(fact.created_at.to_rfc3339()),
    );
    obj.insert(
        "updated_at".to_string(),
        Value::String(fact.updated_at.to_rfc3339()),
    );
    Value::Object(obj)
}

fn value_to_fact(value: &Value) -> Option<Fact> {
    let obj = value.as_object()?;
    let id = obj.get("id")?.as_string()?.clone();
    let text = obj.get("text")?.as_string()?.clone();
    let tags = obj
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_string().cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let source = obj
        .get("source")
        .and_then(|v| v.as_string())
        .cloned()
        .unwrap_or_default();
    let session_id = obj.get("session_id").and_then(|v| v.as_string()).cloned();
    let created_at = obj
        .get("created_at")
        .and_then(|v| v.as_string())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);
    let updated_at = obj
        .get("updated_at")
        .and_then(|v| v.as_string())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or(created_at);

    Some(Fact {
        id,
        text,
        tags,
        source,
        session_id,
        created_at,
        updated_at,
    })
}

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

fn lexical_relevance(query_terms: &HashSet<String>, fact: &Fact) -> f64 {
    if query_terms.is_empty() {
        return 1.0;
    }
    let mut corpus = fact.text.to_ascii_lowercase();
    if !fact.tags.is_empty() {
        corpus.push(' ');
        corpus.push_str(&fact.tags.join(" ").to_ascii_lowercase());
    }
    let mut matches = 0usize;
    for term in query_terms {
        if corpus.contains(term) {
            matches += 1;
        }
    }
    matches as f64 / query_terms.len() as f64
}

fn apply_temporal_decay(score: f64, created_at: DateTime<Utc>, half_life_days: f64) -> f64 {
    let age_days = (Utc::now() - created_at).num_seconds() as f64 / 86400.0;
    let decay = 0.5_f64.powf(age_days / half_life_days.max(1.0));
    score * decay
}

fn mmr_rerank(mut candidates: Vec<FactResult>, limit: usize, lambda: f64) -> Vec<FactResult> {
    if candidates.len() <= 1 {
        candidates.truncate(limit);
        return candidates;
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let mut selected = Vec::new();
    while !candidates.is_empty() && selected.len() < limit {
        let mut best_idx = 0usize;
        let mut best_score = f64::MIN;
        for (idx, candidate) in candidates.iter().enumerate() {
            let redundancy = selected
                .iter()
                .map(|picked: &FactResult| text_similarity(&candidate.fact.text, &picked.fact.text))
                .fold(0.0_f64, f64::max);
            let mmr = lambda * candidate.score - (1.0 - lambda) * redundancy;
            if mmr > best_score {
                best_score = mmr;
                best_idx = idx;
            }
        }
        selected.push(candidates.swap_remove(best_idx));
    }
    selected
}

fn text_similarity(a: &str, b: &str) -> f64 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    if ta.is_empty() && tb.is_empty() {
        return 0.0;
    }
    let intersection = ta.intersection(&tb).count() as f64;
    let union = ta.union(&tb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryConfig;

    #[tokio::test]
    async fn store_search_delete_fact_roundtrip() {
        let system = MemorySystem::new(MemoryConfig::in_memory_ltm())
            .await
            .expect("memory should initialize");
        let id = system
            .store_fact(
                "deploy server is 10.0.1.50",
                &["deploy".to_string(), "infra".to_string()],
                "test",
                Some("session-1".to_string()),
            )
            .await
            .expect("store_fact should succeed");
        let found = system
            .search_facts("deploy server", 5)
            .await
            .expect("search_facts should succeed");
        assert_eq!(found.len(), 1);
        assert!(found[0].fact.text.contains("10.0.1.50"));
        system
            .delete_fact(&id)
            .await
            .expect("delete_fact should succeed");
        let found_after = system
            .search_facts("deploy server", 5)
            .await
            .expect("search_facts should succeed");
        assert!(found_after.is_empty());
    }
}
