//! SQLite storage backend with connection pooling for high performance
//! Provides persistent, queryable storage suitable for LTM (Long-Term Memory)

use crate::storage::{SearchResult, StorageBackend, StorageResult, backend::BackendStats};
use apxm_core::{error::RuntimeError, types::values::Value};
use async_trait::async_trait;
use sqlx::{
    SqlitePool,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

const SQLITE_SPACE: &str = "sqlite";
const SCHEMA_KV_STORE: &str = r#"
CREATE TABLE IF NOT EXISTS kv_store (
    key        TEXT PRIMARY KEY NOT NULL,
    value      TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at INTEGER NOT NULL DEFAULT (unixepoch())
) WITHOUT ROWID;
"#;

#[inline]
fn mem_err(context: &'static str, e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Memory {
        message: format!("sqlite {context}: {e}"),
        space: Some(SQLITE_SPACE.to_string()),
    }
}

#[inline]
fn ser_err(context: &'static str, e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Serialization(format!("{context}: {e}"))
}

/// Escape user input for use inside a `LIKE` pattern, treating the query as a literal substring.
/// Uses backslash as the escape character (paired with `ESCAPE '\\'`).
#[inline]
fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

#[derive(Clone)]
pub struct SqliteBackend {
    pool: SqlitePool,
    path: PathBuf,
}

impl SqliteBackend {
    /// Time:  O(1) per connection (schema is constant-size)
    /// Space: O(1)
    pub async fn new<P: AsRef<Path>>(path: P, max_connections: Option<u32>) -> StorageResult<Self> {
        let path = path.as_ref().to_path_buf();
        let in_memory = path == Path::new(":memory:");

        let max_conn = if in_memory {
            1
        } else {
            max_connections.unwrap_or(8).max(1)
        };

        if !in_memory && let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| mem_err("create_dir_all", e))?;
        }

        let mut options = SqliteConnectOptions::new()
            .busy_timeout(Duration::from_secs(30))
            .foreign_keys(true)
            .statement_cache_capacity(128);

        options = if in_memory {
            options.in_memory(true)
        } else {
            options.filename(&path).create_if_missing(true)
        };

        if !in_memory {
            options = options
                .journal_mode(SqliteJournalMode::Wal)
                .synchronous(SqliteSynchronous::Normal);
        }

        let pool = SqlitePoolOptions::new()
            .max_connections(max_conn)
            .acquire_timeout(Duration::from_secs(30))
            .after_connect(move |conn, _meta| {
                Box::pin(async move {
                    // Reborrow `conn` for each call; do not move it.
                    sqlx::query("PRAGMA foreign_keys = ON;")
                        .execute(&mut *conn)
                        .await?;

                    if !in_memory {
                        sqlx::query("PRAGMA journal_mode = WAL;")
                            .execute(&mut *conn)
                            .await?;
                        sqlx::query("PRAGMA synchronous = NORMAL;")
                            .execute(&mut *conn)
                            .await?;
                    }

                    sqlx::query(SCHEMA_KV_STORE).execute(&mut *conn).await?;

                    Ok(())
                })
            })
            .connect_with(options)
            .await
            .map_err(|e| mem_err("pool connect", e))?;

        Ok(Self { pool, path })
    }

    pub async fn in_memory() -> StorageResult<Self> {
        // Pool will be forced to 1 connection for correctness.
        Self::new(":memory:", Some(4)).await
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn vacuum(&self) -> StorageResult<()> {
        sqlx::query("VACUUM")
            .execute(&self.pool)
            .await
            .map_err(|e| mem_err("vacuum", e))?;
        Ok(())
    }

    pub async fn optimize(&self) -> StorageResult<()> {
        sqlx::query("PRAGMA optimize")
            .execute(&self.pool)
            .await
            .map_err(|e| mem_err("optimize", e))?;
        Ok(())
    }
}

#[async_trait]
impl StorageBackend for SqliteBackend {
    async fn put(&self, key: &str, value: Value) -> StorageResult<()> {
        let value_json =
            serde_json::to_string(&value).map_err(|e| ser_err("serialize value", e))?;

        sqlx::query(
            r#"
            INSERT INTO kv_store (key, value, updated_at)
            VALUES (?1, ?2, unixepoch())
            ON CONFLICT(key) DO UPDATE
              SET value = excluded.value,
                  updated_at = unixepoch();
            "#,
        )
        .bind(key)
        .bind(value_json)
        .execute(&self.pool)
        .await
        .map_err(|e| mem_err("put", e))?;

        Ok(())
    }

    async fn get(&self, key: &str) -> StorageResult<Option<Value>> {
        let value_json =
            sqlx::query_scalar::<_, String>("SELECT value FROM kv_store WHERE key = ?1")
                .bind(key)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| mem_err("get", e))?;

        value_json
            .map(|s| serde_json::from_str::<Value>(&s).map_err(|e| ser_err("deserialize value", e)))
            .transpose()
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        sqlx::query("DELETE FROM kv_store WHERE key = ?1")
            .bind(key)
            .execute(&self.pool)
            .await
            .map_err(|e| mem_err("delete", e))?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        let exists = sqlx::query_scalar::<_, i64>(
            "SELECT EXISTS(SELECT 1 FROM kv_store WHERE key = ?1 LIMIT 1)",
        )
        .bind(key)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| mem_err("exists", e))?;

        Ok(exists != 0)
    }

    async fn list_keys(&self) -> StorageResult<Vec<String>> {
        sqlx::query_scalar::<_, String>("SELECT key FROM kv_store ORDER BY key")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| mem_err("list_keys", e))
    }

    async fn search(&self, query: &str, limit: usize) -> StorageResult<Vec<SearchResult>> {
        if limit == 0 || query.is_empty() {
            return Ok(Vec::new());
        }

        let escaped = escape_like(query);
        let pattern = format!("%{escaped}%");
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);

        let rows: Vec<(String, String)> = sqlx::query_as(
            r#"
            SELECT key, value
            FROM kv_store
            WHERE key LIKE ?1 ESCAPE '\'
            ORDER BY key
            LIMIT ?2;
            "#,
        )
        .bind(pattern)
        .bind(limit_i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| mem_err("search", e))?;

        let mut results = Vec::with_capacity(rows.len());
        for (key, value_json) in rows {
            let value = serde_json::from_str::<Value>(&value_json)
                .map_err(|e| ser_err("deserialize value", e))?;

            results.push(SearchResult {
                key,
                value,
                score: 1.0,
            });
        }

        Ok(results)
    }

    async fn clear(&self) -> StorageResult<()> {
        sqlx::query("DELETE FROM kv_store")
            .execute(&self.pool)
            .await
            .map_err(|e| mem_err("clear", e))?;
        Ok(())
    }

    async fn stats(&self) -> StorageResult<BackendStats> {
        let total_keys = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM kv_store")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| mem_err("stats count", e))?;

        let page_count = sqlx::query_scalar::<_, i64>("PRAGMA page_count")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| mem_err("stats page_count", e))?;

        let page_size = sqlx::query_scalar::<_, i64>("PRAGMA page_size")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| mem_err("stats page_size", e))?;

        let total_size_bytes = (page_count as u128)
            .saturating_mul(page_size as u128)
            .min(usize::MAX as u128) as usize;

        Ok(BackendStats {
            total_keys: total_keys.max(0) as usize,
            total_size_bytes,
            backend_type: SQLITE_SPACE.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task::JoinSet;

    #[tokio::test]
    async fn test_sqlite_basic_operations() -> StorageResult<()> {
        let backend = SqliteBackend::in_memory().await?;

        backend
            .put("key1", Value::String("value1".to_string()))
            .await?;
        let result = backend.get("key1").await?;
        assert_eq!(result, Some(Value::String("value1".to_string())));

        assert!(backend.exists("key1").await?);
        assert!(!backend.exists("key2").await?);

        backend.delete("key1").await?;
        assert!(!backend.exists("key1").await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_persistence() -> StorageResult<()> {
        let temp_file = std::env::temp_dir().join("test_apxm_storage.db");

        {
            let backend = SqliteBackend::new(&temp_file, Some(4)).await?;
            backend
                .put(
                    "persistent_key",
                    Value::String("persistent_value".to_string()),
                )
                .await?;
        }

        {
            let backend = SqliteBackend::new(&temp_file, Some(4)).await?;
            let result = backend.get("persistent_key").await?;
            assert_eq!(result, Some(Value::String("persistent_value".to_string())));
        }

        let _ = std::fs::remove_file(&temp_file);
        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_search() -> StorageResult<()> {
        let backend = SqliteBackend::in_memory().await?;

        backend
            .put("user:1", Value::String("alice".to_string()))
            .await?;
        backend
            .put("user:2", Value::String("bob".to_string()))
            .await?;
        backend
            .put("post:1", Value::String("hello".to_string()))
            .await?;

        let results = backend.search("user", 10).await?;
        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_stats() -> StorageResult<()> {
        let backend = SqliteBackend::in_memory().await?;

        backend
            .put("key1", Value::String("value1".to_string()))
            .await?;
        backend
            .put(
                "key2",
                Value::Number(apxm_core::types::values::Number::Integer(42)),
            )
            .await?;

        let stats = backend.stats().await?;
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.backend_type, "sqlite");

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_clear() -> StorageResult<()> {
        let backend = SqliteBackend::in_memory().await?;

        backend
            .put("key1", Value::String("value1".to_string()))
            .await?;
        backend
            .put("key2", Value::String("value2".to_string()))
            .await?;

        backend.clear().await?;

        let stats = backend.stats().await?;
        assert_eq!(stats.total_keys, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_sqlite_concurrent_access() -> StorageResult<()> {
        let backend = SqliteBackend::in_memory().await?;

        let mut tasks = JoinSet::new();
        for i in 0..10 {
            let b = backend.clone();
            tasks.spawn(async move {
                b.put(&format!("key{i}"), Value::String(format!("value{i}")))
                    .await
            });
        }

        while let Some(joined) = tasks.join_next().await {
            match joined {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(mem_err("join task", e)),
            }
        }

        let stats = backend.stats().await?;
        assert_eq!(stats.total_keys, 10);

        Ok(())
    }
}
