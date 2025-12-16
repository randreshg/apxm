//! redb-backed KV storage backend (pure Rust, embedded, ACID).
//!
//! Design goals:
//! - Async-safe: all blocking redb operations run in `spawn_blocking`.
//! - Performance: O(log N) for point ops; single-transaction scans for search/list.
//! - Security: parameterized statements are irrelevant (not SQL); no unsafe; optional 2PC default.
//!
//! Notes:
//! - redb supports many concurrent readers and a single writer (write transactions serialize).
//! - Table is created lazily (first open_table in a write txn creates it), but we also warm-create it in `new`.

use crate::{SearchResult, StorageBackend, StorageResult, backend::BackendStats};
use apxm_core::{error::RuntimeError, types::values::Value};
use async_trait::async_trait;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::{
    fmt::Display,
    path::{Path, PathBuf},
    sync::Arc,
};

const BACKEND_TYPE: &str = "redb";
const TABLE_NAME: &str = "kv_store";
const DEFAULT_CACHE_SIZE_BYTES: usize = 64 * 1024 * 1024; // 64MiB; adjust to your bundle footprint.

const KV_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new(TABLE_NAME);

#[inline]
fn mem_err(context: &'static str, e: impl Display) -> RuntimeError {
    RuntimeError::Memory {
        message: format!("{BACKEND_TYPE} {context}: {e}"),
        space: Some(BACKEND_TYPE.to_string()),
    }
}

#[inline]
fn ser_err(context: &'static str, e: impl Display) -> RuntimeError {
    RuntimeError::Serialization(format!("{BACKEND_TYPE} {context}: {e}"))
}

#[inline]
fn join_err(context: &'static str, e: tokio::task::JoinError) -> RuntimeError {
    // JoinError implements Display; include context to keep logs actionable.
    mem_err(context, e)
}

async fn blocking<T, F>(context: &'static str, f: F) -> StorageResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, RuntimeError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| join_err(context, e))?
}

#[derive(Clone)]
pub struct RedbBackend {
    db: Arc<Database>,
    path: PathBuf,
    two_phase_commit: bool,
}

impl RedbBackend {
    /// Create or open a redb database at `path`.
    ///
    /// Time:  O(1) (open) + O(1) (table creation)
    /// Space: O(1)
    pub async fn new<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        Self::with_options(path, Some(DEFAULT_CACHE_SIZE_BYTES), true).await
    }

    /// Same as `new`, with explicit options.
    ///
    /// - `cache_size_bytes`: memory budget for redb cache (smaller default fits bundles better).
    /// - `two_phase_commit`: enables 2PC on each write transaction (more robust; slower commits).
    pub async fn with_options<P: AsRef<Path>>(
        path: P,
        cache_size_bytes: Option<usize>,
        two_phase_commit: bool,
    ) -> StorageResult<Self> {
        let path = path.as_ref().to_path_buf();

        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| mem_err("create_dir_all", e))?;
        }

        let path_for_open = path.clone();
        let cache = cache_size_bytes.unwrap_or(DEFAULT_CACHE_SIZE_BYTES);

        let db = blocking("open", move || {
            let mut builder = Database::builder();
            builder.set_cache_size(cache);

            let db = builder
                .create(&path_for_open)
                .map_err(|e| mem_err("create/open", e))?;

            // Warm-create the table so first ops don't pay table creation costs.
            let write_txn = db.begin_write().map_err(|e| mem_err("begin_write", e))?;
            {
                let _table = write_txn
                    .open_table(KV_TABLE)
                    .map_err(|e| mem_err("open_table", e))?;
            }
            write_txn.commit().map_err(|e| mem_err("commit", e))?;

            Ok(db)
        })
        .await?;

        Ok(Self {
            db: Arc::new(db),
            path,
            two_phase_commit,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[async_trait]
impl StorageBackend for RedbBackend {
    /// Put is a single write transaction:
    /// Time:  O(log N) + O(S)  (B-tree insert + JSON serialize)
    /// Space: O(S)
    async fn put(&self, key: &str, value: Value) -> StorageResult<()> {
        let key = key.to_owned();
        let bytes = serde_json::to_vec(&value).map_err(|e| ser_err("serialize", e))?;
        let db = Arc::clone(&self.db);
        let two_pc = self.two_phase_commit;

        blocking("put", move || {
            let mut txn = db.begin_write().map_err(|e| mem_err("begin_write", e))?;
            if two_pc {
                txn.set_two_phase_commit(true);
            }

            {
                let mut table = txn
                    .open_table(KV_TABLE)
                    .map_err(|e| mem_err("open_table", e))?;
                table
                    .insert(key.as_str(), bytes.as_slice())
                    .map_err(|e| mem_err("insert", e))?;
            }

            txn.commit().map_err(|e| mem_err("commit", e))?;
            Ok(())
        })
        .await
    }

    /// Get is a read transaction:
    /// Time:  O(log N) + O(S) (B-tree lookup + JSON deserialize)
    /// Space: O(S)
    async fn get(&self, key: &str) -> StorageResult<Option<Value>> {
        let key = key.to_owned();
        let db = Arc::clone(&self.db);

        let opt_bytes: Option<Vec<u8>> = blocking("get", move || {
            let txn = db.begin_read().map_err(|e| mem_err("begin_read", e))?;
            let table = txn
                .open_table(KV_TABLE)
                .map_err(|e| mem_err("open_table", e))?;

            let guard = table.get(key.as_str()).map_err(|e| mem_err("get", e))?;
            Ok(guard.map(|g| g.value().to_vec()))
        })
        .await?;

        opt_bytes
            .map(|b| serde_json::from_slice::<Value>(&b).map_err(|e| ser_err("deserialize", e)))
            .transpose()
    }

    /// Delete:
    /// Time:  O(log N)
    /// Space: O(1)
    async fn delete(&self, key: &str) -> StorageResult<()> {
        let key = key.to_owned();
        let db = Arc::clone(&self.db);
        let two_pc = self.two_phase_commit;

        blocking("delete", move || {
            let mut txn = db.begin_write().map_err(|e| mem_err("begin_write", e))?;
            if two_pc {
                txn.set_two_phase_commit(true);
            }

            {
                let mut table = txn
                    .open_table(KV_TABLE)
                    .map_err(|e| mem_err("open_table", e))?;
                let _ = table
                    .remove(key.as_str())
                    .map_err(|e| mem_err("remove", e))?;
            }

            txn.commit().map_err(|e| mem_err("commit", e))?;
            Ok(())
        })
        .await
    }

    /// Exists:
    /// Time:  O(log N)
    /// Space: O(1)
    async fn exists(&self, key: &str) -> StorageResult<bool> {
        let key = key.to_owned();
        let db = Arc::clone(&self.db);

        blocking("exists", move || {
            let txn = db.begin_read().map_err(|e| mem_err("begin_read", e))?;
            let table = txn
                .open_table(KV_TABLE)
                .map_err(|e| mem_err("open_table", e))?;
            let hit = table
                .get(key.as_str())
                .map_err(|e| mem_err("get", e))?
                .is_some();
            Ok(hit)
        })
        .await
    }

    /// List keys (ordered):
    /// Time:  O(N)
    /// Space: O(N)
    async fn list_keys(&self) -> StorageResult<Vec<String>> {
        let db = Arc::clone(&self.db);

        blocking("list_keys", move || {
            let txn = db.begin_read().map_err(|e| mem_err("begin_read", e))?;
            let table = txn
                .open_table(KV_TABLE)
                .map_err(|e| mem_err("open_table", e))?;

            let mut out = Vec::new();
            for item in table.iter().map_err(|e| mem_err("iter", e))? {
                let (k, _v) = item.map_err(|e| mem_err("iter item", e))?;
                out.push(k.value().to_owned());
            }
            Ok(out)
        })
        .await
    }

    /// Substring search on keys, single transaction scan with early stop:
    /// Time:  O(N * C + R * D)
    /// Space: O(R)
    async fn search(&self, query: &str, limit: usize) -> StorageResult<Vec<SearchResult>> {
        if limit == 0 || query.is_empty() {
            return Ok(Vec::new());
        }

        let query = query.to_owned();
        let db = Arc::clone(&self.db);

        blocking("search", move || {
            let txn = db.begin_read().map_err(|e| mem_err("begin_read", e))?;
            let table = txn
                .open_table(KV_TABLE)
                .map_err(|e| mem_err("open_table", e))?;

            let mut results = Vec::with_capacity(limit.min(64));
            for item in table.iter().map_err(|e| mem_err("iter", e))? {
                let (k, v) = item.map_err(|e| mem_err("iter item", e))?;
                let key = k.value();

                if !key.contains(&query) {
                    continue;
                }

                let value = serde_json::from_slice::<Value>(v.value())
                    .map_err(|e| ser_err("deserialize", e))?;

                results.push(SearchResult {
                    key: key.to_owned(),
                    value,
                    score: 1.0,
                });

                if results.len() == limit {
                    break;
                }
            }

            Ok(results)
        })
        .await
    }

    /// Clear all entries:
    /// Time:  O(N)
    /// Space: O(1)
    async fn clear(&self) -> StorageResult<()> {
        let db = Arc::clone(&self.db);
        let two_pc = self.two_phase_commit;

        blocking("clear", move || {
            let mut txn = db.begin_write().map_err(|e| mem_err("begin_write", e))?;
            if two_pc {
                txn.set_two_phase_commit(true);
            }

            {
                let mut table = txn
                    .open_table(KV_TABLE)
                    .map_err(|e| mem_err("open_table", e))?;
                table
                    .retain(|_k, _v| false)
                    .map_err(|e| mem_err("retain", e))?;
            }

            txn.commit().map_err(|e| mem_err("commit", e))?;
            Ok(())
        })
        .await
    }

    /// Backend stats:
    /// - `total_keys`: table entry count
    /// - `total_size_bytes`: stored + metadata + fragmentation (table-level estimate)
    async fn stats(&self) -> StorageResult<BackendStats> {
        let db = Arc::clone(&self.db);

        blocking("stats", move || {
            let txn = db.begin_read().map_err(|e| mem_err("begin_read", e))?;
            let table = txn
                .open_table(KV_TABLE)
                .map_err(|e| mem_err("open_table", e))?;

            let total_keys_u64 = table.len().map_err(|e| mem_err("len", e))?;
            let table_stats = table.stats().map_err(|e| mem_err("table stats", e))?;

            let bytes_u128 = (table_stats.stored_bytes() as u128)
                .saturating_add(table_stats.metadata_bytes() as u128)
                .saturating_add(table_stats.fragmented_bytes() as u128);

            let total_size_bytes = bytes_u128.min(usize::MAX as u128) as usize;

            Ok(BackendStats {
                total_keys: (total_keys_u64 as u128).min(usize::MAX as u128) as usize,
                total_size_bytes,
                backend_type: BACKEND_TYPE.to_string(),
            })
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        // Best-effort cleanup from prior runs.
        let _ = std::fs::remove_file(&p);
        p
    }

    #[tokio::test]
    async fn test_redb_basic_operations() {
        let path = temp_path("test_apxm_storage_redb_basic.db");
        let backend = RedbBackend::with_options(&path, Some(8 * 1024 * 1024), true)
            .await
            .unwrap();

        backend
            .put("key1", Value::String("value1".to_string()))
            .await
            .unwrap();

        let got = backend.get("key1").await.unwrap();
        assert_eq!(got, Some(Value::String("value1".to_string())));

        assert!(backend.exists("key1").await.unwrap());
        assert!(!backend.exists("missing").await.unwrap());

        backend.delete("key1").await.unwrap();
        assert!(!backend.exists("key1").await.unwrap());

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_redb_persistence() {
        let path = temp_path("test_apxm_storage_redb_persist.db");

        {
            let backend = RedbBackend::new(&path).await.unwrap();
            backend
                .put(
                    "persistent_key",
                    Value::String("persistent_value".to_string()),
                )
                .await
                .unwrap();
        }

        {
            let backend = RedbBackend::new(&path).await.unwrap();
            let got = backend.get("persistent_key").await.unwrap();
            assert_eq!(got, Some(Value::String("persistent_value".to_string())));
        }

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_redb_search_and_clear() {
        let path = temp_path("test_apxm_storage_redb_search.db");
        let backend = RedbBackend::new(&path).await.unwrap();

        backend
            .put("user:1", Value::String("alice".to_string()))
            .await
            .unwrap();
        backend
            .put("user:2", Value::String("bob".to_string()))
            .await
            .unwrap();
        backend
            .put("post:1", Value::String("hello".to_string()))
            .await
            .unwrap();

        let results = backend.search("user", 10).await.unwrap();
        assert_eq!(results.len(), 2);

        backend.clear().await.unwrap();
        let stats = backend.stats().await.unwrap();
        assert_eq!(stats.total_keys, 0);

        let _ = std::fs::remove_file(&path);
    }
}
