//! File-based artifact cache keyed by SHA-256 of canonicalized graph JSON.
//!
//! Cache location: `~/.cache/apxm/artifacts/<hash>.apxmobj`

use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;

use crate::error::DriverError;

/// Returns `true` when the cache is explicitly disabled via `APXM_NO_CACHE=1`.
pub fn cache_disabled() -> bool {
    env::var("APXM_NO_CACHE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Compute SHA-256 hex digest of canonicalized (sorted-keys) graph JSON.
pub fn graph_hash(graph_json: &str) -> Result<String, DriverError> {
    // Re-parse and re-serialize with sorted keys for deterministic hashing
    let value: serde_json::Value =
        serde_json::from_str(graph_json).map_err(|e| DriverError::Driver(e.to_string()))?;
    let canonical = serde_json::to_string(&value)
        .map_err(|e| DriverError::Driver(format!("canonical JSON error: {e}")))?;

    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Return the cache directory, creating it if necessary.
fn cache_dir() -> Result<PathBuf, DriverError> {
    let base = dirs::cache_dir()
        .or_else(dirs::home_dir)
        .ok_or_else(|| DriverError::Driver("cannot determine cache directory".into()))?;
    let dir = base.join("apxm").join("artifacts");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Try to load a cached artifact by its hash.  Returns `None` on miss.
pub fn load_cached(hash: &str) -> Result<Option<Vec<u8>>, DriverError> {
    if cache_disabled() {
        return Ok(None);
    }
    let path = cache_dir()?.join(format!("{hash}.apxmobj"));
    if path.exists() {
        Ok(Some(fs::read(&path)?))
    } else {
        Ok(None)
    }
}

/// Store compiled artifact bytes under the given hash.
pub fn store_cached(hash: &str, artifact_bytes: &[u8]) -> Result<(), DriverError> {
    if cache_disabled() {
        return Ok(());
    }
    let path = cache_dir()?.join(format!("{hash}.apxmobj"));
    fs::write(&path, artifact_bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let json_a = r#"{"name":"test","nodes":[]}"#;
        let json_b = r#"{"nodes":[],"name":"test"}"#;
        // Both represent the same JSON object; serde_json::Value sorts keys
        // alphabetically on serialization so they must hash identically.
        assert_eq!(graph_hash(json_a).unwrap(), graph_hash(json_b).unwrap());
    }

    #[test]
    fn cache_disabled_env() {
        // Default: not disabled
        assert!(!cache_disabled() || env::var("APXM_NO_CACHE").is_ok());
    }
}
