//! Embedding generation for hybrid search.
//!
//! When the `embeddings` feature is enabled, uses FastEmbed (ONNX) for local
//! embedding generation. Otherwise, provides a no-op implementation.

use crate::storage::StorageResult;
use apxm_core::error::RuntimeError;

/// Trait for embedding generators used by storage backends.
pub trait Embedder: Send + Sync {
    /// Generate embeddings for multiple texts.
    fn embed(&self, texts: &[&str]) -> StorageResult<Vec<Vec<f32>>>;

    /// Generate embedding for a single text.
    fn embed_one(&self, text: &str) -> StorageResult<Vec<f32>> {
        let embeddings = self.embed(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| RuntimeError::Memory {
                message: "No embedding returned".to_string(),
                space: Some("embedder".to_string()),
            })
    }

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(feature = "embeddings")]
mod local {
    use super::*;
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
    use std::sync::Mutex;

    /// Local embedder using FastEmbed (ONNX-based, no Python).
    pub struct LocalEmbedder {
        model: Mutex<TextEmbedding>,
        dimension: usize,
    }

    impl LocalEmbedder {
        /// Create a new embedder with BGE-small-en-v1.5 (384 dimensions, fast).
        pub fn new() -> StorageResult<Self> {
            let options =
                InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true);
            let model = TextEmbedding::try_new(options).map_err(|e| RuntimeError::Memory {
                message: format!("Failed to initialize FastEmbed: {e}"),
                space: Some("embedder".to_string()),
            })?;
            Ok(Self {
                model: Mutex::new(model),
                dimension: 384,
            })
        }
    }

    impl Embedder for LocalEmbedder {
        fn embed(&self, texts: &[&str]) -> StorageResult<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
            let model = self.model.lock().map_err(|e| RuntimeError::Memory {
                message: format!("embedder lock poisoned: {e}"),
                space: Some("embedder".to_string()),
            })?;
            model
                .embed(texts_owned, None)
                .map_err(|e| RuntimeError::Memory {
                    message: format!("embedding failed: {e}"),
                    space: Some("embedder".to_string()),
                })
        }

        fn dimension(&self) -> usize {
            self.dimension
        }
    }
}

#[cfg(feature = "embeddings")]
pub use local::LocalEmbedder;
