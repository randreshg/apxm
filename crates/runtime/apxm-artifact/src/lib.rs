use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use apxm_core::types::execution::ExecutionDag;
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod wire;

const MAGIC: &[u8; 4] = b"APXM";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 4 + 4 + 8 + 32 + 4;

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] Box<bincode::ErrorKind>),
    #[error("Invalid artifact header")]
    InvalidHeader,
    #[error("Artifact version mismatch: {0}")]
    VersionMismatch(u32),
    #[error("Artifact hash mismatch")]
    HashMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub module_name: Option<String>,
    pub created_at: u64,
    pub compiler_version: String,
}

impl ArtifactMetadata {
    pub fn new(module_name: Option<String>, compiler_version: impl Into<String>) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            module_name,
            created_at,
            compiler_version: compiler_version.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArtifactSection {
    pub kind: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArtifactPayload {
    metadata: ArtifactMetadata,
    dags: Vec<wire::WireDag>,
    sections: Vec<ArtifactSection>,
}

#[derive(Debug, Clone)]
pub struct Artifact {
    metadata: ArtifactMetadata,
    dags: Vec<ExecutionDag>,
    sections: Vec<ArtifactSection>,
    flags: u32,
}

impl Artifact {
    pub fn new(metadata: ArtifactMetadata, dags: Vec<ExecutionDag>) -> Self {
        Self {
            metadata,
            dags,
            sections: Vec::new(),
            flags: 0,
        }
    }

    pub fn metadata(&self) -> &ArtifactMetadata {
        &self.metadata
    }

    /// Get all DAGs in the artifact
    pub fn dags(&self) -> &[ExecutionDag] {
        &self.dags
    }

    /// Get the @entry DAG (first with is_entry=true)
    pub fn entry_dag(&self) -> Option<&ExecutionDag> {
        self.dags.iter().find(|d| d.metadata.is_entry)
    }

    /// Get non-entry DAGs (for flow registration)
    pub fn flow_dags(&self) -> impl Iterator<Item = &ExecutionDag> {
        self.dags.iter().filter(|d| !d.metadata.is_entry)
    }

    /// Legacy: Get first DAG (backward compat for single-DAG artifacts)
    pub fn dag(&self) -> &ExecutionDag {
        self.dags.first().expect("Artifact has no DAGs")
    }

    pub fn sections(&self) -> &[ArtifactSection] {
        &self.sections
    }

    /// Consume artifact and return all DAGs
    pub fn into_dags(self) -> Vec<ExecutionDag> {
        self.dags
    }

    /// Legacy: Consume and return entry DAG only
    pub fn into_dag(self) -> ExecutionDag {
        self.entry_dag()
            .cloned()
            .or_else(|| self.dags.into_iter().next())
            .expect("Artifact has no DAGs")
    }

    fn payload(&self) -> Result<Vec<u8>, Box<bincode::ErrorKind>> {
        let payload = ArtifactPayload {
            metadata: self.metadata.clone(),
            dags: self.dags.iter().map(wire::WireDag::from_execution_dag).collect(),
            sections: self.sections.clone(),
        };
        bincode::serialize(&payload)
    }

    pub fn payload_hash(&self) -> ArtifactResult<[u8; 32]> {
        let payload = self.payload()?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(hasher.finalize().as_bytes());
        Ok(bytes)
    }

    pub fn to_bytes(&self) -> ArtifactResult<Vec<u8>> {
        let payload = self.payload()?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let digest = hasher.finalize();

        let mut bytes = Vec::with_capacity(HEADER_SIZE + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(digest.as_bytes());
        bytes.extend_from_slice(&self.flags.to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> ArtifactResult<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(ArtifactError::InvalidHeader);
        }

        if &bytes[..4] != MAGIC {
            return Err(ArtifactError::InvalidHeader);
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(ArtifactError::VersionMismatch(version));
        }

        let payload_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let hash = &bytes[16..48];
        let flags = u32::from_le_bytes(bytes[48..52].try_into().unwrap());

        if bytes.len() < HEADER_SIZE + payload_len {
            return Err(ArtifactError::InvalidHeader);
        }

        let payload = &bytes[HEADER_SIZE..HEADER_SIZE + payload_len];
        let mut hasher = Hasher::new();
        hasher.update(payload);
        if hasher.finalize().as_bytes() != hash {
            return Err(ArtifactError::HashMismatch);
        }

        let ArtifactPayload {
            metadata,
            dags,
            sections,
        } = bincode::deserialize(payload)?;

        let dags = dags.into_iter().map(|d| d.into_execution_dag()).collect();

        Ok(Self {
            metadata,
            dags,
            sections,
            flags,
        })
    }

    pub fn write_to_path<P: AsRef<Path>>(&self, path: P) -> ArtifactResult<()> {
        let mut file = std::fs::File::create(path)?;
        let bytes = self.to_bytes()?;
        file.write_all(&bytes)?;
        Ok(())
    }

    pub fn read_from_path<P: AsRef<Path>>(path: P) -> ArtifactResult<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }
}

pub type ArtifactResult<T> = std::result::Result<T, ArtifactError>;

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::AISOperationType;
    use apxm_core::types::execution::{DagMetadata, ExecutionDag, Node};
    use std::collections::HashMap;

    fn sample_dag() -> ExecutionDag {
        ExecutionDag {
            nodes: vec![Node {
                id: 1,
                op_type: AISOperationType::Inv,
                attributes: HashMap::new(),
                input_tokens: vec![],
                output_tokens: vec![1],
                metadata: Default::default(),
            }],
            edges: vec![],
            entry_nodes: vec![1],
            exit_nodes: vec![1],
            metadata: DagMetadata {
                name: Some("test".into()),
                is_entry: false,
            },
        }
    }

    #[test]
    fn artifact_round_trip() {
        let dag = sample_dag();
        let metadata = ArtifactMetadata::new(Some("sample".into()), "test-compiler");
        let artifact = Artifact::new(metadata.clone(), vec![dag.clone()]);

        let bytes = artifact.to_bytes().expect("serialize artifact");
        let stored_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        assert_eq!(stored_len, bytes.len() - HEADER_SIZE);
        let payload = &bytes[HEADER_SIZE..];
        let expected_payload = bincode::serialize(&ArtifactPayload {
            metadata: metadata.clone(),
            dags: vec![wire::WireDag::from_execution_dag(&dag)],
            sections: Vec::new(),
        })
        .unwrap();
        assert_eq!(payload, expected_payload.as_slice());
        let raw: ArtifactPayload = bincode::deserialize(payload).expect("payload deserialize");
        assert!(raw.metadata.module_name.as_deref() == Some("sample"));
        let decoded = Artifact::from_bytes(&bytes).expect("deserialize artifact");

        assert_eq!(decoded.metadata.module_name, metadata.module_name);
        assert_eq!(decoded.metadata.compiler_version, metadata.compiler_version);
        assert_eq!(decoded.dag().nodes.len(), dag.nodes.len());
    }

    #[test]
    fn detects_invalid_hash() {
        let dag = sample_dag();
        let metadata = ArtifactMetadata::new(Some("sample".into()), "test-compiler");
        let artifact = Artifact::new(metadata, vec![dag]);
        let mut bytes = artifact.to_bytes().expect("serialize artifact");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let err = Artifact::from_bytes(&bytes).expect_err("expected hash mismatch");
        assert!(matches!(err, ArtifactError::HashMismatch));
    }
}
