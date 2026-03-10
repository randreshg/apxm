# apxm-artifact

Binary artifact format for compiled APXM programs.

## Overview

`apxm-artifact` handles serialization and deserialization of compiled programs. Artifacts are binary files containing:
- Magic header (`APXM`)
- Version information
- Metadata (module name, timestamps, compiler version)
- Execution DAG
- Optional sections

## Responsibilities

- Serialize/deserialize `ExecutionDag` binaries
- Maintain artifact metadata and integrity hashes
- Provide portable artifact I/O for runtime loading

## How It Fits

Artifacts are the contract between the compiler and the runtime: the compiler
serializes an `ExecutionDag` into a portable binary, and the runtime executes it.

## Format

```
┌────────────────────────────────────────┐
│  Magic: "APXM" (4 bytes)               │
│  Version: u32                          │
│  Flags: u32                            │
├────────────────────────────────────────┤
│  Metadata                              │
│  ├── module_name: String               │
│  ├── created_at: DateTime              │
│  └── compiler_version: String          │
├────────────────────────────────────────┤
│  ExecutionDag                          │
│  ├── nodes: Vec<Node>                  │
│  └── edges: Vec<Edge>                  │
├────────────────────────────────────────┤
│  Sections (optional)                   │
│  └── Vec<ArtifactSection>              │
├────────────────────────────────────────┤
│  Payload Hash (Blake3)                 │
└────────────────────────────────────────┘
```

## Key Types

- `Artifact` - Main artifact struct
- `ArtifactMetadata` - Module metadata
- `ArtifactSection` - Additional data sections
- `ArtifactError` - Error type

## Usage

### Creating Artifacts

```rust
use apxm_artifact::{Artifact, ArtifactMetadata};
use apxm_core::types::execution::ExecutionDag;

let dag = ExecutionDag::new("my_module");

let artifact = Artifact::new(
    ArtifactMetadata {
        module_name: "my_module".into(),
        created_at: chrono::Utc::now(),
        compiler_version: "0.1.0".into(),
    },
    dag,
);
```

### Serialization

```rust
// To bytes
let bytes = artifact.to_bytes()?;

// To file
artifact.write_to_path("program.apxm")?;
```

### Deserialization

```rust
// From bytes
let artifact = Artifact::from_bytes(&bytes)?;

// From file
let artifact = Artifact::read_from_path("program.apxm")?;

// Extract DAG for execution
let dag = artifact.into_dag();
```

### Hash Verification

```rust
// Get payload hash (Blake3)
let hash = artifact.payload_hash();
println!("Hash: {}", hex::encode(hash));
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| apxm-core | ExecutionDag, Node, Edge types |

## Building

```bash
cargo build -p apxm-artifact
```

## Testing

```bash
cargo test -p apxm-artifact
```
