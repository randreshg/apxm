# Compiler ↔ Runtime Contract

This document defines the contract between the AIS compiler and runtime. It is the
single source of truth for what the compiler must emit and what the runtime must accept.

## Scope

- AIS operations are defined in `apxm-ais`
- The compiler emits artifacts (`apxm-artifact`)
- The runtime executes artifacts (`apxm-runtime`)

## Guarantees

Compiler guarantees:
- Emits a valid `Artifact` with required metadata
- Produces a DAG that passes `ExecutionDag::validate()`
- Uses AIS operation metadata from `apxm-ais` (no ad‑hoc ops)

Runtime guarantees:
- Rejects malformed artifacts with clear errors
- Executes valid DAGs deterministically under the scheduler
- Provides metrics only when `metrics` feature is enabled (zero overhead otherwise)

## Artifact Contract

Artifacts must include:
- Magic header and version
- `ArtifactMetadata` (module name, timestamp, compiler version)
- Serialized `ExecutionDag`
- Optional sections (future‑compatible)

The runtime treats artifacts as immutable and does not mutate metadata or DAG structure.

## Operation Contract

Operations:
- Are defined solely in `apxm-ais`
- Must match the TableGen definitions generated from `apxm-ais`
- Must have the required fields populated by the compiler

`COMMUNICATE` remains stubbed at runtime. If emitted by the compiler, runtime behavior
is documented as a stub in the paper and system docs.

## Versioning

- `ArtifactMetadata.compiler_version` is used for compatibility checks.
- Breaking changes to AIS ops or artifact layout require a version bump.
