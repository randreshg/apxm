//! Operation Metadata — Deleted (Deduplicated)
//!
//! This module previously contained `OperationMetadata` with 32 static instances
//! that duplicated field definitions and `needs_submission` values from
//! `OperationSpec` in `definitions.rs`.
//!
//! All consumers should use `get_operation_spec()` and `OperationSpec` instead.
//! The `OperationSpec` struct already provides:
//!   - `fields` — required and optional field definitions
//!   - `needs_submission` — whether async submission is needed
//!   - `get_field(name)` — field lookup by name
//!   - `required_fields()` — iterator over required fields
//!   - `op_type.mlir_mnemonic()` — the MLIR name (was `metadata.name`)
