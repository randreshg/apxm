//! Operation metadata registry.
//!
//! Contains metadata definitions for AIS operations used across compiler, runtime, and other tools.

/// A field in an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationField {
    /// Field name
    pub name: &'static str,
    /// Whether the field is required
    pub required: bool,
}

impl OperationField {
    /// Create a required field
    pub const fn required(name: &'static str) -> Self {
        Self {
            name,
            required: true,
        }
    }

    /// Create an optional field
    pub const fn optional(name: &'static str) -> Self {
        Self {
            name,
            required: false,
        }
    }
}

/// Metadata about an AIS operation
#[derive(Debug, Clone, Copy)]
pub struct OperationMetadata {
    /// Operation name (e.g., "QMem", "UMem")
    pub name: &'static str,
    /// Fields expected by this operation
    pub fields: &'static [OperationField],
    /// Whether this operation needs to be submitted to executor
    pub needs_submission: bool,
}

impl OperationMetadata {
    /// Create new operation metadata
    pub const fn new(
        name: &'static str,
        fields: &'static [OperationField],
        needs_submission: bool,
    ) -> Self {
        Self {
            name,
            fields,
            needs_submission,
        }
    }

    /// Get a field by name
    pub fn get_field(&self, name: &str) -> Option<&OperationField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Check if a field is required
    pub fn is_field_required(&self, name: &str) -> bool {
        self.get_field(name).map(|f| f.required).unwrap_or(false)
    }

    /// Get all required fields
    pub fn required_fields(&self) -> impl Iterator<Item = &OperationField> {
        self.fields.iter().filter(|f| f.required)
    }

    /// Get all optional fields
    pub fn optional_fields(&self) -> impl Iterator<Item = &OperationField> {
        self.fields.iter().filter(|f| !f.required)
    }
}

// Operation metadata constants
pub mod operations {
    use super::{OperationField, OperationMetadata};

    // Memory Operations
    pub const QMEM: OperationMetadata = OperationMetadata::new(
        "QMem",
        &[
            OperationField::required("query"),
            OperationField::optional("memory_tier"),
        ],
        true,
    );

    pub const UMEM: OperationMetadata = OperationMetadata::new(
        "UMem",
        &[
            OperationField::required("key"),
            OperationField::required("value"),
            OperationField::optional("memory_tier"),
        ],
        true,
    );

    // Reasoning Operations
    pub const RSN: OperationMetadata = OperationMetadata::new(
        "Rsn",
        &[
            OperationField::required("prompt"),
            OperationField::optional("model"),
            OperationField::optional("context"),
        ],
        true,
    );

    pub const REFLECT: OperationMetadata = OperationMetadata::new(
        "Reflect",
        &[
            OperationField::required("trace_query"),
            OperationField::optional("reflection_prompt"),
        ],
        true,
    );

    pub const VERIFY: OperationMetadata = OperationMetadata::new(
        "Verify",
        &[
            OperationField::required("claim"),
            OperationField::required("evidence"),
        ],
        true,
    );

    // Execution Operations
    pub const INV: OperationMetadata = OperationMetadata::new(
        "Inv",
        &[
            OperationField::required("capability"),
            OperationField::optional("parameters"),
        ],
        true,
    );

    pub const PLAN: OperationMetadata = OperationMetadata::new(
        "Plan",
        &[
            OperationField::required("goal"),
            OperationField::optional("constraints"),
        ],
        true,
    );

    pub const EXC: OperationMetadata = OperationMetadata::new(
        "Exc",
        &[
            OperationField::required("code"),
            OperationField::optional("sandbox_config"),
        ],
        true,
    );

    // Coordination Operations
    pub const FENCE: OperationMetadata =
        OperationMetadata::new("Fence", &[OperationField::optional("ordering")], true);

    pub const WAIT_ALL: OperationMetadata =
        OperationMetadata::new("WaitAll", &[OperationField::required("tokens")], true);

    pub const MERGE: OperationMetadata =
        OperationMetadata::new("Merge", &[OperationField::required("tokens")], true);

    // Communication Operations
    pub const COMMUNICATE: OperationMetadata = OperationMetadata::new(
        "Communicate",
        &[
            OperationField::required("target_agent"),
            OperationField::required("message"),
        ],
        true,
    );

    pub const ERR: OperationMetadata =
        OperationMetadata::new("Err", &[OperationField::required("error_handler")], true);

    // Literals
    pub const CONST_STR: OperationMetadata =
        OperationMetadata::new("ConstStr", &[OperationField::required("value")], false);
}

/// Registry of all operation metadata
pub static OPERATION_REGISTRY: &[&OperationMetadata] = &[
    &operations::QMEM,
    &operations::UMEM,
    &operations::RSN,
    &operations::REFLECT,
    &operations::VERIFY,
    &operations::INV,
    &operations::PLAN,
    &operations::EXC,
    &operations::FENCE,
    &operations::WAIT_ALL,
    &operations::MERGE,
    &operations::COMMUNICATE,
    &operations::CONST_STR,
    &operations::ERR,
];

/// Find operation metadata by name
pub fn find_operation(name: &str) -> Option<&'static OperationMetadata> {
    OPERATION_REGISTRY
        .iter()
        .find(|op| op.name == name)
        .copied()
}
