//! Extension trait to add code emission to operation metadata

use crate::codegen::emitter::RustEmitter;
use apxm_core::error::builder::ErrorBuilder;
use apxm_core::error::codes::ErrorCode;
use apxm_core::error::compiler::{CompilerError, Result};
use apxm_core::types::OperationMetadata;
use std::collections::HashMap;

/// Extension trait to add code emission to operation metadata
pub trait OperationEmit {
    fn emit(
        &self,
        emitter: &mut RustEmitter,
        op_id: &str,
        args: &HashMap<&str, String>,
    ) -> Result<()>;
}

impl OperationEmit for OperationMetadata {
    /// Emit the operation metadata to the given emitter
    fn emit(
        &self,
        emitter: &mut RustEmitter,
        op_id: &str,
        args: &HashMap<&str, String>,
    ) -> Result<()> {
        emitter.emit_line(&format!("let {} = {}Op::new()", op_id, self.name));
        emitter.indent();

        for field in self.fields {
            if let Some(value) = args.get(field.name) {
                emitter.emit_line(&format!(".with_{}({})", field.name, value));
            } else if field.required {
                let err = ErrorBuilder::generic(
                    ErrorCode::InternalError,
                    format!(
                        "Missing required field '{}' for operation '{}'",
                        field.name, self.name
                    ),
                );
                return Err(CompilerError::InvalidInput(Box::new(err)));
            }
        }

        emitter.emit_line(".build()?;");
        emitter.dedent();

        if self.needs_submission {
            emitter.emit_line(&format!("executor.submit_operation({}).await?;", op_id));
        }

        emitter.blank_line();
        Ok(())
    }
}

pub fn generate_op_id(counter: &mut usize) -> String {
    let id = *counter;
    *counter += 1;
    format!("op_{}", id)
}
