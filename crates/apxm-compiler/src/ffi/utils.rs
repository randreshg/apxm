//! Module containing utility functions for handling FFI results.

use crate::ffi;
use apxm_core::error::builder::ErrorBuilder;
use apxm_core::error::compiler::{CompilerError, Result};

/// Handles a null result from an FFI function.
pub fn handle_null_result<T>(ptr: *mut T, context: &str) -> Result<*mut T> {
    if !ptr.is_null() {
        return Ok(ptr);
    }

    let errors = ffi::collect_errors();
    if let Some(first_error) = errors.into_iter().next() {
        return Err(CompilerError::Compilation(Box::new(first_error)));
    }

    Err(CompilerError::Internal(Box::new(ErrorBuilder::internal(
        format!("{}: operation returned null", context),
    ))))
}

/// Handles a boolean result from an FFI function.
pub fn handle_bool_result(success: bool, context: &str) -> Result<()> {
    if success {
        return Ok(());
    }

    let errors = ffi::collect_errors();
    if let Some(first_error) = errors.into_iter().next() {
        return Err(CompilerError::Compilation(Box::new(first_error)));
    }

    Err(CompilerError::Internal(Box::new(ErrorBuilder::internal(
        format!("{}: operation failed", context),
    ))))
}
