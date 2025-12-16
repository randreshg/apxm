//! Error handling utilities for the C API
//!
//! Provides safe wrappers around the C error API.

use crate::ffi::raw;
use apxm_core::error::Error;
use apxm_core::error::codes::ErrorCode;

/// Get error count
pub fn error_collector_count() -> usize {
    unsafe { raw::apxm_error_collector_count() }
}

/// Get all errors
///
/// Returns raw pointers that must be freed with `error_free`
pub fn error_collector_get_all() -> Vec<*mut raw::ApxmError> {
    let count = error_collector_count();
    if count == 0 {
        return Vec::new();
    }

    let mut ptrs = vec![std::ptr::null_mut(); count];

    let actual_count = unsafe { raw::apxm_error_collector_get_all(ptrs.as_mut_ptr(), count) };

    ptrs.truncate(actual_count);
    ptrs
}

/// Get first error
///
/// Returns a raw pointer that must be freed with `error_free`
#[allow(dead_code)]
pub fn error_collector_get_first() -> Option<*const raw::ApxmError> {
    let ptr = unsafe { raw::apxm_error_collector_get_first() };
    if ptr.is_null() { None } else { Some(ptr) }
}

/// Free a error allocated by the C API
pub fn error_free(err: *mut raw::ApxmError) {
    if !err.is_null() {
        unsafe {
            raw::apxm_error_free(err);
        }
    }
}

/// Convert C Error to Rust Error
///
/// # Safety
/// The pointer must be valid and non-null.
pub unsafe fn error_from_raw(c_err: &raw::ApxmError) -> Error {
    use apxm_core::error::span::Span;

    // Convert error code with safe fallback
    let code = ErrorCode::from_u32(c_err.code).unwrap_or(ErrorCode::InternalError);

    // Convert message
    let message = unsafe { raw::cstr_or_default(c_err.message, "Unknown error") };

    // Convert span fields
    let file = unsafe { raw::cstr_or_default(c_err.file_path, "<unknown>") };
    let snippet = unsafe { raw::cstr_from_ptr(c_err.snippet) };
    let label = unsafe { raw::cstr_from_ptr(c_err.label) };

    let highlight = if c_err.highlight_start != 0 || c_err.highlight_end != 0 {
        Some((c_err.highlight_start as usize, c_err.highlight_end as usize))
    } else {
        None
    };

    let span = Span {
        file,
        line_start: c_err.file_line as usize,
        col_start: c_err.file_col as usize,
        line_end: c_err.file_line_end as usize,
        col_end: c_err.file_col_end as usize,
        snippet,
        highlight,
        label,
    };

    let mut err = Error::new(code, message, span);

    // Add help text if present
    if let Some(help) = unsafe { raw::cstr_from_ptr(c_err.help) } {
        err = err.with_help(help);
    }

    err
}

/// Collect all errors from C++ and convert to Rust
pub fn collect_errors() -> Vec<Error> {
    error_collector_get_all()
        .into_iter()
        .filter_map(|ptr| {
            if ptr.is_null() {
                return None;
            }

            // SAFETY: `ptr` is non-null and points to a valid `ApxmError`
            let rust_err = unsafe { error_from_raw(&*ptr) };
            error_free(ptr);

            Some(rust_err)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_collector_count_exists() {
        // Integration test that verifies the function can be called without panicking.
        // Actual behavior depends on C API state.
        let _ = error_collector_count();
    }
}
