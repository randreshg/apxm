//! Raw FFI bindings to the C API.
//!
//! This module contains the automatically generated bindings and
//! safe wrappers for string conversion utilities.

use std::ffi::CStr;
use std::os::raw::c_char;

// Include generated bindings from build.rs
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Convert a C string pointer to an optional Rust String
///
/// # Safety
/// The pointer must be valid, null-terminated, and remain valid for the duration of the call.
pub unsafe fn cstr_from_ptr(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: Caller guarantees `ptr` is valid and null-terminated
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// Convert a C string pointer to a Rust String with a default fallback
///
/// # Safety
/// The pointer must be valid, null-terminated, and remain valid for the duration of the call.
pub unsafe fn cstr_or_default(ptr: *const c_char, default: &str) -> String {
    unsafe { cstr_from_ptr(ptr) }.unwrap_or_else(|| default.to_owned())
}
