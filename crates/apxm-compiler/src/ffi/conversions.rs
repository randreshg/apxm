//! FFI type conversions
//!
//! Utilities for converting between C FFI types and Rust types.

use crate::ffi::raw;
use apxm_core::types::compiler::{PassCategory, PassInfo};
use std::ffi::CStr;

/// Convert a raw pass info pointer into a Rust-friendly struct
///
/// # Safety
/// The pointer must be valid and non-null.
pub(crate) unsafe fn pass_info_from_raw(info: *const raw::ApxmPassInfo) -> Option<PassInfo> {
    if info.is_null() {
        return None;
    }

    // SAFETY: Caller guarantees pointer is valid and non-null
    unsafe {
        let info_ref = &*info;
        Some(PassInfo {
            name: CStr::from_ptr(info_ref.name).to_string_lossy().into_owned(),
            description: CStr::from_ptr(info_ref.description)
                .to_string_lossy()
                .into_owned(),
            category: PassCategory::from(info_ref.category),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_info_from_raw_with_null_returns_none() {
        let result = unsafe { pass_info_from_raw(std::ptr::null()) };
        assert!(result.is_none());
    }
}
