//! Passes registry

use crate::ffi;
use apxm_core::types::compiler::PassInfo;
use std::ffi::CString;

pub fn get_pass_count() -> usize {
    unsafe { ffi::apxm_pass_registry_get_count() }
}

pub fn get_pass_info(index: usize) -> Option<PassInfo> {
    let raw = unsafe { ffi::apxm_pass_registry_get_pass(index) };
    if raw.is_null() {
        return None;
    }

    unsafe { ffi::conversions::pass_info_from_raw(raw) }
}

pub fn find_pass(name: &str) -> Option<PassInfo> {
    let c_name = CString::new(name).ok()?;
    let raw = unsafe { ffi::apxm_pass_registry_find_pass(c_name.as_ptr()) };

    if raw.is_null() {
        return None;
    }

    unsafe { ffi::conversions::pass_info_from_raw(raw) }
}

pub fn list_passes() -> Vec<PassInfo> {
    let count = get_pass_count();
    (0..count).filter_map(get_pass_info).collect()
}
