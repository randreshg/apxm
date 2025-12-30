//! This file is responsible for managing passes in the compiler.

use crate::api::{Context, Module, module::invalid_input_error};
use crate::ffi;
use apxm_core::error::compiler::Result;
use apxm_core::types::OptimizationLevel;
use std::ffi::CString;
use std::marker::PhantomData;

pub struct PassManager<'ctx> {
    raw: *mut ffi::ApxmPassManager,
    _context: PhantomData<&'ctx Context>,
}

impl<'ctx> PassManager<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<Self> {
        let raw = ffi::handle_null_result(
            unsafe { ffi::apxm_pass_manager_create(context.as_ptr()) },
            "pass manager creation",
        )?;

        Ok(Self {
            raw,
            _context: PhantomData,
        })
    }

    pub fn from_opt_level(context: &'ctx Context, level: OptimizationLevel) -> Result<Self> {
        let mut pm = Self::new(context)?;
        super::pipeline::build_pipeline(&mut pm, level)?;
        Ok(pm)
    }

    pub fn add_pass(&mut self, name: &str) -> Result<&mut Self> {
        let c_name = CString::new(name)
            .map_err(|e| invalid_input_error(format!("Invalid pass name: {}", e)))?;

        ffi::handle_bool_result(
            unsafe { ffi::apxm_pass_manager_add_pass_by_name(self.raw, c_name.as_ptr()) },
            &format!("adding pass '{}'", name),
        )?;

        Ok(self)
    }

    pub fn normalize(&mut self) -> Result<&mut Self> {
        self.add_pass("normalize")
    }

    pub fn scheduling(&mut self) -> Result<&mut Self> {
        self.add_pass("scheduling")
    }

    pub fn fuse_reasoning(&mut self) -> Result<&mut Self> {
        self.add_pass("fuse-reasoning")
    }

    pub fn canonicalizer(&mut self) -> Result<&mut Self> {
        self.add_pass("canonicalizer")
    }

    pub fn cse(&mut self) -> Result<&mut Self> {
        self.add_pass("cse")
    }

    pub fn symbol_dce(&mut self) -> Result<&mut Self> {
        self.add_pass("symbol-dce")
    }

    pub fn lower_to_async(&mut self) -> Result<&mut Self> {
        self.add_pass("lower-to-async")
    }

    pub fn unconsumed_value_warning(&mut self) -> Result<&mut Self> {
        self.add_pass("unconsumed-value-warning")
    }

    pub fn run(&self, module: &Module) -> Result<()> {
        ffi::handle_bool_result(
            unsafe { ffi::apxm_pass_manager_run(self.raw, module.as_ptr()) },
            "pass manager execution",
        )
    }

    pub fn clear(&mut self) {
        unsafe {
            ffi::apxm_pass_manager_clear(self.raw);
        }
    }
}

impl Drop for PassManager<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::apxm_pass_manager_destroy(self.raw);
            }
        }
    }
}
