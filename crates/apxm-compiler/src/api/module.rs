//! Module API for the Apxm compiler.
//!
//! Modules are the building blocks of Apxm programs.
//! They contain a set of functions, variables, and types that can be used by other modules.
//!
//! Modules can be imported using the `import` keyword, which allows the functions, variables,
//! and types defined in the imported module to be used in the current module.

use crate::api::Context;
use crate::ffi;
use apxm_core::error::builder::ErrorBuilder;
use apxm_core::error::codes::ErrorCode;
use apxm_core::error::compiler::{CompilerError, Result};
use apxm_core::types::CodegenOptions;
use std::ffi::CString;
use std::fs;
use std::marker::PhantomData;
use std::path::Path;

pub(crate) fn invalid_input_error(message: impl Into<String>) -> CompilerError {
    CompilerError::InvalidInput(Box::new(ErrorBuilder::generic(
        ErrorCode::InternalError,
        message,
    )))
}

pub struct Module {
    raw: *mut ffi::ApxmModule,
    _context: PhantomData<Context>,
}

impl Module {
    /// # Safety
    ///
    /// The caller must ensure that `raw` is a valid pointer to an `ApxmModule`
    /// and that ownership is properly transferred to the returned `Module`.
    pub unsafe fn from_raw(raw: *mut ffi::ApxmModule) -> Self {
        Self {
            raw,
            _context: PhantomData,
        }
    }

    /// Parses the given source string into a module.
    pub fn parse(context: &Context, source: &str) -> Result<Self> {
        let c_source = CString::new(source)
            .map_err(|e| invalid_input_error(format!("Invalid source string: {}", e)))?;

        let raw = ffi::handle_null_result(
            unsafe { ffi::apxm_module_parse(context.as_ptr(), c_source.as_ptr()) },
            "module parsing",
        )?;

        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Parses the given DSL source string into a module.
    pub fn parse_dsl(context: &Context, source: &str, filename: &str) -> Result<Self> {
        let c_source = CString::new(source)
            .map_err(|e| invalid_input_error(format!("Invalid DSL source: {}", e)))?;
        let c_filename = CString::new(filename)
            .map_err(|e| invalid_input_error(format!("Invalid filename: {}", e)))?;

        let raw = ffi::handle_null_result(
            unsafe {
                ffi::apxm_parse_dsl(context.as_ptr(), c_source.as_ptr(), c_filename.as_ptr())
            },
            "DSL parsing",
        )?;

        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Parses the given DSL source file into a module.
    pub fn parse_dsl_file(context: &Context, path: &Path) -> Result<Self> {
        let path_str = path
            .to_str()
            .ok_or_else(|| invalid_input_error("Invalid path encoding".to_string()))?;
        let c_path = CString::new(path_str)
            .map_err(|e| invalid_input_error(format!("Invalid path: {}", e)))?;

        let raw = ffi::handle_null_result(
            unsafe { ffi::apxm_parse_dsl_file(context.as_ptr(), c_path.as_ptr()) },
            "DSL file parsing",
        )?;

        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Parses the given source file into a module.
    pub fn parse_file(context: &Context, path: &Path) -> Result<Self> {
        let source = fs::read_to_string(path).map_err(CompilerError::Io)?;
        Self::parse(context, &source)
    }

    /// Verifies the module's structure and syntax.
    pub fn verify(&self) -> Result<()> {
        ffi::handle_bool_result(
            unsafe { ffi::apxm_module_verify(self.raw) },
            "module verification",
        )
    }

    pub fn to_string(&self) -> Result<String> {
        let c_str = ffi::handle_null_result(
            unsafe { ffi::apxm_module_to_string(self.raw) },
            "module serialization",
        )?;

        let result = unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_string_lossy()
                .into_owned()
        };

        unsafe { ffi::apxm_string_free(c_str as *mut _) };

        Ok(result)
    }

    pub fn generate_rust_code(&self) -> Result<String> {
        let c_str = ffi::handle_null_result(
            unsafe { ffi::apxm_codegen_emit_rust(self.raw) },
            "Rust code generation",
        )?;

        let result = unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_string_lossy()
                .into_owned()
        };

        unsafe { ffi::apxm_string_free(c_str as *mut _) };

        Ok(result)
    }

    pub fn generate_rust_code_with_options(&self, options: &CodegenOptions) -> Result<String> {
        let module_name_cstr = options
            .module_name
            .as_deref()
            .map(CString::new)
            .transpose()
            .map_err(|e| invalid_input_error(format!("Invalid module name: {}", e)))?;

        let ffi_opts = ffi::ApxmCodegenOptions {
            optimize: options.optimize,
            emit_comments: options.emit_comments,
            emit_debug_symbols: options.emit_debug_symbols,
            standalone: options.standalone,
            module_name: module_name_cstr
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null()),
        };

        let c_str = ffi::handle_null_result(
            unsafe { ffi::apxm_codegen_emit_rust_with_options(self.raw, &ffi_opts) },
            "Rust code generation",
        )?;

        let result = unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_string_lossy()
                .into_owned()
        };

        unsafe { ffi::apxm_string_free(c_str as *mut _) };

        Ok(result)
    }

    pub fn as_ptr(&self) -> *mut ffi::ApxmModule {
        self.raw
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::apxm_module_destroy(self.raw);
            }
        }
    }
}
