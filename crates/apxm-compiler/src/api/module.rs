//! Module API for the Apxm compiler.
//!
//! Modules are the building blocks of Apxm programs.
//! They contain a set of functions, variables, and types that can be used by other modules.
//!
//! Modules can be imported using the `import` keyword, which allows the functions, variables,
//! and types defined in the imported module to be used in the current module.

use crate::api::Context;
use crate::codegen::artifact::parse_wire_dags;
use crate::ffi;
use apxm_artifact::{Artifact, ArtifactMetadata};
use apxm_core::error::builder::ErrorBuilder;
use apxm_core::error::codes::ErrorCode;
use apxm_core::error::compiler::{CompilerError, Result};
use std::ffi::CString;
use std::fs;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

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

    pub fn as_ptr(&self) -> *mut ffi::ApxmModule {
        self.raw
    }

    pub fn generate_artifact_bytes(&self) -> Result<Vec<u8>> {
        self.generate_artifact_bytes_with_name(None)
    }

    pub fn generate_artifact_bytes_with_name(&self, module_name: Option<&str>) -> Result<Vec<u8>> {
        let payload = self.emit_artifact_payload(module_name)?;
        let dags = parse_wire_dags(&payload)?;

        // Find entry DAG for metadata naming
        let entry_dag = dags.iter().find(|d| d.metadata.is_entry);
        let name = entry_dag
            .and_then(|d| d.metadata.name.clone())
            .or_else(|| dags.first().and_then(|d| d.metadata.name.clone()));

        let metadata = ArtifactMetadata::new(name, crate::VERSION);
        Artifact::new(metadata, dags)
            .to_bytes()
            .map_err(|err| invalid_input_error(err.to_string()))
    }

    pub fn generate_artifact_to_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.generate_artifact_bytes()?;
        std::fs::write(path, &bytes).map_err(CompilerError::Io)
    }

    fn emit_artifact_payload(&self, module_name: Option<&str>) -> Result<Vec<u8>> {
        let module_name_cstr = module_name
            .map(|name| {
                CString::new(name)
                    .map_err(|e| invalid_input_error(format!("Invalid module name: {e}")))
            })
            .transpose()?;

        let c_options = ffi::ApxmArtifactOptions {
            module_name: module_name_cstr
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            emit_debug_json: false,
            target_version: ptr::null(),
        };

        let raw = ffi::handle_null_result(
            unsafe { ffi::apxm_codegen_emit_artifact(self.raw, &c_options) },
            "artifact generation",
        )?;

        unsafe { copy_artifact_buffer(raw) }
    }
}

unsafe fn copy_artifact_buffer(ptr: *mut c_char) -> Result<Vec<u8>> {
    struct BufferGuard(*mut c_char);

    impl Drop for BufferGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ffi::apxm_codegen_free(self.0) };
            }
        }
    }

    let guard = BufferGuard(ptr);
    if guard.0.is_null() {
        return Err(invalid_input_error("Artifact emitter returned null buffer"));
    }

    let mut len_buf = [0u8; 8];
    unsafe {
        len_buf.copy_from_slice(std::slice::from_raw_parts(guard.0 as *const u8, 8));
    }
    let len = u64::from_le_bytes(len_buf) as usize;

    let data = unsafe {
        let data_ptr = guard.0.add(std::mem::size_of::<u64>()) as *const u8;
        std::slice::from_raw_parts(data_ptr, len)
    };
    Ok(data.to_vec())
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
