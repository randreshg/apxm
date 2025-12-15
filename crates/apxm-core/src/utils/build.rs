use anyhow::{Context, Result};
use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::log_debug;

/// Platform-specific library naming conventions
#[derive(Debug, Clone, Copy)]
pub enum Platform {
    Windows,
    MacOS,
    Linux,
}

impl Platform {
    pub fn current() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOS
        } else {
            Self::Linux
        }
    }

    pub fn mlir_library_patterns(&self) -> Vec<&'static str> {
        match self {
            Self::Windows => vec!["MLIR.dll"],
            Self::MacOS => vec!["libMLIR.dylib", "libMLIR.so"],
            Self::Linux => vec!["libMLIR.so"],
        }
    }

    pub fn mlir_unversioned_name(&self) -> &'static str {
        match self {
            Self::Windows => "MLIR.dll",
            Self::MacOS => "libMLIR.dylib",
            Self::Linux => "libMLIR.so",
        }
    }

    pub fn llvm_library_prefix(&self) -> &'static str {
        match self {
            Self::Windows => "LLVM-",
            Self::MacOS | Self::Linux => "libLLVM-",
        }
    }

    pub fn llvm_library_suffix(&self) -> &'static str {
        match self {
            Self::Windows => ".dll",
            Self::MacOS => ".dylib",
            Self::Linux => ".so",
        }
    }
}

/// Link specification for native library integration.
#[derive(Debug)]
pub struct LinkSpec {
    pub search_paths: Vec<PathBuf>,
    pub rpaths: Vec<PathBuf>,
    pub libs: Vec<String>,
    pub symlink: Option<(PathBuf, PathBuf)>,
}

impl LinkSpec {
    pub fn new(
        search_paths: Vec<PathBuf>,
        rpaths: Vec<PathBuf>,
        libs: Vec<String>,
        symlink: Option<(PathBuf, PathBuf)>,
    ) -> Self {
        Self {
            search_paths,
            rpaths,
            libs,
            symlink,
        }
    }

    pub fn simple(lib_path: &Path, lib_name: &str) -> Self {
        let parent = lib_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        Self {
            search_paths: vec![parent.clone()],
            rpaths: vec![parent],
            libs: vec![lib_name.to_string()],
            symlink: None,
        }
    }
}

/// Configuration for library discovery.
#[derive(Debug)]
pub struct LibraryConfig {
    pub env_vars: Vec<String>,
    pub lib_dirs: Vec<String>,
    pub lib_patterns: Vec<String>,
    pub platform: Platform,
}

impl LibraryConfig {
    pub fn for_mlir() -> Self {
        let platform = Platform::current();
        Self {
            env_vars: vec![
                "MLIR_PREFIX".to_string(),
                "CONDA_PREFIX".to_string(),
                "LLVM_PREFIX".to_string(),
                "MLIR_DIR".to_string(),
            ],
            lib_dirs: vec!["lib".to_string(), "lib64".to_string()],
            lib_patterns: platform
                .mlir_library_patterns()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            platform,
        }
    }

    pub fn with_env_var(mut self, var: impl Into<String>) -> Self {
        self.env_vars.push(var.into());
        self
    }

    pub fn with_lib_dir(mut self, dir: impl Into<String>) -> Self {
        self.lib_dirs.push(dir.into());
        self
    }
}

/// Locate a shared library using the provided configuration.
pub fn locate_library(config: &LibraryConfig) -> Result<LinkSpec> {
    let mut candidates = gather_prefix_candidates(&config.env_vars);

    // Try to find mlir-tblgen in PATH as a fallback
    if let Ok(path) = which::which("mlir-tblgen") {
        if let Some(prefix) = path.parent().and_then(|p| p.parent()) {
            candidates.push(prefix.to_path_buf());
        }
    }

    let library_path = candidates
        .iter()
        .filter_map(|prefix| find_library(prefix, config))
        .next()
        .with_context(|| {
            format!(
                "Library not found in any candidate path. Set one of: {}",
                config.env_vars.join(", ")
            )
        })?;

    build_link_spec(&library_path, config)
}

/// Gather candidate installation prefixes from environment variables.
fn gather_prefix_candidates(env_vars: &[String]) -> Vec<PathBuf> {
    env_vars
        .iter()
        .filter_map(|key| env::var(key).ok())
        .map(PathBuf::from)
        .filter(|path| path.exists() && path.is_dir())
        .collect()
}

/// Find a library under the given prefix using the configuration.
fn find_library(prefix: &Path, config: &LibraryConfig) -> Option<PathBuf> {
    config
        .lib_dirs
        .iter()
        .map(|dir_name| prefix.join(dir_name))
        .filter(|libdir| libdir.exists() && libdir.is_dir())
        .find_map(|libdir| {
            config
                .lib_patterns
                .iter()
                .map(|pattern| libdir.join(pattern))
                .find(|candidate| candidate.exists() && candidate.is_file())
                .or_else(|| find_versioned_library(&libdir, config))
        })
}

/// Find versioned libraries (e.g., libMLIR.so.17.0).
fn find_versioned_library(libdir: &Path, config: &LibraryConfig) -> Option<PathBuf> {
    let base_names: Vec<String> = config
        .lib_patterns
        .iter()
        .filter_map(|p| {
            let stem = Path::new(p).file_stem()?.to_str()?;
            Some(format!("{}.", stem))
        })
        .collect();

    fs::read_dir(libdir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .map(|name| base_names.iter().any(|base| name.starts_with(base)))
                    .unwrap_or(false)
        })
        .max_by_key(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .map(|s| s.to_string())
                .unwrap_or_default()
        })
}

/// Build a LinkSpec from a library path.
fn build_link_spec(library_path: &Path, config: &LibraryConfig) -> Result<LinkSpec> {
    let parent = library_path
        .parent()
        .context("Library parent directory missing")?
        .to_path_buf();

    let file_name = library_path
        .file_name()
        .and_then(OsStr::to_str)
        .context("Library filename missing")?;

    let is_unversioned = config
        .lib_patterns
        .iter()
        .any(|pattern| file_name.eq_ignore_ascii_case(pattern.as_str()));

    let symlink = (!is_unversioned).then(|| {
        let symlink_name = config.platform.mlir_unversioned_name();
        (library_path.to_path_buf(), PathBuf::from(symlink_name))
    });

    Ok(LinkSpec::new(
        vec![parent.clone()],
        vec![parent],
        vec!["MLIR".to_string()],
        symlink,
    ))
}

/// Emit cargo directives for the provided LinkSpec.
pub fn emit_link_directives(spec: &LinkSpec, out_dir: &Path) -> Result<()> {
    // Create symlink if requested (Unix only)
    if cfg!(unix) {
        if let Some((src, dst_name)) = &spec.symlink {
            let dst = out_dir.join(dst_name);
            if !dst.exists() {
                std::os::unix::fs::symlink(src, &dst).with_context(|| {
                    format!(
                        "Failed to create symlink {} -> {}",
                        dst.display(),
                        src.display()
                    )
                })?;
            }
        }
    }

    // Emit search paths
    let mut search_paths = spec.search_paths.clone();
    if spec.symlink.is_some() {
        search_paths.push(out_dir.to_path_buf());
    }

    for path in &search_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    // Emit rpaths (Unix only)
    if cfg!(unix) {
        for rpath in &spec.rpaths {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath.display());
        }
    }

    // Emit libraries
    for lib in &spec.libs {
        if lib.starts_with(':') {
            println!("cargo:rustc-link-lib={lib}");
        } else {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }

    Ok(())
}

/// Detect LLVM major version from llvm-config or library files.
pub fn detect_llvm_version(prefix: &Path) -> Option<String> {
    // Try llvm-config first
    let config_name = if cfg!(target_os = "windows") {
        "llvm-config.exe"
    } else {
        "llvm-config"
    };

    let llvm_config = prefix.join("bin").join(config_name);
    if llvm_config.exists() {
        Command::new(&llvm_config)
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|v| v.trim().split('.').next().map(String::from))
            .inspect(|version| log_debug!("Detected LLVM version from llvm-config: {}", version))
            .or_else(|| {
                log_debug!(
                    "Failed to get LLVM version from llvm-config, falling back to library scan",
                );
                None
            })
    } else {
        find_llvm_version_from_libraries(prefix)
    }
}

/// Find LLVM version by scanning library files.
fn find_llvm_version_from_libraries(prefix: &Path) -> Option<String> {
    let platform = Platform::current();
    let lib_dir = prefix.join("lib");
    if !lib_dir.exists() || !lib_dir.is_dir() {
        return None;
    }

    let prefix_str = platform.llvm_library_prefix();
    let suffix_str = platform.llvm_library_suffix();

    fs::read_dir(lib_dir)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }

            path.file_name().and_then(OsStr::to_str).and_then(|name| {
                if name.starts_with(prefix_str) && name.ends_with(suffix_str) {
                    let version_str = &name[prefix_str.len()..name.len() - suffix_str.len()];
                    version_str.split('.').next().map(String::from)
                } else {
                    None
                }
            })
        })
        .max_by(|a, b| a.cmp(b))
        .inspect(|version| log_debug!("Detected LLVM version from libraries: {}", version))
}

/// Find versioned MLIR shared library with version suffix.
pub fn find_versioned_mlir_library(lib_dir: &Path, llvm_version: &str) -> Option<PathBuf> {
    let platform = Platform::current();
    let patterns = match platform {
        Platform::Windows => vec![format!("MLIR-{}.dll", llvm_version)],
        Platform::MacOS => vec![
            format!("libMLIR.{}.dylib", llvm_version),
            format!("libMLIR.dylib.{}", llvm_version),
        ],
        Platform::Linux => vec![format!("libMLIR.so.{}", llvm_version)],
    };

    // Try versioned libraries first
    for pattern in &patterns {
        if let Ok(entries) = fs::read_dir(lib_dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_file()
                    && path
                        .file_name()
                        .and_then(OsStr::to_str)
                        .map(|name| name.starts_with(pattern))
                        .unwrap_or(false)
                {
                    return Some(path);
                }
            }
        }
    }

    // Fallback to unversioned library
    let fallback_names = platform.mlir_library_patterns();
    fallback_names
        .iter()
        .map(|name| lib_dir.join(name))
        .find(|path| path.exists() && path.is_file())
}

/// Get the workspace root directory from a manifest directory.
pub fn get_workspace_root(manifest_dir: &Path) -> Result<PathBuf> {
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .context("Could not find workspace root")
}

/// Get the workspace target directory.
pub fn get_target_dir(workspace_root: &Path, profile: &str) -> PathBuf {
    workspace_root.join("target").join(profile)
}
