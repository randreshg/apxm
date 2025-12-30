use anyhow::{Context, Result};
use std::{
    collections::BTreeMap,
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
                "LLVM_DIR".to_string(),
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
    if let Ok(path) = which::which("mlir-tblgen")
        && let Some(prefix) = path.parent().and_then(|p| p.parent())
    {
        candidates.push(prefix.to_path_buf());
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
        .map(|value| normalize_candidate_prefix(&PathBuf::from(value)))
        .filter(|path| path.exists() && path.is_dir())
        .collect()
}

fn normalize_candidate_prefix(path: &Path) -> PathBuf {
    if let Some(prefix) = cmake_dir_to_prefix(path) {
        return prefix;
    }
    path.to_path_buf()
}

fn cmake_dir_to_prefix(path: &Path) -> Option<PathBuf> {
    let parts: Vec<String> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_string()))
        .collect();

    if parts.len() < 3 {
        return None;
    }

    let tail = &parts[parts.len() - 3..];
    let lib_dir = tail[0].as_str();
    let cmake_dir = tail[1].as_str();
    let package_dir = tail[2].as_str();

    let is_lib_dir = matches!(lib_dir, "lib" | "lib64");
    let is_cmake_dir = cmake_dir.eq_ignore_ascii_case("cmake");
    let is_package_dir = matches!(package_dir, "mlir" | "llvm");

    if is_lib_dir && is_cmake_dir && is_package_dir {
        let mut prefix = path.to_path_buf();
        for _ in 0..3 {
            prefix = prefix.parent()?.to_path_buf();
        }
        return Some(prefix);
    }

    None
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
    if cfg!(unix)
        && let Some((src, dst_name)) = &spec.symlink
    {
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
            .inspect(|version| {
                log_debug!(
                    "build::llvm",
                    "Detected LLVM version from llvm-config: {}",
                    version
                )
            })
            .or_else(|| {
                log_debug!(
                    "build::llvm",
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
        .inspect(|version| {
            log_debug!(
                "build::llvm",
                "Detected LLVM version from libraries: {}",
                version
            )
        })
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

#[derive(Debug, Clone)]
pub struct CandidateReport {
    pub prefix: PathBuf,
    pub has_mlir_lib: bool,
    pub has_llvm_lib: bool,
    pub has_mlir_cmake: bool,
    pub has_llvm_cmake: bool,
}

#[derive(Debug, Clone)]
pub struct MlirEnvReport {
    pub env_vars: BTreeMap<String, Option<String>>,
    pub extra_env_vars: BTreeMap<String, String>,
    pub cmake_available: bool,
    pub mlir_tblgen_path: Option<PathBuf>,
    pub llvm_config_path: Option<PathBuf>,
    pub candidates: Vec<CandidateReport>,
    pub resolved_prefix: Option<PathBuf>,
    pub resolved_lib_dir: Option<PathBuf>,
    pub resolved_mlir_cmake_dir: Option<PathBuf>,
    pub resolved_llvm_cmake_dir: Option<PathBuf>,
    pub llvm_version: Option<String>,
    pub locate_error: Option<String>,
}

impl MlirEnvReport {
    pub fn detect() -> Self {
        let config = LibraryConfig::for_mlir();
        let env_vars = gather_env_vars(&config.env_vars);
        let extra_env_vars = gather_extra_env_vars();
        let cmake_available = command_available("cmake");
        let mlir_tblgen_path = find_in_path("mlir-tblgen");
        let llvm_config_path = find_in_path("llvm-config");
        let candidates = build_candidate_reports(
            &config,
            &env_vars,
            mlir_tblgen_path.as_ref(),
            llvm_config_path.as_ref(),
        );

        let mut resolved_prefix = None;
        let mut resolved_lib_dir = None;
        let mut resolved_mlir_cmake_dir = None;
        let mut resolved_llvm_cmake_dir = None;
        let mut llvm_version = None;
        let mut locate_error = None;

        match locate_library(&config) {
            Ok(spec) => {
                if let Some(lib_dir) = spec.search_paths.first() {
                    resolved_lib_dir = Some(lib_dir.clone());
                    if let Some(prefix) = lib_dir.parent().map(|p| p.to_path_buf()) {
                        let mlir_cmake = prefix.join("lib/cmake/mlir");
                        if mlir_cmake.is_dir() {
                            resolved_mlir_cmake_dir = Some(mlir_cmake);
                        }

                        let llvm_cmake = prefix.join("lib/cmake/llvm");
                        if llvm_cmake.is_dir() {
                            resolved_llvm_cmake_dir = Some(llvm_cmake);
                        }

                        llvm_version = detect_llvm_version(&prefix);
                        resolved_prefix = Some(prefix);
                    }
                }
            }
            Err(err) => {
                locate_error = Some(format!("{err:#}"));
            }
        }

        Self {
            env_vars,
            extra_env_vars,
            cmake_available,
            mlir_tblgen_path,
            llvm_config_path,
            candidates,
            resolved_prefix,
            resolved_lib_dir,
            resolved_mlir_cmake_dir,
            resolved_llvm_cmake_dir,
            llvm_version,
            locate_error,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.resolved_prefix.is_some() && self.resolved_lib_dir.is_some()
    }

    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("MLIR environment diagnostics:".to_string());

        lines.push("Env vars:".to_string());
        for (key, value) in &self.env_vars {
            match value {
                Some(val) => lines.push(format!("- {key}={val}")),
                None => lines.push(format!("- {key}=<not set>")),
            }
        }

        if !self.extra_env_vars.is_empty() {
            lines.push("Extra env vars:".to_string());
            for (key, value) in &self.extra_env_vars {
                lines.push(format!("- {key}={value}"));
            }
        }

        lines.push(format!(
            "cmake: {}",
            if self.cmake_available {
                "available"
            } else {
                "missing"
            }
        ));
        lines.push(format!(
            "mlir-tblgen: {}",
            self.mlir_tblgen_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<not found in PATH>".to_string())
        ));
        lines.push(format!(
            "llvm-config: {}",
            self.llvm_config_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<not found in PATH>".to_string())
        ));

        if !self.candidates.is_empty() {
            lines.push("Candidate prefixes:".to_string());
            for candidate in &self.candidates {
                lines.push(format!(
                    "- {} (mlir_lib: {}, llvm_lib: {}, mlir_cmake: {}, llvm_cmake: {})",
                    candidate.prefix.display(),
                    yes_no(candidate.has_mlir_lib),
                    yes_no(candidate.has_llvm_lib),
                    yes_no(candidate.has_mlir_cmake),
                    yes_no(candidate.has_llvm_cmake),
                ));
            }
        }

        if let Some(prefix) = &self.resolved_prefix {
            lines.push(format!("Resolved MLIR prefix: {}", prefix.display()));
        } else {
            lines.push("Resolved MLIR prefix: <not found>".to_string());
        }

        if let Some(lib_dir) = &self.resolved_lib_dir {
            lines.push(format!("Resolved MLIR lib dir: {}", lib_dir.display()));
        }

        if let Some(mlir_dir) = &self.resolved_mlir_cmake_dir {
            lines.push(format!("Resolved MLIR cmake dir: {}", mlir_dir.display()));
        }

        if let Some(llvm_dir) = &self.resolved_llvm_cmake_dir {
            lines.push(format!("Resolved LLVM cmake dir: {}", llvm_dir.display()));
        }

        if let Some(version) = &self.llvm_version {
            lines.push(format!("Detected LLVM version: {version}"));
        }

        if let Some(error) = &self.locate_error {
            lines.push(format!("Locate error: {error}"));
        }

        lines.join("\n")
    }

    pub fn apply_env(&self) {
        if let Some(prefix) = &self.resolved_prefix {
            set_env_if_missing("MLIR_PREFIX", prefix);
            set_env_if_missing("LLVM_PREFIX", prefix);
        }

        if let Some(mlir_dir) = &self.resolved_mlir_cmake_dir {
            set_env_if_missing("MLIR_DIR", mlir_dir);
        }

        if let Some(llvm_dir) = &self.resolved_llvm_cmake_dir {
            set_env_if_missing("LLVM_DIR", llvm_dir);
        }
    }
}

fn gather_env_vars(keys: &[String]) -> BTreeMap<String, Option<String>> {
    keys.iter()
        .map(|key| (key.clone(), env::var(key).ok()))
        .collect()
}

fn gather_extra_env_vars() -> BTreeMap<String, String> {
    let mut vars = BTreeMap::new();
    for (key, value) in env::vars() {
        if key.starts_with("MLIR_SYS_") || key.starts_with("LLVM_SYS_") {
            vars.insert(key, value);
        }
    }
    vars
}

fn build_candidate_reports(
    config: &LibraryConfig,
    env_vars: &BTreeMap<String, Option<String>>,
    mlir_tblgen_path: Option<&PathBuf>,
    llvm_config_path: Option<&PathBuf>,
) -> Vec<CandidateReport> {
    let mut prefixes = Vec::new();
    for value in env_vars.values().flatten() {
        let path = normalize_candidate_prefix(&PathBuf::from(value));
        if path.is_dir() {
            prefixes.push(path);
        }
    }

    if let Some(path) = mlir_tblgen_path {
        if let Some(prefix) = path.parent().and_then(|p| p.parent()) {
            prefixes.push(prefix.to_path_buf());
        }
    }

    if let Some(path) = llvm_config_path {
        if let Some(prefix) = path.parent().and_then(|p| p.parent()) {
            prefixes.push(prefix.to_path_buf());
        }
    }

    prefixes.sort();
    prefixes.dedup();

    prefixes
        .into_iter()
        .map(|prefix| {
            let (has_mlir_lib, has_llvm_lib) = scan_lib_dirs(&prefix, config);
            let has_mlir_cmake = prefix.join("lib/cmake/mlir").is_dir();
            let has_llvm_cmake = prefix.join("lib/cmake/llvm").is_dir();
            CandidateReport {
                prefix,
                has_mlir_lib,
                has_llvm_lib,
                has_mlir_cmake,
                has_llvm_cmake,
            }
        })
        .collect()
}

fn scan_lib_dirs(prefix: &Path, config: &LibraryConfig) -> (bool, bool) {
    let platform = Platform::current();
    let mut has_mlir = false;
    let mut has_llvm = false;

    for dir_name in &config.lib_dirs {
        let lib_dir = prefix.join(dir_name);
        if !lib_dir.is_dir() {
            continue;
        }

        for pattern in &config.lib_patterns {
            if lib_dir.join(pattern).is_file() {
                has_mlir = true;
                break;
            }
        }

        if !has_llvm {
            if let Ok(entries) = fs::read_dir(&lib_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                        if file_name.starts_with(platform.llvm_library_prefix())
                            && file_name.ends_with(platform.llvm_library_suffix())
                        {
                            has_llvm = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    (has_mlir, has_llvm)
}

fn command_available(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn find_in_path(cmd: &str) -> Option<PathBuf> {
    let mut candidates = vec![cmd.to_string()];
    if cfg!(windows) {
        candidates.push(format!("{cmd}.exe"));
    }

    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        for name in &candidates {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn set_env_if_missing(key: &str, value: &Path) {
    if env::var_os(key).is_none() {
        unsafe {
            env::set_var(key, value);
        }
    }
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
