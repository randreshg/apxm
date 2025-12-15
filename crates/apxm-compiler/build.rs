use anyhow::{Context, Result};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use apxm_core::utils::build::{
    LibraryConfig, LinkSpec, Platform, detect_llvm_version, emit_link_directives,
    find_versioned_mlir_library, get_target_dir, get_workspace_root, locate_library,
};
use apxm_core::{log_debug, log_info};

/// Build configuration derived from environment variables
struct BuildConfig {
    manifest_dir: PathBuf,
    out_dir: PathBuf,
    workspace_dir: PathBuf,
    profile: String,
    install_dir: PathBuf,
    build_dir: PathBuf,
}

impl BuildConfig {
    fn from_env() -> Result<Self> {
        let manifest_dir: PathBuf = env::var("CARGO_MANIFEST_DIR")
            .context("CARGO_MANIFEST_DIR not set")?
            .into();
        let out_dir: PathBuf = env::var("OUT_DIR").context("OUT_DIR not set")?.into();
        let workspace_dir = get_workspace_root(&manifest_dir)?;
        let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
        let install_dir = get_target_dir(&workspace_dir, &profile);
        let build_dir = out_dir.join("build");

        Ok(Self {
            manifest_dir,
            out_dir,
            workspace_dir,
            profile,
            install_dir,
            build_dir,
        })
    }

    fn ensure_directories(&self) -> Result<()> {
        std::fs::create_dir_all(&self.install_dir)?;
        std::fs::create_dir_all(&self.build_dir)?;
        Ok(())
    }
}

/// Paths describing the discovered MLIR installation.
struct MlirLayout {
    prefix: PathBuf,
    lib_dir: PathBuf,
    link_spec: LinkSpec,
}

impl MlirLayout {
    fn new(link_spec: LinkSpec) -> Result<Self> {
        let lib_dir = link_spec
            .search_paths
            .first()
            .context("No search paths in MLIR link spec")?
            .clone();

        let prefix = lib_dir
            .parent()
            .map(|p| p.to_path_buf())
            .context("Invalid MLIR path structure")?;

        Ok(Self {
            prefix,
            lib_dir,
            link_spec,
        })
    }
}

/// Set up cargo rerun triggers for relevant files
fn setup_rerun_triggers() {
    for path in ["CMakeLists.txt", "lib/CMakeLists.txt", "lib", "include"] {
        println!("cargo:rerun-if-changed={}", path);
    }
}

/// Locate MLIR installation and return key directories.
fn locate_mlir_layout() -> Result<MlirLayout> {
    let config = LibraryConfig::for_mlir();
    let link_spec = locate_library(&config)
        .context("MLIR not found. Please set MLIR_DIR or install LLVM/MLIR.")?;

    MlirLayout::new(link_spec)
}

/// Execute a command and check for success
fn run_command(cmd: &mut Command, error_msg: &str) -> Result<()> {
    log_debug!("apxm-compiler-build", "Running: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("Failed to execute command: {}", error_msg))?;

    if status.success() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "{} (exit code: {:?})",
            error_msg,
            status.code()
        ))
    }
}

/// Configure CMake build
fn configure_cmake(
    build_dir: &Path,
    manifest_dir: &Path,
    mlir_dir: &Path,
    install_dir: &Path,
    runtime_rpaths: &[PathBuf],
    workspace_root: &Path,
    profile: &str,
) -> Result<()> {
    let mlir_cmake_dir = mlir_dir.join("lib/cmake/mlir");
    let llvm_cmake_dir = mlir_dir.join("lib/cmake/llvm");

    // Map cargo PROFILE to an appropriate CMake build type.
    let cmake_build_type = if profile.eq_ignore_ascii_case("release") {
        "Release"
    } else {
        "Debug"
    };

    let mut cmd = Command::new("cmake");
    cmd.current_dir(build_dir)
        .arg(manifest_dir)
        .arg(format!("-DMLIR_DIR={}", mlir_cmake_dir.display()))
        .arg(format!("-DLLVM_DIR={}", llvm_cmake_dir.display()))
        .arg(format!("-DCMAKE_BUILD_TYPE={}", cmake_build_type))
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()))
        .arg(format!(
            "-DAPXM_WORKSPACE_ROOT={}",
            workspace_root.display()
        ));

    if !runtime_rpaths.is_empty() {
        let joined = runtime_rpaths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(";");
        cmd.arg(format!("-DAPXM_RUNTIME_RPATHS={joined}"));
    }

    run_command(&mut cmd, "CMake configuration failed")
}

/// Build CMake target
fn build_cmake(build_dir: &Path) -> Result<()> {
    run_command(
        Command::new("cmake").current_dir(build_dir).args([
            "--build",
            ".",
            "--target",
            "apxm_compiler_c",
            "--config",
            "Release",
            "-j",
        ]),
        "CMake build failed",
    )
}

/// Install CMake build
fn install_cmake(build_dir: &Path) -> Result<()> {
    run_command(
        Command::new("cmake")
            .current_dir(build_dir)
            .args(["--install", "."]),
        "CMake install failed",
    )
}

/// Generate Rust bindings using bindgen
fn generate_bindings(
    manifest_dir: &Path,
    out_dir: &Path,
    mlir_include_dir: &Path,
    project_include_dir: &Path,
    mlir_prefix: &Path,
) -> Result<()> {
    let bindings_path = out_dir.join("bindings.rs");
    let header_path = manifest_dir.join("include/apxm/CAPI/Compiler.h");

    // Set up libclang path for bindgen
    let clang_lib_path = mlir_prefix.join("lib");
    if clang_lib_path.exists() {
        log_info!(
            "apxm-compiler-build",
            "Setting clang library path for bindgen: {}",
            clang_lib_path.display()
        );
        unsafe {
            env::set_var("LIBCLANG_PATH", &clang_lib_path);
        }
    }

    let builder = bindgen::Builder::default()
        .header(header_path.to_str().context("Invalid header path")?)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("apxm_.*")
        .allowlist_type("Apxm.*")
        .clang_arg(format!("-I{}", mlir_include_dir.display()))
        .clang_arg(format!("-I{}", project_include_dir.display()))
        .size_t_is_usize(true);

    let bindings = builder.generate().context("Failed to generate bindings")?;

    bindings
        .write_to_file(&bindings_path)
        .context("Failed to write bindings")?;

    // Post-process to add unsafe to extern blocks (Rust 1.86+ requirement)
    post_process_bindings(&bindings_path)?;

    Ok(())
}

/// Post-process generated bindings to ensure Rust 1.86+ compatibility
fn post_process_bindings(bindings_path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(bindings_path)
        .context("Failed to read bindings for post-processing")?;

    let processed = content.replace("extern \"C\" {", "unsafe extern \"C\" {");

    std::fs::write(bindings_path, processed).context("Failed to write processed bindings")
}

/// Emit linker directives for C++ libraries
fn emit_compiler_link_directives(install_dir: &Path, mlir_layout: &MlirLayout) -> Result<()> {
    let apxm_lib_dir = install_dir.join("lib");
    let mlir_lib_dir = &mlir_layout.lib_dir;
    let mlir_prefix = &mlir_layout.prefix;

    // Link our own library
    println!("cargo:rustc-link-search=native={}", apxm_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=apxm_compiler_c");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", apxm_lib_dir.display());

    // Link MLIR/LLVM libraries
    println!("cargo:rustc-link-search=native={}", mlir_lib_dir.display());

    let llvm_version = detect_llvm_version(mlir_prefix).context("Failed to detect LLVM version")?;

    log_info!(
        "apxm-compiler-build",
        "Detected LLVM version: {}",
        llvm_version
    );

    // Platform-specific LLVM linking
    let platform = Platform::current();
    match platform {
        Platform::Windows => {
            println!("cargo:rustc-link-lib=dylib=LLVM-{}", llvm_version);
        }
        Platform::MacOS | Platform::Linux => {
            println!("cargo:rustc-link-lib=dylib=LLVM-{}", llvm_version);
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mlir_lib_dir.display());
        }
    }

    // Find and link MLIR library
    if let Some(mlir_lib) = find_versioned_mlir_library(mlir_lib_dir, &llvm_version) {
        log_info!(
            "apxm-compiler-build",
            "Using MLIR library: {}",
            mlir_lib.display()
        );

        // Emit the full path on macOS and Linux for better resolution
        match platform {
            Platform::Windows => {
                // Windows just needs the library name
                println!("cargo:rustc-link-lib=dylib=MLIR-{}", llvm_version);
            }
            Platform::MacOS | Platform::Linux => {
                println!("cargo:rustc-link-arg={}", mlir_lib.display());
            }
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=MLIR");
    }

    // Emit MLIR link spec directives
    emit_link_directives(&mlir_layout.link_spec, install_dir)?;

    Ok(())
}

/// Main build process orchestrator
fn build() -> Result<()> {
    let config = BuildConfig::from_env()?;
    config.ensure_directories()?;

    let MlirLayout {
        prefix: mlir_dir,
        lib_dir: mlir_lib_dir,
        link_spec: _,
    } = locate_mlir_layout()?;

    log_info!(
        "apxm-compiler-build",
        "Found MLIR at: {}",
        mlir_dir.display()
    );

    let apxm_lib_dir = config.install_dir.join("lib");
    let mut runtime_rpaths = vec![apxm_lib_dir, mlir_lib_dir.clone()];
    runtime_rpaths.sort();
    runtime_rpaths.dedup();

    // Check if CMake is available
    let cmake_available = Command::new("cmake")
        .arg("--version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if cmake_available {
        log_info!("apxm-compiler-build", "Configuring CMake...");
        configure_cmake(
            &config.build_dir,
            &config.manifest_dir,
            &mlir_dir,
            &config.install_dir,
            &runtime_rpaths,
            &config.workspace_dir,
            &config.profile,
        )?;

        log_info!("apxm-compiler-build", "Building CMake...");
        build_cmake(&config.build_dir)?;

        log_info!("apxm-compiler-build", "Installing CMake...");
        install_cmake(&config.build_dir)?;
    } else {
        log_info!(
            "apxm-compiler-build",
            "CMake not found in PATH. Skipping native library build."
        );
        log_info!(
            "apxm-compiler-build",
            "Please install CMake and rerun the build."
        );
    }

    // Generate bindings regardless of CMake availability
    let mlir_include_dir = mlir_dir.join("include");
    let project_include_dir = config.manifest_dir.join("include");

    generate_bindings(
        &config.manifest_dir,
        &config.out_dir,
        &mlir_include_dir,
        &project_include_dir,
        &mlir_dir,
    )?;

    emit_compiler_link_directives(&config.install_dir, &locate_mlir_layout()?)?;

    Ok(())
}

fn main() {
    setup_rerun_triggers();

    if let Err(e) = build() {
        eprintln!("apxm-compiler-build: build failed: {:#}", e);
        std::process::exit(1);
    }
}
