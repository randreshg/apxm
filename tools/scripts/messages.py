"""Externalized user-facing messages for APXM CLI.

All user-visible strings (errors, warnings, info, labels, suggestions)
are defined here so they can be maintained, reviewed, and translated
in a single place. Code modules import from this file instead of
embedding raw strings.
"""


# ── Headers ───────────────────────────────────────────────────────────
HEADER_INSTALL = "APXM Install"
HEADER_ACTION_REQUIRED = "Action Required"

# ── Install stage labels ──────────────────────────────────────────────
STAGE_PLATFORM_DETECTION = "Stage 1: Platform detection"
STAGE_CHECKING_DEPS = "Stage 2: Checking dependencies"
STAGE_CONDA_ENV = "Stage 3: Conda environment"
STAGE_RUST_TOOLCHAIN = "Stage 4: Rust toolchain"
STAGE_BUILD = "Stage 5: Build"

# ── Platform messages ─────────────────────────────────────────────────
MSG_OS = "OS: {os_label}"
MSG_DISTRO = "Distro: {distro} {version}"
MSG_WSL_DETECTED = "WSL detected"
MSG_PKG_MANAGER = "Package manager: {pkg_manager}"

# ── Dependency messages ───────────────────────────────────────────────
MSG_DEP_OK = "{name}{version}"
MSG_DEP_NEEDS_UPGRADE = "{name}{version} -- needs upgrade"
MSG_DEP_NOT_FOUND = "{name} -- not found"
MSG_DEP_NOT_FOUND_OPTIONAL = "{name} -- not found (optional)"
MSG_DEP_UPGRADE_REQUIRED = "{name} (upgrade required)"
MSG_DEP_INSTALL_GENERIC = "Install {name} and ensure it is on your PATH"
MSG_DEP_UPGRADE_GENERIC = "Upgrade {name} to a newer version"

# ── Fix suggestions (keyed by command name) ───────────────────────────
FIX_SUGGESTIONS = {
    "rustc": (
        "Install Rust: curl --proto '=https' --tlsv1.2 -sSf "
        "https://sh.rustup.rs | sh && rustup default nightly"
    ),
    "cargo": (
        "Install Rust: curl --proto '=https' --tlsv1.2 -sSf "
        "https://sh.rustup.rs | sh"
    ),
    "mamba": (
        "Install Miniforge: curl -fsSL "
        "https://github.com/conda-forge/miniforge/releases/latest/download/"
        "Miniforge3-$(uname)-$(uname -m).sh | bash"
    ),
    "cmake": "Install CMake: conda install -c conda-forge cmake>=3.20",
    "ninja": "Install Ninja: conda install -c conda-forge ninja",
    "git": "Install Git: sudo apt install git  (or your distro's equivalent)",
    "llvm-config": "Install LLVM: conda install -c conda-forge llvmdev=21",
}

# ── Rust toolchain ────────────────────────────────────────────────────
MSG_RUST_VERSION = "Rust: {version}"
MSG_RUST_NIGHTLY_RECOMMENDED = "Rust: {version} (nightly recommended)"
MSG_RUSTUP_FOUND = "rustup: found"
MSG_RUSTUP_NOT_FOUND = "rustup -- not found"
MSG_RUST_NIGHTLY_INSTALLED = "Rust nightly: installed"
MSG_RUST_NIGHTLY_INSTALLING = "Rust nightly not found, installing..."
MSG_RUST_CHECK_FAILED = "Could not check Rust toolchain: {error}"
MSG_RUST_VERSION_UNKNOWN = "Could not determine Rust version"
MSG_RUST_NIGHTLY_MISSING = "Rust nightly (rustup toolchain install nightly)"
MSG_RUSTUP_INSTALLING = "Installing rustup..."
MSG_RUSTUP_INSTALL_FAILED = "Failed to install rustup: {error}"

# ── Conda environment ────────────────────────────────────────────────
MSG_CONDA_NOT_FOUND = "Mamba/Conda -- not found"
MSG_CONDA_FOUND = "{cmd}: found"
MSG_CONDA_ENV_YAML_NOT_FOUND = "environment.yaml not found: {path}"
MSG_CONDA_ENV_OK = "Conda env 'apxm': {prefix}"
MSG_CONDA_ENV_NOT_FOUND_WILL_CREATE = (
    "Conda env 'apxm': not found (will be created)"
)
MSG_CONDA_ENV_NOT_FOUND_RUN_INSTALL = (
    "Conda env 'apxm' (run install without --check)"
)
MSG_CONDA_CREATING = "Creating/updating conda environment..."
MSG_CONDA_CREATE_FAILED = "{cmd} env create/update failed"
MSG_CONDA_ENV_CREATED = "Conda environment created"
MSG_CONDA_ENV_UPDATED = "Conda environment updated"
MSG_CONDA_ENV_CREATE_FAILED = "Conda env (create/update failed)"

# ── Build status ──────────────────────────────────────────────────────
MSG_COMPILER_BIN_OK_SHORT = "Binary: {path}"
MSG_COMPILER_NOT_BUILT_YET = "Compiler not built yet"
MSG_BUILD_COMPLETE = "Build complete!"
MSG_BUILD_FAILED = "Build failed!"
MSG_BUILD_CONDA_ACTIVATE = "Build (apxm build)"
MSG_BUILD_INSTALL_RUST = "Build (install Rust first)"
MSG_BUILD_FIX_RETRY = "Build (fix errors and retry: apxm build)"
MSG_CARGO_NOT_AVAILABLE = "Cargo not available -- skipping build"
MSG_CONDA_NOT_ACTIVATED = "Conda env not found -- skipping build"

# ── Summary ───────────────────────────────────────────────────────────
MSG_EVERYTHING_OK = "Everything looks good!"
MSG_FIX_RERUN = "Fix the above, then re-run: apxm install"
