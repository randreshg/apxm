"""Install command for APXM CLI."""

import json
import os
import shutil
import subprocess as _sp
from pathlib import Path


def _find_conda_env_prefix(conda_cmd: str, env_name: str) -> "Path | None":
    """Return the prefix Path for *env_name* by scanning envs directories.

    Checks the filesystem directly so it finds broken directories (present but
    missing conda-meta) that mamba's env registry silently omits.
    """
    try:
        info = _sp.run(
            [conda_cmd, "info", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(info.stdout or "{}")
        # mamba uses "envs directories", conda uses "envs_dirs"
        envs_dirs = data.get("envs directories") or data.get("envs_dirs") or []
        for d in envs_dirs:
            candidate = Path(d) / env_name
            if candidate.is_dir():
                return candidate
    except Exception:
        pass
    return None

from sniff import (
    BinaryInstaller, CondaDetector, Exit, Option, ToolChecker, Typer,
    print_blank, print_error, print_header, print_info,
    print_next_steps, print_numbered_list, print_step,
    print_success, print_warning, print_dep_results, run_logged,
)
from sniff.activation import EnvironmentActivator
from sniff.envspec import EnvironmentSpec

from . import get_config, messages as msg
from .deps import check_all

# Long-running subprocess output (conda, cargo) is captured here so an agent
# can read the full log without parsing streaming terminal noise.
INSTALL_LOG_REL = ".apxm/install.log"


def register_commands(app: Typer) -> None:
    """Register install command on the app."""

    @app.command()
    def install(
        check: bool = Option(False, "--check", help="Dry-run: report status without changes"),
        skip_deps: bool = Option(False, "--skip-deps", help="Skip dependency checks"),
        skip_build: bool = Option(False, "--skip-build", help="Skip build step"),
        auto: bool = Option(False, "--auto", "-y", help="Automatic mode (no prompts)"),
    ):
        """Install or update APXM environment.

        Checks all dependencies, reports what's missing, and proceeds with
        whatever stages are possible. Never crashes on missing deps.

        Stages:
          1. Platform detection
          2. Dependency checks
          3. Conda environment
          4. Rust toolchain
          5. Build
          6. Summary
        """
        config = get_config()
        platform = app.platform
        missing: list[str] = []

        log_path = config.apxm_dir / INSTALL_LOG_REL
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")  # fresh log per run

        print_header(msg.HEADER_INSTALL)

        # -- Stage 1: Platform detection ---
        print_step(msg.STAGE_PLATFORM_DETECTION)
        print_info(msg.MSG_OS.format(os_label=f"{platform.os} ({platform.arch})"))
        if platform.distro:
            print_info(msg.MSG_DISTRO.format(distro=platform.distro, version=platform.distro_version or ""))
        if platform.is_wsl:
            print_info(msg.MSG_WSL_DETECTED)
        if platform.pkg_manager:
            print_info(msg.MSG_PKG_MANAGER.format(pkg_manager=platform.pkg_manager))
        print_blank()

        # -- Stage 2: Dependency checks ---
        # Conda and Rust are validated in detail in Stages 3 & 4; skip them
        # here so they don't appear twice in the action-required summary.
        _stage3_4 = {"Mamba/Conda", "Rust (nightly)", "Cargo"}
        if not skip_deps:
            print_step(msg.STAGE_CHECKING_DEPS)
            missing += print_dep_results(check_all(), skip_names=_stage3_4)
            print_blank()

        # -- Stage 3: Conda environment ---
        print_step(msg.STAGE_CONDA_ENV)
        conda_cmd = shutil.which("mamba") or shutil.which("conda")
        if not conda_cmd:
            print_error(msg.MSG_CONDA_NOT_FOUND)
            missing.append("Mamba/Conda")
        else:
            print_success(msg.MSG_CONDA_FOUND.format(cmd=conda_cmd))
            env_yaml = config.apxm_dir / "environment.yaml"
            if not env_yaml.exists():
                print_error(msg.MSG_CONDA_ENV_YAML_NOT_FOUND.format(path=env_yaml))
                missing.append("environment.yaml")
            elif check:
                detector = CondaDetector()
                prefix = detector.find_prefix("apxm", probe_common=True)
                if prefix:
                    print_success(msg.MSG_CONDA_ENV_OK.format(prefix=prefix))
                else:
                    print_warning(msg.MSG_CONDA_ENV_NOT_FOUND_WILL_CREATE)
                    missing.append(msg.MSG_CONDA_ENV_NOT_FOUND_RUN_INSTALL)
            else:
                env_name = "apxm"
                env_prefix = _find_conda_env_prefix(conda_cmd, env_name)
                env_valid = env_prefix is not None and (env_prefix / "conda-meta").is_dir()
                env_broken = env_prefix is not None and not env_valid

                if env_broken:
                    print_warning(f"Removing broken environment: {env_prefix}")
                    shutil.rmtree(env_prefix, ignore_errors=True)
                    env_valid = False

                if env_valid:
                    conda_result = run_logged(
                        [conda_cmd, "env", "update", "-f", str(env_yaml), "-n", env_name],
                        log_path=log_path, label="Conda env update",
                        spinner_text="Updating conda environment...", append=True,
                    )
                    conda_ok_msg = msg.MSG_CONDA_ENV_UPDATED
                else:
                    conda_result = run_logged(
                        [conda_cmd, "env", "create", "-f", str(env_yaml)],
                        log_path=log_path, label="Conda env create",
                        spinner_text="Creating conda environment...", append=True,
                    )
                    conda_ok_msg = msg.MSG_CONDA_ENV_CREATED

                if conda_result.ok:
                    print_success(conda_ok_msg)
                else:
                    print_error(msg.MSG_CONDA_CREATE_FAILED.format(cmd=conda_cmd))
                    missing.append(msg.MSG_CONDA_ENV_CREATE_FAILED)
        print_blank()

        # -- Stage 4: Rust toolchain ---
        print_step(msg.STAGE_RUST_TOOLCHAIN)
        checker = ToolChecker()
        if shutil.which("rustup"):
            print_success(msg.MSG_RUSTUP_FOUND)
            if check:
                rustc_ver = checker.get_version("rustc", pattern=r"rustc (\S+)") or "unknown"
                if "nightly" in rustc_ver:
                    print_success(msg.MSG_RUST_VERSION.format(version=rustc_ver))
                else:
                    print_warning(msg.MSG_RUST_NIGHTLY_RECOMMENDED.format(version=rustc_ver))
                    missing.append(msg.MSG_RUST_NIGHTLY_MISSING)
            else:
                if checker.get_version("rustc", pattern=r"nightly"):
                    print_success(msg.MSG_RUST_NIGHTLY_INSTALLED)
                else:
                    run_logged(
                        ["rustup", "toolchain", "install", "nightly"],
                        log_path=log_path, label="Rustup install nightly",
                        spinner_text="Installing Rust nightly toolchain...", append=True,
                    )
                    print_success("Rust nightly installed")
        else:
            print_error(msg.MSG_RUSTUP_NOT_FOUND)
            missing.append("rustup")
            if not check and auto:
                run_logged(
                    ["sh", "-c",
                     "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
                     " | sh -s -- -y --default-toolchain nightly"],
                    log_path=log_path, label="Rustup install",
                    spinner_text="Installing rustup and Rust nightly...", append=True,
                )
                print_success("Rustup installed")
        print_blank()

        # -- Stage 5: Build ---
        if not skip_build:
            print_step(msg.STAGE_BUILD)
            if check:
                if config.compiler_bin.exists():
                    print_success(msg.MSG_COMPILER_BIN_OK_SHORT.format(path=config.compiler_bin))
                else:
                    print_warning(msg.MSG_COMPILER_NOT_BUILT_YET)
                    missing.append(msg.MSG_BUILD_CONDA_ACTIVATE)
            elif not shutil.which("cargo"):
                print_warning(msg.MSG_CARGO_NOT_AVAILABLE)
                missing.append(msg.MSG_BUILD_INSTALL_RUST)
            else:
                # Use sniff's EnvironmentActivator to get build env vars
                # (auto_activate may not have run if conda was just created)
                _PREPEND_VARS = {"PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PYTHONPATH", "PKG_CONFIG_PATH"}
                spec_file = config.apxm_dir / ".sniff.toml"
                spec = None
                env = None
                if spec_file.exists():
                    try:
                        spec = EnvironmentSpec.from_file(spec_file)
                        activator = EnvironmentActivator(spec, config.apxm_dir)
                        result = activator.activate(use_cache=False)
                        if result.env_vars:
                            env = dict(os.environ)
                            for key, value in result.env_vars.items():
                                if key in _PREPEND_VARS:
                                    current = env.get(key, "")
                                    env[key] = f"{value}:{current}" if current else value
                                else:
                                    env[key] = value
                    except Exception:
                        pass  # Fall through to build without custom env

                print_info("Building APXM (this may take a few minutes)...")
                build = run_logged(
                    ["cargo", "build", "-p", "apxm-cli",
                     "--features", "driver,metrics", "--release"],
                    log_path=log_path, label="Cargo build",
                    spinner_text="Building APXM...",
                    env=env, cwd=config.apxm_dir, append=True,
                )
                if build.ok:
                    print_success(msg.MSG_BUILD_COMPLETE)
                    binary_path = config.apxm_dir / "target" / "release" / "apxm"
                    if binary_path.exists():
                        try:
                            if spec is None:
                                spec = EnvironmentSpec.from_file(spec_file)
                            conda_prefix = env.get("CONDA_PREFIX") if env else None
                            res = BinaryInstaller(config.apxm_dir).install_wrapper(
                                target=config.apxm_dir / "tools" / "apxm_cli.py",
                                spec=spec,
                                python=Path(conda_prefix) / "bin" / "python3" if conda_prefix else None,
                                name="apxm",
                            )
                            print_success(res.message)
                        except Exception as e:
                            print_warning(f"Wrapper install failed: {e}")
                else:
                    print_error(msg.MSG_BUILD_FAILED)
                    missing.append(msg.MSG_BUILD_FIX_RETRY)
            print_blank()

        # -- Stage 6: Summary ---
        if missing:
            print_header(msg.HEADER_ACTION_REQUIRED)
            print_numbered_list(missing)
            print_blank()
            print_info(msg.MSG_FIX_RERUN)
            raise Exit(1)
        else:
            print_success(msg.MSG_EVERYTHING_OK)
            print_next_steps(["apxm doctor"])
