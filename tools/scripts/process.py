"""Subprocess utilities for APXM CLI."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    capture: bool = False,
    timeout: Optional[int] = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command with optional real-time streaming output.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        env: Environment variables (merged with current env).
        capture: If True, capture output instead of streaming.
        timeout: Timeout in seconds.
        verbose: If True, print the command before running.

    Returns:
        CompletedProcess with return code and optional output.
    """
    if verbose:
        print(f"  $ {' '.join(cmd)}", file=sys.stderr)

    run_env = None
    if env:
        run_env = os.environ.copy()
        run_env.update(env)

    if capture:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    # Streaming mode: use Popen for real-time output
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
        proc.wait(timeout=timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout="".join(output_lines),
            stderr="",
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        raise


def run_cargo(
    args: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    release: bool = True,
    features: Optional[list[str]] = None,
    package: Optional[str] = None,
    capture: bool = False,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run a cargo command with common options.

    Args:
        args: Cargo subcommand and extra arguments (e.g., ["build"]).
        cwd: Working directory (defaults to project root).
        env: Extra environment variables.
        release: Add --release flag.
        features: Cargo features to enable.
        package: Target package (-p flag).
        capture: Capture output instead of streaming.
        verbose: Print command before running.

    Returns:
        CompletedProcess.
    """
    cmd = ["cargo"] + args

    if package:
        cmd.extend(["-p", package])
    if features:
        cmd.extend(["--features", ",".join(features)])
    if release:
        cmd.append("--release")

    return run_command(cmd, cwd=cwd, env=env, capture=capture, verbose=verbose)
