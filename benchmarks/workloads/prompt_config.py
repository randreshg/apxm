#!/usr/bin/env python3
"""
Read system prompts from ~/.apxm/config.toml [instruction] section.

This module provides access to system prompts that are shared between APXM runtime
and LangGraph benchmarks to ensure consistent behavior across both implementations.

Example config.toml:
    [instruction]
    ask = "You are a helpful AI assistant."
    think = "Think step by step."
    reason = "Provide structured reasoning."
    plan = "Create actionable plans."
    reflect = "Analyze execution patterns."
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_CONFIG_CACHE: Optional[dict] = None


def _find_apxm_config() -> Optional[Path]:
    """Find config.toml, checking project dir first, then home dir."""
    cwd = Path.cwd()
    for ancestor in [cwd] + list(cwd.parents):
        candidate = ancestor / ".apxm" / "config.toml"
        if candidate.exists():
            return candidate

    home = Path.home() / ".apxm" / "config.toml"
    if home.exists():
        return home
    return None


def _load_instruction_config() -> dict:
    """Load instruction config from config.toml, with caching."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = _find_apxm_config()
    if path is None:
        raise RuntimeError(
            "No config.toml found. Please create ~/.apxm/config.toml with [instruction] section."
        )

    try:
        import tomllib

        config = tomllib.loads(path.read_text())
        _CONFIG_CACHE = config.get("instruction", {})
        return _CONFIG_CACHE
    except ImportError:
        # Python < 3.11 fallback
        try:
            import tomli as tomllib

            config = tomllib.loads(path.read_text())
            _CONFIG_CACHE = config.get("instruction", {})
            return _CONFIG_CACHE
        except ImportError:
            raise RuntimeError(
                "tomllib (Python 3.11+) or tomli package required to parse config.toml"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to parse config.toml: {e}")


def get_system_prompt(operation: str) -> str:
    """Get system prompt for operation (ask, think, reason, plan, reflect).

    Always reads from config - no defaults.
    Raises KeyError if operation not found in config.

    Args:
        operation: One of "ask", "think", "reason", "plan", "reflect"

    Returns:
        The system prompt string from config

    Raises:
        KeyError: If the operation is not found in [instruction] config
        RuntimeError: If config.toml cannot be found or parsed
    """
    config = _load_instruction_config()
    operation_lower = operation.lower()
    if operation_lower not in config:
        raise KeyError(
            f"Prompt '{operation}' not found in [instruction] config. "
            f"Available prompts: {list(config.keys())}"
        )
    return config[operation_lower]


def get_system_prompt_or_none(operation: str) -> Optional[str]:
    """Get system prompt for operation, returning None if not configured.

    This is a safer version that doesn't raise if the prompt is missing.

    Args:
        operation: One of "ask", "think", "reason", "plan", "reflect"

    Returns:
        The system prompt string from config, or None if not found
    """
    try:
        config = _load_instruction_config()
        return config.get(operation.lower())
    except RuntimeError:
        return None


def clear_cache() -> None:
    """Clear the config cache. Useful for testing."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


if __name__ == "__main__":
    # Simple test
    try:
        for op in ["ask", "think", "reason", "plan", "reflect"]:
            prompt = get_system_prompt_or_none(op)
            if prompt:
                print(f"{op}: {prompt[:50]}...")
            else:
                print(f"{op}: (not configured)")
    except RuntimeError as e:
        print(f"Error: {e}")
