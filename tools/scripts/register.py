"""Register command for APXM CLI (delegates to Rust binary)."""

import subprocess
from typing import Optional

from sniff import Typer, Argument, Option, Exit

from . import get_config, ensure_binary


def create_app() -> Typer:
    """Create the register sub-app."""
    register_app = Typer(help="Manage API credentials")

    def _run_register(args: list[str]) -> None:
        """Delegate to the Rust binary's register subcommand."""
        config = get_config()
        ensure_binary(config)
        cmd = [str(config.compiler_bin), "register"] + args
        result = subprocess.run(cmd)
        raise Exit(result.returncode)

    @register_app.command("add")
    def add(
        name: str = Argument(..., help="Unique name for this credential"),
        provider: str = Option(..., help="Provider type (openai, anthropic, google, ollama, openrouter)"),
        api_key: Optional[str] = Option(None, help="API key (omit to enter interactively)"),
        base_url: Optional[str] = Option(None, help="Custom base URL endpoint"),
        model: Optional[str] = Option(None, help="Default model name"),
        header: Optional[list[str]] = Option(
            None, help='Extra HTTP headers (repeatable: --header "Key=Value")'
        ),
    ):
        """Register a new API credential."""
        args = ["add", name, "--provider", provider]
        if api_key:
            args.extend(["--api-key", api_key])
        if base_url:
            args.extend(["--base-url", base_url])
        if model:
            args.extend(["--model", model])
        if header:
            for h in header:
                args.extend(["--header", h])
        _run_register(args)

    @register_app.command("list")
    def list_creds():
        """List all registered credentials."""
        _run_register(["list"])

    @register_app.command("remove")
    def remove(
        name: str = Argument(..., help="Name of the credential to remove"),
    ):
        """Remove a credential by name."""
        _run_register(["remove", name])

    @register_app.command("test")
    def test_cred(
        name: str = Argument(..., help="Name of the credential to test"),
    ):
        """Test a credential by making a minimal API call."""
        _run_register(["test", name])

    return register_app
