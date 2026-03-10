"""Register command for APXM CLI (delegates to Rust binary)."""

import subprocess
from typing import Optional

import typer

from apxm_styles import print_error, print_info

from tools.scripts import get_config


def create_app() -> typer.Typer:
    """Create the register sub-app."""
    register_app = typer.Typer(help="Manage API credentials")

    def _run_register(args: list[str]) -> None:
        """Delegate to the Rust binary's register subcommand."""
        config = get_config()
        if not config.compiler_bin.exists():
            print_error("APXM binary not built!")
            print_info("Run: apxm build")
            raise typer.Exit(1)
        cmd = [str(config.compiler_bin), "register"] + args
        env = config.get_mlir_env()
        result = subprocess.run(cmd, env=env)
        raise typer.Exit(result.returncode)

    @register_app.command("add")
    def add(
        name: str = typer.Argument(..., help="Unique name for this credential"),
        provider: str = typer.Option(..., help="Provider type (openai, anthropic, google, ollama, openrouter)"),
        api_key: Optional[str] = typer.Option(None, help="API key (omit to enter interactively)"),
        base_url: Optional[str] = typer.Option(None, help="Custom base URL endpoint"),
        model: Optional[str] = typer.Option(None, help="Default model name"),
        header: Optional[list[str]] = typer.Option(
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
        name: str = typer.Argument(..., help="Name of the credential to remove"),
    ):
        """Remove a credential by name."""
        _run_register(["remove", name])

    @register_app.command("test")
    def test_cred(
        name: str = typer.Argument(..., help="Name of the credential to test"),
    ):
        """Test a credential by making a minimal API call."""
        _run_register(["test", name])

    @register_app.command("generate-config")
    def generate_config():
        """Generate config.toml entries from registered credentials."""
        _run_register(["generate-config"])

    return register_app
