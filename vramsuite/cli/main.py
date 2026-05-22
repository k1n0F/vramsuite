"""
VRAM Suite CLI entrypoint.

Status:
    pre-alpha / project skeleton
"""

import typer
from rich.console import Console

app = typer.Typer(
    help="VRAM Suite CLI",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def main() -> None:
    """
    Predictive GPU memory framework for AI inference workflows.
    """
    pass


@app.command()
def doctor() -> None:
    """Show basic VRAM Suite diagnostic information."""
    console.print("[bold cyan]VRAM Suite Doctor[/bold cyan]")
    console.print("Status: pre-alpha / project skeleton")
    console.print("Next target: v0.1-alpha - system.vramcard generation")


if __name__ == "__main__":
    app()