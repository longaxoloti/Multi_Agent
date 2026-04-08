import os
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Tesla: Multi-Model AI Agent System")

USER_HOME = Path.home()
APP_DIR = USER_HOME / ".tesla"
DB_DIR = APP_DIR / "db"
LOG_DIR = APP_DIR / "logs"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from main import __version__

@app.command()
def init():
    """Initialize the local application data and database."""
    typer.echo(f"Initializing Tesla at {APP_DIR}...")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Note: Copying the docker_compose structure into ~/.tesla/db
    source_dc = PROJECT_ROOT / "infra" / "db" / "docker-compose.yml"
    target_dc = DB_DIR / "docker-compose.yml"
    
    if source_dc.exists():
        target_dc.write_text(source_dc.read_text())
        typer.echo("Starting database containers...")
        subprocess.run(["docker", "compose", "up", "-d"], cwd=str(DB_DIR), check=True)
    else:
        typer.echo(f"Warning: {source_dc} not found.", err=True)

    # Initialize alembic seamlessly natively from the project.
    typer.echo("Applying database schemas via Alembic...")
    alembic_ini = PROJECT_ROOT / "infra" / "db" / "alembic.ini"
    if alembic_ini.exists():
        subprocess.run(["alembic", "-c", str(alembic_ini), "upgrade", "head"], cwd=str(PROJECT_ROOT / "infra" / "db"), check=True)
    
    typer.secho("Tesla Agent initialized successfully!", fg=typer.colors.GREEN)

@app.command()
def start():
    """Start the foreground agent process."""
    typer.echo("Starting Tesla Agent...")
    # Delay import to avoid loading everything on fast CLI commands
    from main.main import main
    main()

@app.command()
def log():
    """Tail the agent logs."""
    log_file = LOG_DIR / "agent.log"
    if not log_file.exists():
        typer.echo("No logs found yet.")
        raise typer.Exit(1)
    # Using python's os.execvp to replace current process with tail
    os.execvp("tail", ["tail", "-f", str(log_file)])

@app.command()
def daemon(action: str = typer.Argument(..., help="'install', 'start', 'stop', or 'status'")):
    """Manage the background macOS launchd service."""
    plist_path = USER_HOME / "Library" / "LaunchAgents" / "com.geniuslab.tesla.plist"
    
    if action == "install":
        python_bin = sys.executable
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.geniuslab.tesla</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_bin}</string>
        <string>-m</string>
        <string>main.cli</string>
        <string>start</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{LOG_DIR}/agent.out.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR}/agent.err.log</string>
</dict>
</plist>"""
        plist_path.write_text(plist_content)
        typer.echo(f"Installed daemon launchd plist to {plist_path}")

    elif action == "start":
        typer.echo("Starting Tesla daemon...")
        subprocess.run(["launchctl", "load", str(plist_path)])
    
    elif action == "stop":
        typer.echo("Stopping Tesla daemon...")
        subprocess.run(["launchctl", "unload", str(plist_path)])

    elif action == "status":
        subprocess.run(["launchctl", "list", "com.geniuslab.tesla"])
    
    else:
        typer.echo(f"Unknown action: {action}. Use install, start, stop, or status.", err=True)

@app.command()
def version(bump: str = typer.Option(None, "--bump", help="Bump version: 'patch', 'minor', or 'major'")):
    """Show or bump the version of Tesla."""
    if bump:
        if bump not in ["patch", "minor", "major"]:
            typer.secho("Invalid bump type. Use 'patch', 'minor', or 'major'.", fg=typer.colors.RED)
            raise typer.Exit(1)
        
        typer.echo(f"Bumping {bump} version...")
        subprocess.run(["bumpver", "update", "--" + bump], cwd=str(PROJECT_ROOT), check=True)
    else:
        typer.echo(f"Tesla Version: {__version__}")

if __name__ == "__main__":
    app()
