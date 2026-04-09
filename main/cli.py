import os
import signal
import subprocess
import sys
import time
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


def _read_pid_file(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        value = pid_path.read_text().strip()
        if value.isdigit():
            return int(value)
    except OSError:
        return None
    return None


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, grace_seconds: int = 3) -> tuple[bool, str]:
    if pid <= 0:
        return False, "invalid pid"
    if not _pid_exists(pid):
        return False, "not running"

    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        return False, "permission denied"
    except ProcessLookupError:
        return False, "not running"

    deadline = time.time() + max(grace_seconds, 0)
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True, "terminated"
        time.sleep(0.2)

    if _pid_exists(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except PermissionError:
            return False, "permission denied"
        except ProcessLookupError:
            return True, "terminated"

    return (not _pid_exists(pid), "killed")


def _iter_tesla_pid_files() -> list[Path]:
    return [
        PROJECT_ROOT / "data" / "logs" / "bot.pid",
        PROJECT_ROOT / "data" / "logs" / "camofox.pid",
        PROJECT_ROOT / "data" / "logs" / "camofox_mcp.pid",
        PROJECT_ROOT / "data" / "logs" / "bot.run.lock" / "pid",
        PROJECT_ROOT / "airflow_home" / "webserver.pid",
        PROJECT_ROOT / "airflow_home" / "scheduler.pid",
        LOG_DIR / "bot.instance.lock",
    ]


def _is_tesla_process(command: str) -> bool:
    cmd = command.lower()
    project_root = str(PROJECT_ROOT).lower()
    airflow_home = str((PROJECT_ROOT / "airflow_home")).lower()
    markers = [
        "python -m main.main",
        "python -m main.cli start",
        "tesla start",
        "com.geniuslab.tesla",
        "camofox-mcp",
        "camofox-browser",
    ]

    if project_root in cmd or airflow_home in cmd:
        return True

    if any(marker in cmd for marker in markers):
        return True

    # Airflow processes can be started outside project root, but are still Tesla-related.
    if ("airflow webserver" in cmd or "airflow scheduler" in cmd) and "multi_agent" in cmd:
        return True

    return False


def _collect_tesla_processes() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    processes: list[tuple[int, str]] = []
    if result.returncode != 0:
        return processes

    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        command = parts[1]
        if pid == current_pid:
            continue
        if _is_tesla_process(command):
            processes.append((pid, command))
    return processes


@app.command()
def stop(grace: int = typer.Option(3, "--grace", min=0, help="Grace period in seconds before force kill.")):
    """Stop all Tesla-related processes and services."""
    typer.echo("Stopping Tesla services and background processes...")

    stopped: list[str] = []
    skipped: list[str] = []

    plist_path = USER_HOME / "Library" / "LaunchAgents" / "com.geniuslab.tesla.plist"
    if plist_path.exists():
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
        subprocess.run(["launchctl", "remove", "com.geniuslab.tesla"], check=False)
        stopped.append("launchd service")
    else:
        skipped.append("launchd service (plist not found)")

    stop_stack_script = PROJECT_ROOT / "scripts" / "stop_stack.sh"
    if stop_stack_script.exists():
        subprocess.run(["bash", str(stop_stack_script)], cwd=str(PROJECT_ROOT), check=False)
        stopped.append("project stack script")
    else:
        skipped.append("project stack script (not found)")

    db_compose = DB_DIR / "docker-compose.yml"
    if db_compose.exists():
        subprocess.run(["docker", "compose", "down", "--remove-orphans"], cwd=str(DB_DIR), check=False)
        stopped.append("database containers")
    else:
        skipped.append("database containers (compose not found)")

    pid_targets: set[int] = set()
    for pid_file in _iter_tesla_pid_files():
        pid = _read_pid_file(pid_file)
        if pid:
            pid_targets.add(pid)

    for pid, _ in _collect_tesla_processes():
        pid_targets.add(pid)

    terminated_count = 0
    for pid in sorted(pid_targets):
        ok, _ = _terminate_pid(pid, grace_seconds=grace)
        if ok:
            terminated_count += 1

    lock_dir = PROJECT_ROOT / "data" / "logs" / "bot.run.lock"
    if lock_dir.exists():
        try:
            for child in lock_dir.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
            lock_dir.rmdir()
        except OSError:
            pass

    typer.secho(
        f"Tesla stop completed. Terminated {terminated_count} process(es).",
        fg=typer.colors.GREEN,
    )
    if stopped:
        typer.echo("Stopped components: " + ", ".join(stopped))
    if skipped:
        typer.echo("Skipped: " + ", ".join(skipped))

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
