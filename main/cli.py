import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

import typer

app = typer.Typer(help="Tesla: Multi-Model AI Agent System")

USER_HOME = Path.home()
APP_DIR = USER_HOME / ".tesla"
DB_DIR = APP_DIR / "db"
LOG_DIR = APP_DIR / "logs"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from main import __version__
from main.config import OLLAMA_BASE_URL, OLLAMA_ENABLED

OLLAMA_PID_FILE = LOG_DIR / "ollama.pid"
OLLAMA_BOOT_LOG_FILE = LOG_DIR / "ollama.log"


def _has_command(name: str) -> bool:
    result = subprocess.run(["which", name], capture_output=True, text=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


def _resolve_compose_command() -> list[str] | None:
    """Return a usable compose command: docker compose (preferred) or docker-compose."""
    if _has_command("docker"):
        probe = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return ["docker", "compose"]

    if _has_command("docker-compose"):
        probe = subprocess.run(
            ["docker-compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return ["docker-compose"]

    return None


def _colima_context_name() -> str:
    return "colima"


def _colima_is_ready() -> bool:
    if not _has_command("colima"):
        return False

    probe = subprocess.run(["colima", "status"], capture_output=True, text=True, check=False)
    output = f"{probe.stdout}\n{probe.stderr}".lower()
    return probe.returncode == 0 and "running" in output


def _ensure_colima_running(timeout_seconds: int = 120) -> None:
    if not _has_command("colima"):
        raise RuntimeError("Colima is not installed. Install Colima and retry tesla start.")

    if not _colima_is_ready():
        typer.echo("Colima is not running. Starting Colima...")
        start_result = subprocess.run(["colima", "start"], capture_output=True, text=True, check=False)
        if start_result.returncode != 0:
            stderr_text = (start_result.stderr or start_result.stdout or "").strip()
            raise RuntimeError(f"Failed to start Colima: {stderr_text or 'unknown colima error'}")

    deadline = time.time() + max(timeout_seconds, 0)
    while time.time() < deadline:
        if _colima_is_ready():
            break
        time.sleep(2)
    else:
        raise RuntimeError("Colima did not become ready in time. Please retry tesla start after Colima finishes booting.")

    if _has_command("docker"):
        context_result = subprocess.run(
            ["docker", "context", "use", _colima_context_name()],
            capture_output=True,
            text=True,
            check=False,
        )
        if context_result.returncode != 0:
            stderr_text = (context_result.stderr or context_result.stdout or "").strip()
            raise RuntimeError(f"Failed to switch Docker context to Colima: {stderr_text or 'unknown docker error'}")

    probe = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        stderr_text = (probe.stderr or probe.stdout or "").strip()
        raise RuntimeError(f"Docker daemon is not available through Colima: {stderr_text or 'unknown docker error'}")


def _ensure_local_data_dirs() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _ollama_health_url() -> str:
    base = (OLLAMA_BASE_URL or "http://127.0.0.1:11434").rstrip("/")
    return f"{base}/api/tags"


def _ollama_is_reachable(timeout_seconds: float = 2.0) -> bool:
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(_ollama_health_url())
            return resp.status_code < 400
    except Exception:
        return False


def _read_pid_from_file(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        raw = pid_file.read_text().strip()
    except OSError:
        return None
    if raw.isdigit():
        return int(raw)
    return None


def _collect_ollama_server_pids() -> list[int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    pids: list[int] = []
    for line in result.stdout.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        cmd = parts[1].lower()
        if "ollama serve" in cmd:
            pids.append(pid)
    return pids


def _wait_for_ollama_ready(timeout_seconds: int = 25) -> bool:
    deadline = time.time() + max(timeout_seconds, 0)
    while time.time() < deadline:
        if _ollama_is_reachable():
            return True
        time.sleep(1)
    return False


def _start_ollama_if_needed() -> None:
    if not OLLAMA_ENABLED:
        return
    if _ollama_is_reachable():
        typer.echo("Ollama is already running.")
        return

    if not _has_command("ollama"):
        raise RuntimeError(
            "Ollama is enabled but 'ollama' command is not available. Install Ollama or set OLLAMA_ENABLED=false."
        )

    _ensure_local_data_dirs()
    typer.echo("Ollama is not reachable. Starting ollama service...")

    with OLLAMA_BOOT_LOG_FILE.open("a", encoding="utf-8") as logf:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )

    OLLAMA_PID_FILE.write_text(str(process.pid))
    if not _wait_for_ollama_ready(timeout_seconds=30):
        try:
            os.kill(process.pid, signal.SIGTERM)
        except OSError:
            pass
        raise RuntimeError(
            "Started 'ollama serve' but endpoint did not become ready in time. Check ~/.tesla/logs/ollama.log."
        )

    parsed = urlparse((OLLAMA_BASE_URL or "").strip() or "http://127.0.0.1:11434")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    typer.secho(f"Ollama started and reachable at {host}:{port}.", fg=typer.colors.GREEN)


def _stop_ollama_service(grace_seconds: int = 3) -> tuple[int, int]:
    targets: set[int] = set()

    pid_from_file = _read_pid_from_file(OLLAMA_PID_FILE)
    if pid_from_file:
        targets.add(pid_from_file)

    for pid in _collect_ollama_server_pids():
        targets.add(pid)

    stopped = 0
    for pid in sorted(targets):
        ok, reason = _terminate_pid(pid, grace_seconds=grace_seconds)
        if ok or reason in {"not running", "killed"}:
            stopped += 1

    if OLLAMA_PID_FILE.exists():
        OLLAMA_PID_FILE.unlink(missing_ok=True)

    return stopped, len(targets)


def _sync_db_compose_file() -> Path | None:
    source_dc = PROJECT_ROOT / "infra" / "db" / "docker-compose.yml"
    if not source_dc.exists():
        return None
    target_dc = DB_DIR / "docker-compose.yml"
    target_dc.write_text(source_dc.read_text())
    return target_dc


def _bootstrap_runtime(*, migrate: bool) -> None:
    _ensure_local_data_dirs()
    _ensure_colima_running()

    compose_cmd = _resolve_compose_command()
    if compose_cmd is None:
        raise RuntimeError(
            "No usable Docker Compose command found. Install Docker Compose and ensure either 'docker compose' or 'docker-compose' works in PATH."
        )

    target_dc = _sync_db_compose_file()
    if target_dc is None:
        raise RuntimeError(f"Database compose file not found at {PROJECT_ROOT / 'infra' / 'db' / 'docker-compose.yml'}")

    typer.echo("Ensuring database containers are running through Colima...")
    try:
        subprocess.run([*compose_cmd, "up", "-d"], cwd=str(DB_DIR), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to start database containers through Colima. Ensure Colima is running, then retry tesla start/init."
        ) from exc

    if migrate:
        typer.echo("Applying database schemas via Alembic...")
        alembic_ini = PROJECT_ROOT / "infra" / "db" / "alembic.ini"
        if not alembic_ini.exists():
            raise RuntimeError(f"Alembic config not found at {alembic_ini}")

        subprocess.run(
            ["alembic", "-c", str(alembic_ini), "upgrade", "head"],
            cwd=str(PROJECT_ROOT / "infra" / "db"),
            check=True,
        )

@app.command()
def init():
    """Initialize the local application data and database."""
    typer.echo(f"Initializing Tesla at {APP_DIR}...")
    _ensure_local_data_dirs()

    typer.echo("Applying database schemas via Alembic...")
    alembic_ini = PROJECT_ROOT / "infra" / "db" / "alembic.ini"
    if not alembic_ini.exists():
        raise RuntimeError(f"Alembic config not found at {alembic_ini}")

    subprocess.run(
        ["alembic", "-c", str(alembic_ini), "upgrade", "head"],
        cwd=str(PROJECT_ROOT / "infra" / "db"),
        check=True,
    )
    
    typer.secho("Tesla Agent initialized successfully!", fg=typer.colors.GREEN)

@app.command()
def start():
    """Start the foreground agent process."""
    typer.echo("Starting Tesla Agent...")
    _bootstrap_runtime(migrate=False)
    _start_ollama_if_needed()
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

    stop_stack_script = PROJECT_ROOT / "scripts" / "stop_stack.sh"
    if stop_stack_script.exists():
        subprocess.run(["bash", str(stop_stack_script)], cwd=str(PROJECT_ROOT), check=False)
        stopped.append("project stack script")
    else:
        skipped.append("project stack script (not found)")

    db_compose_dirs = [DB_DIR, PROJECT_ROOT / "infra" / "db"]
    compose_cmd = _resolve_compose_command()
    if compose_cmd is None:
        skipped.append("database containers (docker compose command not found)")
    else:
        db_stopped = False
        for compose_dir in db_compose_dirs:
            db_compose = compose_dir / "docker-compose.yml"
            if not db_compose.exists():
                continue
            subprocess.run([*compose_cmd, "down", "--remove-orphans"], cwd=str(compose_dir), check=False)
            db_stopped = True
        if db_stopped:
            stopped.append("database containers")
        else:
            skipped.append("database containers (compose not found)")

        if OLLAMA_ENABLED:
            ollama_stopped, ollama_targets = _stop_ollama_service(grace_seconds=grace)
            if ollama_targets > 0:
                stopped.append(f"ollama service ({ollama_stopped}/{ollama_targets} process(es))")
            else:
                skipped.append("ollama service (no running ollama serve process found)")

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
    """Deprecated: Tesla no longer manages launchd background services."""
    _ = action
    typer.echo(
        "`tesla daemon` has been deprecated. Tesla now runs foreground-only via `tesla start` in your terminal."
    )
    typer.echo(
        "If you previously installed launchd, remove it manually with: rm ~/Library/LaunchAgents/com.geniuslab.tesla.plist"
    )

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
