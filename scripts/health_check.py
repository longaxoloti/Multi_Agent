#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid_status(pid_file: Path) -> dict[str, Any]:
    if not pid_file.exists():
        return {"present": False, "alive": False, "pid": None}

    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return {"present": True, "alive": False, "pid": None, "error": "invalid pid file"}

    return {"present": True, "alive": is_pid_alive(pid), "pid": pid}


def airflow_health(url: str = "http://127.0.0.1:8080/health") -> dict[str, Any]:
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=5) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
            if resp.status != 200:
                return {"reachable": False, "status": resp.status}
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return {"reachable": True, "raw": payload[:300]}
            return {"reachable": True, "data": data}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def http_health(url: str) -> dict[str, Any]:
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=5) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
            if resp.status != 200:
                return {"reachable": False, "status": resp.status}
            try:
                data = json.loads(payload)
                return {"reachable": True, "data": data}
            except json.JSONDecodeError:
                return {"reachable": True, "raw": payload[:300]}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def get_ollama_models() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception as exc:
        return {"ok": False, "error": f"failed to execute ollama list: {exc}", "models": []}

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout).strip() or f"ollama list exited with {proc.returncode}",
            "models": [],
        }

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": True, "error": "", "models": []}

    model_names: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            model_names.append(parts[0])
    return {"ok": True, "error": "", "models": model_names}


def prestart_health() -> tuple[dict[str, Any], int]:
    from main.config import (
        CAMOFOX_API_KEY,
        CAMOFOX_AUTH_REQUIRED,
        CAMOFOX_BLOCK_REMOTE,
        CAMOFOX_MCP_TRANSPORT,
        CAMOFOX_REQUIRE_HTTPS_REMOTE,
        OLLAMA_BASE_URL,
        OLLAMA_ENABLED,
        OLLAMA_ORCHESTRATOR_MODEL,
        OLLAMA_RESEARCH_MODEL,
        OLLAMA_CODER_MODEL,
    )

    providers = {
        "research": os.getenv("MODEL_RESEARCH", "ollama").strip().lower(),
        "analysis": os.getenv("MODEL_ANALYSIS", "ollama").strip().lower(),
        "chat": os.getenv("MODEL_CHAT", "ollama").strip().lower(),
        "code": os.getenv("MODEL_CODE", "ollama").strip().lower(),
        "orchestrator": os.getenv("MODEL_ORCHESTRATOR", "ollama").strip().lower(),
    }

    task_model_map = {
        "research": OLLAMA_RESEARCH_MODEL,
        "analysis": OLLAMA_RESEARCH_MODEL,
        "chat": OLLAMA_ORCHESTRATOR_MODEL,
        "code": OLLAMA_CODER_MODEL,
        "orchestrator": OLLAMA_ORCHESTRATOR_MODEL,
    }

    required_ollama_models = sorted(
        {
            task_model_map[task]
            for task, provider in providers.items()
            if provider == "ollama"
        }
    )

    ollama_api = http_health(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")
    ollama_list = get_ollama_models() if OLLAMA_ENABLED else {"ok": True, "models": [], "error": ""}
    installed_set = set(ollama_list.get("models", []))
    missing_models = [m for m in required_ollama_models if m not in installed_set]

    telegram_ok = bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip())

    status = {
        "mode": "prestart",
        "camofox_auth_policy": {
            "auth_required": CAMOFOX_AUTH_REQUIRED,
            "api_key_present": bool((CAMOFOX_API_KEY or "").strip()),
            "block_remote": CAMOFOX_BLOCK_REMOTE,
            "require_https_remote": CAMOFOX_REQUIRE_HTTPS_REMOTE,
            "mcp_transport": CAMOFOX_MCP_TRANSPORT,
        },
        "providers": providers,
        "telegram_token_present": telegram_ok,
        "ollama_enabled": OLLAMA_ENABLED,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_api": ollama_api,
        "ollama_list": ollama_list,
        "required_ollama_models": required_ollama_models,
        "missing_ollama_models": missing_models,
        "suggested_pull_commands": [f"ollama pull {m}" for m in missing_models],
    }

    is_ok = True
    if not telegram_ok:
        is_ok = False
    if CAMOFOX_AUTH_REQUIRED and not (CAMOFOX_API_KEY or "").strip():
        is_ok = False
    if OLLAMA_ENABLED and not ollama_api.get("reachable", False):
        is_ok = False
    if OLLAMA_ENABLED and (not ollama_list.get("ok", False)):
        is_ok = False
    if missing_models:
        is_ok = False

    return status, (0 if is_ok else 1)


def runtime_health() -> tuple[dict[str, Any], int]:
    data_logs = PROJECT_ROOT / "data" / "logs"
    airflow_home = Path(os.getenv("AIRFLOW_HOME", str(PROJECT_ROOT / "airflow_home")))

    bot = read_pid_status(data_logs / "bot.pid")
    airflow_web = read_pid_status(airflow_home / "webserver.pid")
    airflow_sched = read_pid_status(airflow_home / "scheduler.pid")
    airflow_api = airflow_health()
    camofox_api_url = os.getenv("CAMOFOX_URL", os.getenv("CAMOUFOX_API_URL", "http://127.0.0.1:9377"))
    camofox_api = http_health(f"{camofox_api_url.rstrip('/')}/health")
    camofox_mcp_transport = os.getenv("CAMOFOX_MCP_TRANSPORT", "stdio").strip().lower()
    camofox_mcp_url = os.getenv("CAMOFOX_MCP_URL", "http://127.0.0.1:3000/mcp")
    camofox_mcp_api = (
        http_health(camofox_mcp_url)
        if camofox_mcp_transport == "http"
        else {"skipped": True, "reason": "transport=stdio"}
    )

    status = {
        "mode": "runtime",
        "bot": bot,
        "airflow_webserver": airflow_web,
        "airflow_scheduler": airflow_sched,
        "airflow_api": airflow_api,
        "camofox_api": camofox_api,
        "camofox_mcp_transport": camofox_mcp_transport,
        "camofox_mcp_api": camofox_mcp_api,
    }

    mandatory = [
        bot.get("alive", False),
        airflow_web.get("alive", False),
        airflow_sched.get("alive", False),
    ]
    return status, (0 if all(mandatory) else 1)


def main() -> int:
    mode = "runtime"
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        if arg in {"--prestart", "prestart"}:
            mode = "prestart"
        elif arg in {"--runtime", "runtime"}:
            mode = "runtime"
        else:
            print("Usage: python scripts/health_check.py [--prestart|--runtime]")
            return 2

    if mode == "prestart":
        status, code = prestart_health()
    else:
        status, code = runtime_health()

    print(json.dumps(status, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
