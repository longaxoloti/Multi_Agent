#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def main() -> int:
    data_logs = PROJECT_ROOT / "data" / "logs"
    airflow_home = Path(os.getenv("AIRFLOW_HOME", str(PROJECT_ROOT / "airflow_home")))

    bot = read_pid_status(data_logs / "bot.pid")
    airflow_web = read_pid_status(airflow_home / "webserver.pid")
    airflow_sched = read_pid_status(airflow_home / "scheduler.pid")
    airflow_api = airflow_health()
    camofox_api_url = os.getenv("CAMOUFOX_API_URL", "http://127.0.0.1:9377")
    camofox_api = http_health(f"{camofox_api_url.rstrip('/')}/health")

    status = {
        "bot": bot,
        "airflow_webserver": airflow_web,
        "airflow_scheduler": airflow_sched,
        "airflow_api": airflow_api,
        "camofox_api": camofox_api,
    }

    print(json.dumps(status, indent=2))

    mandatory = [bot.get("alive", False), airflow_web.get("alive", False), airflow_sched.get("alive", False)]
    if all(mandatory):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
