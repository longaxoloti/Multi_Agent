#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

kill_if_running() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    PID=$(cat "$pid_file")
    if kill -0 "$PID" >/dev/null 2>&1; then
      kill "$PID"
      echo "Stopped PID $PID"
    fi
    rm -f "$pid_file"
  fi
}

kill_if_running data/logs/bot.pid
kill_if_running data/logs/camofox.pid

"$PROJECT_DIR/scripts/stop_airflow.sh"

echo "Stack stop complete"
