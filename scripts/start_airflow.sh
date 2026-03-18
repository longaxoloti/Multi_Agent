#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export AIRFLOW_HOME="${AIRFLOW_HOME:-$PROJECT_DIR/airflow_home}"
mkdir -p "$AIRFLOW_HOME" "$AIRFLOW_HOME/logs"

if [ ! -f ".env" ]; then
  echo "Missing .env file"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate agent-stock

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

airflow db migrate

airflow webserver --port 8080 > "$AIRFLOW_HOME/logs/webserver.log" 2>&1 &
WEBSERVER_PID=$!
airflow scheduler > "$AIRFLOW_HOME/logs/scheduler.log" 2>&1 &
SCHEDULER_PID=$!

echo "$WEBSERVER_PID" > "$AIRFLOW_HOME/webserver.pid"
echo "$SCHEDULER_PID" > "$AIRFLOW_HOME/scheduler.pid"

echo "Airflow started"
echo "  Webserver PID: $WEBSERVER_PID"
echo "  Scheduler PID: $SCHEDULER_PID"
echo "  UI: http://localhost:8080"
