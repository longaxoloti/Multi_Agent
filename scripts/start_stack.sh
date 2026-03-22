#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -f ".env" ]; then
  echo "Missing .env file"
  exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ''|'#'*) continue ;;
  esac
  key="${line%%=*}"
  value="${line#*=}"
  if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    export "$key=$value"
  fi
done < .env

eval "$(conda shell.bash hook)"
conda activate agent-stock

mkdir -p data/logs

if [ "${CAMOUFOX_ENABLED:-true}" = "true" ]; then
  echo "Camoufox mode: CAMOFOX_MCP_TRANSPORT=${CAMOFOX_MCP_TRANSPORT:-stdio}"
  "$PROJECT_DIR/scripts/start_camoufox.sh"
fi

python -m main.main > data/logs/bot.log 2>&1 &
BOT_PID=$!
echo "$BOT_PID" > data/logs/bot.pid

"$PROJECT_DIR/scripts/start_airflow.sh"

echo "Stack started"
echo "  Bot PID: $BOT_PID"
