#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ -f ".env" ]; then
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
fi

CAMOUFOX_API_URL="${CAMOUFOX_API_URL:-http://127.0.0.1:9377}"
CAMOFOX_PLUGIN_DIR="${CAMOFOX_PLUGIN_DIR:-$HOME/.openclaw/extensions/camofox-browser}"
MAX_CONCURRENT_PER_USER="${MAX_CONCURRENT_PER_USER:-1}"
LOG_DIR="$PROJECT_DIR/data/logs"
PID_FILE="$LOG_DIR/camofox.pid"
LOG_FILE="$LOG_DIR/camofox.log"

mkdir -p "$LOG_DIR"

if curl -fsS -m 2 "$CAMOUFOX_API_URL/health" >/dev/null 2>&1; then
	echo "camofox server already healthy at $CAMOUFOX_API_URL"
	exit 0
fi

if [ ! -d "$CAMOFOX_PLUGIN_DIR" ]; then
	echo "camofox plugin directory not found: $CAMOFOX_PLUGIN_DIR"
	echo "Set CAMOFOX_PLUGIN_DIR or install plugin at ~/.openclaw/extensions/camofox-browser"
	exit 1
fi

if [ -f "$PID_FILE" ]; then
	OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
	if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
		echo "camofox process already running (pid=$OLD_PID)"
	else
		rm -f "$PID_FILE"
	fi
fi

cd "$CAMOFOX_PLUGIN_DIR"
nohup env MAX_CONCURRENT_PER_USER="$MAX_CONCURRENT_PER_USER" npm start > "$LOG_FILE" 2>&1 &
CAMOFOX_PID=$!
echo "$CAMOFOX_PID" > "$PID_FILE"

for _ in $(seq 1 30); do
	if curl -fsS -m 2 "$CAMOUFOX_API_URL/health" >/dev/null 2>&1; then
		echo "camofox started (pid=$CAMOFOX_PID) at $CAMOUFOX_API_URL"
		exit 0
	fi
	sleep 1
done

echo "camofox failed to become healthy. Check $LOG_FILE"
exit 1
