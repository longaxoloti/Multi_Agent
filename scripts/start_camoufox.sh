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
CAMOFOX_MCP_TRANSPORT="${CAMOFOX_MCP_TRANSPORT:-stdio}"
CAMOFOX_MCP_COMMAND="${CAMOFOX_MCP_COMMAND:-npx}"
CAMOFOX_MCP_ARGS="${CAMOFOX_MCP_ARGS:--y camofox-mcp@latest}"
CAMOFOX_AUTH_REQUIRED="${CAMOFOX_AUTH_REQUIRED:-true}"
CAMOFOX_API_KEY="${CAMOFOX_API_KEY:-}"

LOG_DIR="$PROJECT_DIR/data/logs"
PID_FILE="$LOG_DIR/camofox.pid"
LOG_FILE="$LOG_DIR/camofox.log"
MCP_PID_FILE="$LOG_DIR/camofox_mcp.pid"
MCP_LOG_FILE="$LOG_DIR/camofox_mcp.log"

mkdir -p "$LOG_DIR"

if [ "$CAMOFOX_AUTH_REQUIRED" = "true" ] && [ -z "$CAMOFOX_API_KEY" ]; then
	echo "CAMOFOX_AUTH_REQUIRED=true but CAMOFOX_API_KEY is empty. Refusing to start insecurely."
	exit 1
fi

if curl -fsS -m 2 "$CAMOUFOX_API_URL/health" >/dev/null 2>&1; then
	echo "camofox server already healthy at $CAMOUFOX_API_URL"
else
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
	nohup env \
		MAX_CONCURRENT_PER_USER="$MAX_CONCURRENT_PER_USER" \
		CAMOFOX_API_KEY="$CAMOFOX_API_KEY" \

		npm start > "$LOG_FILE" 2>&1 &
	CAMOFOX_PID=$!
	echo "$CAMOFOX_PID" > "$PID_FILE"

	for _ in $(seq 1 30); do
		if curl -fsS -m 2 "$CAMOUFOX_API_URL/health" >/dev/null 2>&1; then
			echo "camofox started (pid=$CAMOFOX_PID) at $CAMOUFOX_API_URL"
			break
		fi
		sleep 1
	done

	if ! curl -fsS -m 2 "$CAMOUFOX_API_URL/health" >/dev/null 2>&1; then
		echo "camofox failed to become healthy. Check $LOG_FILE"
		exit 1
	fi
fi

if [ "$CAMOFOX_MCP_TRANSPORT" = "http" ]; then
	if [ -f "$MCP_PID_FILE" ]; then
		OLD_MCP_PID="$(cat "$MCP_PID_FILE" 2>/dev/null || true)"
		if [ -n "$OLD_MCP_PID" ] && kill -0 "$OLD_MCP_PID" >/dev/null 2>&1; then
			echo "camofox-mcp http already running (pid=$OLD_MCP_PID)"
			exit 0
		fi
		rm -f "$MCP_PID_FILE"
	fi

	cd "$PROJECT_DIR"
	read -r -a CAMOFOX_MCP_ARGS_ARR <<< "$CAMOFOX_MCP_ARGS"
	nohup env \
		CAMOFOX_TRANSPORT=http \
		CAMOFOX_URL="$CAMOUFOX_API_URL" \
		CAMOFOX_API_KEY="${CAMOFOX_API_KEY:-}" \

		"$CAMOFOX_MCP_COMMAND" "${CAMOFOX_MCP_ARGS_ARR[@]}" > "$MCP_LOG_FILE" 2>&1 &
	MCP_PID=$!
	echo "$MCP_PID" > "$MCP_PID_FILE"

	for _ in $(seq 1 10); do
		if kill -0 "$MCP_PID" >/dev/null 2>&1; then
			sleep 1
			continue
		fi
		echo "camofox-mcp http exited early. Check $MCP_LOG_FILE"
		exit 1
	done

	echo "camofox-mcp http started (pid=$MCP_PID). Logs: $MCP_LOG_FILE"
	if [ -n "${CAMOFOX_MCP_URL:-}" ]; then
		echo "camofox-mcp endpoint configured at ${CAMOFOX_MCP_URL}"
	fi
else
	echo "camofox-mcp transport=stdio (process spawned by Python MCP client)"
fi

exit 0
