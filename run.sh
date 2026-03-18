#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Tesla Agent: Startup Script
# Activates the conda environment and launches the system
# ──────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Tesla Agent..."
echo "   Directory: $SCRIPT_DIR"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Copy .env.example to .env and fill in your API keys:"
    echo "   cp .env.example .env"
    exit 1
fi

# Activate conda and run
eval "$(conda shell.bash hook)"
conda activate agent-stock

echo "✅ Conda environment 'agent-stock' activated"
echo "   Python: $(python --version)"
echo ""

MODE="${1:-bot}"
LOCK_DIR="data/logs/bot.run.lock"

cleanup() {
    rm -rf "$LOCK_DIR"
}

trap cleanup EXIT

acquire_bot_lock() {
    mkdir -p data/logs
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "$$" > "$LOCK_DIR/pid"
        return 0
    fi

    if [ -f "$LOCK_DIR/pid" ]; then
        LOCK_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")"
        if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "❌ Another bot launcher is already running (pid=$LOCK_PID)."
            return 1
        fi
    fi

    rm -rf "$LOCK_DIR"
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "$$" > "$LOCK_DIR/pid"
        return 0
    fi

    echo "❌ Could not acquire bot startup lock."
    return 1
}

if [ "$MODE" = "bot" ]; then
    # Launch via Python wrapper
    acquire_bot_lock || exit 1
    python main.py
elif [ "$MODE" = "airflow" ]; then
    ./scripts/start_airflow.sh
elif [ "$MODE" = "stack" ]; then
    ./scripts/start_stack.sh
else
    echo "Usage: ./run.sh [bot|airflow|stack]"
    echo ""
    echo "Modes:"
    echo "  bot      - Start via Python launcher (default)"
    echo "  airflow  - Start Airflow scheduler"
    echo "  stack    - Start full stack (Airflow + crawler)"
    exit 1
fi
