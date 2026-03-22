# Multi Agent

**A self-hosted, multi-model AI agent system built with LangGraph — designed to run entirely on local hardware via Ollama, with optional cloud LLM fallback.**

This project builds an autonomous agent that receives messages through Telegram, uses the Orchestrator to analyze your intent and plan multi-step work across specialized worker models, and synthesizes coherent responses — all while managing RAM on a single machine by loading one model at a time.

---

## Features

- **Multi-Model Orchestration** — Each task type is handled by the most suitable model:
  - Orchestrator — Intent analysis, multi-step planning, and routing
  - Researcher — Web research & deep reasoning
  - Coder — Code generation & debugging

- **LangGraph Workflow** — A stateful graph pipeline:
  ```
  Request Router → Orchestrate → [Research | Coding | Reasoning | Briefing] → Orchestrator(Progress) → ... → Synthesize → END
  ```

- **Smart Web Research** — Anti-detect browsing using Chrome CDP (direct control of your real browser) with fallback to Camoufox + article extraction with Crawl4AI. Includes human-like mouse movements (Bezier curves), smooth scrolling, and visual custom cursors to bypass bot detection.
- **Workspace Priming** — Markdown-based system prompts loaded per model role, fully customizable in the `workspace/` directory

- **RAM-Aware Model Swapping** — Automatically unloads the previous model before loading the next, with context serialized to JSON for seamless handoff

- **Telegram Bot Interface** — Single-instance locking, retry with backoff on Telegram API conflicts

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/longaxoloti/Multi_Agent.git
cd Multi_Agent
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate multi-agent
```

### 3. Configure Environment Variables

Edit `.env` and fill in:
- `TELEGRAM_BOT_TOKEN` — Your Telegram bot token from [@BotFather](https://t.me/BotFather)
- `TELEGRAM_USER_ID` — Your Telegram user ID
- API keys for cloud LLM providers (optional)
- Database URL (optional)

### 4. Run the Agent

```bash
# Start the Telegram bot (default mode)
./main/run.sh

# Or start the full stack (bot + Airflow + crawler)
./main/run.sh stack

# Or start only Airflow scheduler
./main/run.sh airflow
```

### 4.1 Run with Docker

```bash
cp .env.example .env  # or create .env manually if you do not have .env.example

# Build and start bot + Postgres
docker compose up -d --build

# Follow logs
docker compose logs -f multi-agent
```

Default container command runs the Telegram bot via `python -m main.main`.

### 5. Talk to Your Agent

Open Telegram, find your bot, and send `/start`.

---

### Model Routing

You can override which provider handles each task type:

```env
MODEL_RESEARCH      # Research tasks
MODEL_CODE          # Coding tasks
MODEL_CHAT          # Direct chat
MODEL_ORCHESTRATOR
```

Set any of these to `gemini`, `openai`, `anthropic`, or `ollama` if you want to self-host.

---

## Health Check

```bash
# Prestart validation (env + provider routes + Ollama models)
python scripts/health_check.py --prestart
```

---

## Chrome CDP Research Mode (Primary)

The agent now defaults to using Chrome CDP to control your real, daily-use browser. This approach bypasses Google's bot detection 100% by utilizing your real IP, established Cookies/Account history, and injecting human-like browser interactions (smooth scrolling, Bezier-curve mouse paths, and custom visual cursor).

### Setup

1. Completely close Google Chrome on your machine.
2. Watch the agent create a new tab, navigate, smoothly scroll, move its own injected custom red cursor arrow, and click links autonomously.

---

## CamoFox MCP Research Mode (Fallback)

If Chrome CDP is unavailable, the agent will fall back to using `camofox-mcp`.

### Required Components

1. `camofox-browser` must be reachable (`/health` on port `9377` by default).
2. `camofox-mcp` transport can be:
  - `stdio` (default): spawned by Python MCP client.
  - `http`: started as a background process by `scripts/start_camoufox.sh`.

### MCP Environment Variables

Add/update these in `.env`:

```env
# Browser server URL (used by both app and camofox-mcp)
CAMOFOX_URL=http://127.0.0.1:9377
CAMOUFOX_API_URL=http://127.0.0.1:9377

# MCP transport: stdio|http
CAMOFOX_MCP_TRANSPORT=stdio

# For stdio spawn
CAMOFOX_MCP_COMMAND=npx
CAMOFOX_MCP_ARGS=-y camofox-mcp@latest

# For HTTP mode
CAMOFOX_MCP_URL=http://127.0.0.1:3000/mcp

# Optional auth passthrough
CAMOFOX_API_KEY=

# MCP client runtime
CAMOFOX_MCP_TIMEOUT_MS=30000
CAMOFOX_MCP_MAX_RETRIES=2
CAMOFOX_MCP_RETRY_BACKOFF_SECONDS=1.0
```

### Verify MCP Setup

```bash
# Browser health
curl -fsS http://127.0.0.1:9377/health

# MCP smoke (server_status -> create_tab -> navigate/snapshot -> close_tab)
python scripts/e2e_mcp_workflow_smoke.py
```

### Startup Behavior

- `CAMOFOX_MCP_TRANSPORT=stdio`: `start_camoufox.sh` starts only `camofox-browser`; MCP server process is spawned per Python session.
- `CAMOFOX_MCP_TRANSPORT=http`: `start_camoufox.sh` starts both `camofox-browser` and `camofox-mcp` background process, with PID/log files in `data/logs`.