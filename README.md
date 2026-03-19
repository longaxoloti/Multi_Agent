# Multi Agent

**A self-hosted, multi-model AI agent system built with LangGraph — designed to run entirely on local hardware via Ollama, with optional cloud LLM fallback.**

This project builds an autonomous agent that receives messages through Telegram, classifies your intent, routes tasks to specialized worker models, and synthesizes coherent responses — all while managing RAM on a single machine by loading one model at a time.

---

## Features

- **Multi-Model Orchestration** — Each task type is handled by the most suitable model:
  - Classifier — Lightweight intent detection
  - Orchestrator — Task decomposition & user interaction
  - Researcher — Web research & deep reasoning
  - Coder — Code generation & debugging

- **LangGraph Workflow** — A stateful graph pipeline:
  ```
  Request Router → Classify → Orchestrate → [Research | Coding | Reasoning | Briefing | Chat] → Synthesize → END
  ```

- **Smart Web Research** — Anti-detect browsing with Camoufox + article extraction with Crawl4AI, including LLM-powered source selection and reranking

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
MODEL_FAST          # Intent classification
MODEL_RESEARCH      # Research tasks
MODEL_CODE          # Coding tasks
MODEL_CHAT          # Direct chat
MODEL_ORCHESTRATOR
```

Set any of these to `gemini`, `openai`, `anthropic`, or `ollama` if you want to self-host.

---

## Health Check

```bash
python scripts/health_check.py
```