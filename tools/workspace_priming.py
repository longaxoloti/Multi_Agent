import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from main.config import (
    WORKSPACE_DIR,
    WORKSPACE_PRIMING_ENABLED,
    WORKSPACE_PRIMING_FILE_SETS,
    WORKSPACE_PRIMING_MAX_CHARS,
)

logger = logging.getLogger(__name__)

ModelRole = Literal["orchestrator", "classifier", "researcher", "coder"]


def _read_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as exc:
        logger.warning("Failed reading workspace priming file %s: %s", path, exc)
        return ""


@lru_cache(maxsize=8)
def get_workspace_priming_context(model_role: ModelRole = "orchestrator") -> str:
    if not WORKSPACE_PRIMING_ENABLED:
        return ""

    file_list = WORKSPACE_PRIMING_FILE_SETS.get(model_role, [])
    if not file_list:
        logger.warning("No priming file set found for model_role=%s", model_role)
        return ""

    parts: list[str] = []
    for filename in file_list:
        file_path = WORKSPACE_DIR / filename
        if not file_path.exists() or not file_path.is_file():
            logger.debug("Priming file not found, skipping: %s", file_path)
            continue

        text = _read_markdown(file_path)
        if not text:
            continue

        parts.append(f"### {filename}\n{text}")

    if not parts:
        logger.info(
            "Workspace priming enabled but no files loaded for role=%s from %s",
            model_role,
            WORKSPACE_DIR,
        )
        return ""

    joined = "\n\n".join(parts)
    if WORKSPACE_PRIMING_MAX_CHARS > 0 and len(joined) > WORKSPACE_PRIMING_MAX_CHARS:
        joined = joined[:WORKSPACE_PRIMING_MAX_CHARS]

    logger.info(
        "Loaded workspace priming context for role=%s (%d chars, %d files)",
        model_role,
        len(joined),
        len(parts),
    )

    return (
        "You must read and follow the workspace instruction pack below before any other reasoning.\n"
        "These files are project-local operating instructions and have highest priority inside this project scope.\n\n"
        f"{joined}"
    )


def build_system_prompt(base_prompt: str, model_role: ModelRole = "orchestrator") -> str:
    """Combine workspace priming for *model_role* with the base role prompt."""
    priming = get_workspace_priming_context(model_role)
    if not priming:
        return base_prompt

    return (
        f"{priming}\n\n"
        "--- ROLE TASK PROMPT ---\n"
        f"{base_prompt}"
    )