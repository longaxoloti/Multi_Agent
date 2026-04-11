import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATE_PATH = Path("data/state/google_guard.json")


@dataclass
class GoogleGuardConfig:
    min_interval_seconds: float = 15.0
    cooldown_on_challenge_seconds: float = 3600.0


class GoogleGuard:
    """A tiny, persistent circuit-breaker + rate limiter for google.com requests.

    Goal: prevent bursty behavior and stop retry storms after a challenge.
    This does NOT "bypass" bot detection; it reduces self-inflicted risk.
    """

    def __init__(self, *, config: GoogleGuardConfig | None = None, state_path: Path = STATE_PATH):
        self.config = config or GoogleGuardConfig()
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _now(self) -> float:
        return time.time()

    def can_hit_google(self) -> tuple[bool, str]:
        state = self._load()
        now = self._now()

        cooldown_until = float(state.get("cooldown_until", 0) or 0)
        if now < cooldown_until:
            remaining = int(cooldown_until - now)
            return False, f"google cooldown active ({remaining}s remaining)"

        last_hit = float(state.get("last_hit", 0) or 0)
        delta = now - last_hit
        if last_hit and delta < self.config.min_interval_seconds:
            wait = self.config.min_interval_seconds - delta
            return False, f"google rate-limited (wait {wait:.1f}s)"

        return True, "ok"

    def mark_google_hit(self) -> None:
        state = self._load()
        state["last_hit"] = self._now()
        self._save(state)

    def mark_challenge(self, *, reason: str = "challenge") -> None:
        state = self._load()
        now = self._now()
        state["cooldown_until"] = now + float(self.config.cooldown_on_challenge_seconds)
        state["last_challenge_reason"] = reason
        state["last_challenge_at"] = now
        self._save(state)
