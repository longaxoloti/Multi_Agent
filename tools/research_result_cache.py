import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CACHE_DIR = Path("data/cache/google_search")


@dataclass
class CacheConfig:
    ttl_seconds: float = 48 * 3600  # 48h


class ResearchResultCache:
    """Disk cache for Google-derived URL candidates.

    We cache *outputs* of the risky step (Google SERP -> selected article URLs)
    to reduce repeated hits to google.com for similar queries.
    """

    def __init__(self, *, config: CacheConfig | None = None, cache_dir: Path = CACHE_DIR):
        self.config = config or CacheConfig()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str) -> str:
        normalized = (query or "").strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def _path(self, query: str) -> Path:
        return self.cache_dir / f"{self._key(query)}.json"

    def get(self, query: str) -> dict[str, Any] | None:
        path = self._path(query)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        ts = float(payload.get("ts", 0) or 0)
        if not ts:
            return None
        if (time.time() - ts) > float(self.config.ttl_seconds):
            return None

        items = payload.get("selected_sources")
        if not isinstance(items, list) or not items:
            return None
        return payload

    def set(self, query: str, *, selected_sources: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
        path = self._path(query)
        payload: dict[str, Any] = {
            "ts": time.time(),
            "query": (query or "").strip(),
            "selected_sources": selected_sources,
        }
        if meta:
            payload["meta"] = meta
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
