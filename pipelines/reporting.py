from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from storage.trusted_db import TrustedClaim, UserKnowledgeRecord


def build_daily_report_text(claims: list[TrustedClaim], generated_at: datetime | None = None) -> str:
    generated_at = generated_at or datetime.utcnow()
    if not claims:
        return "No trusted updates found in the last 24 hours."

    grouped: dict[str, list[TrustedClaim]] = defaultdict(list)
    for claim in claims:
        grouped[claim.topic].append(claim)

    lines: list[str] = [
        "🌅 Daily Trusted Intelligence Report",
        f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    for topic, items in grouped.items():
        lines.append(f"## {topic}")
        for idx, item in enumerate(items[:5], 1):
            source = item.sources[0] if item.sources else "n/a"
            lines.append(
                f"{idx}. {item.claim[:240]}\n   - confidence: {item.confidence:.2f}\n   - source: {source}"
            )
        lines.append("")

    return "\n".join(lines)


def build_daily_knowledge_report_text(
    records: list[UserKnowledgeRecord],
    generated_at: datetime | None = None,
) -> str:
    generated_at = generated_at or datetime.utcnow()
    if not records:
        return "No user-requested knowledge updates found in this interval."

    grouped: dict[str, list[UserKnowledgeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category or "note"].append(record)

    lines: list[str] = [
        "🌅 Daily Knowledge Report",
        f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    for category, items in grouped.items():
        lines.append(f"## {category}")
        for idx, item in enumerate(items[:10], 1):
            snippet = (item.content or "").strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            title = (item.title or "").strip()
            tags = ", ".join(item.tags[:5]) if item.tags else "n/a"
            lines.append(
                f"{idx}. {title or snippet}\n"
                f"   - id: {item.id}\n"
                f"   - chat_id: {item.chat_id}\n"
                f"   - tags: {tags}\n"
                f"   - created_at: {item.created_at.strftime('%Y-%m-%d %H:%M UTC')}"
            )
        lines.append("")

    return "\n".join(lines)
