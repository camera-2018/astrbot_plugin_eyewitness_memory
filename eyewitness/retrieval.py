"""Bounded conversation hints and balanced lexical/semantic candidate ranking."""

from __future__ import annotations

import re

from .models import safe_text


def conversation_hint(messages, limit: int = 3600) -> list[dict[str, str]]:
    """Copy recent user/assistant text, never media, tools or earlier injected memories."""
    result = []
    remaining = limit
    for message in reversed(list(messages or [])[-6:]):

        def field(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        role = field(message, "role")
        if role not in ("user", "assistant"):
            continue
        content = field(message, "content")
        if isinstance(content, list):
            content = "\n".join(
                field(part, "text")
                for part in content
                if field(part, "type") == "text" and isinstance(field(part, "text"), str)
            )
        if not isinstance(content, str):
            continue
        # Memory is appended at the user tail. Do not use it to establish relevance again.
        content = re.split(r"\[历史记忆参考|\[群聊记忆:", content, maxsplit=1)[0]
        text = safe_text(content.strip(), min(1200, remaining))
        if text:
            result.append({"role": role, "text": text})
            remaining -= len(text)
        if remaining <= 0:
            break
    return list(reversed(result))


def merge_rankings(*rankings: list[dict]) -> list[dict]:
    """Reciprocal rank fusion: neither retrieval path consumes all slots first."""
    records, scores = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            mid = item["id"]
            if mid in seen:
                continue
            seen.add(mid)
            records[mid] = item
            scores[mid] = scores.get(mid, 0) + 1 / (60 + rank)
    return [records[mid] for mid in sorted(records, key=lambda mid: -scores[mid])]
