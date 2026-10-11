"""Bounded conversation hints and balanced lexical/semantic candidate ranking."""

from __future__ import annotations

import re
from copy import deepcopy

from .models import safe_text, terms


def compact_review(data):
    """Request-local source aliases avoid repeatedly tokenizing UUIDs.

    Real IDs remain in storage and injected snapshots; aliases never escape this
    one verification request. Unknown returned IDs still fail source validation.
    """
    payload = deepcopy(data)
    source_ids = {f"s{i}": row["id"] for i, row in enumerate(payload["messages"], 1)}
    memory_ids = {f"m{i}": row["id"] for i, row in enumerate(payload["candidates"], 1)}
    sources = {value: key for key, value in source_ids.items()}
    memories = {value: key for key, value in memory_ids.items()}
    for message in payload["messages"]:
        message["id"] = sources[message["id"]]
    for memory in payload["candidates"]:
        memory["id"] = memories[memory["id"]]
        for field in ("source_ids", "context_ids"):
            memory[field] = [sources[mid] for mid in memory[field]]
    return payload, memory_ids, source_ids


def rank_candidates(
    candidates, query: str, subjects: list[str], semantic_min: float, *, context_query: str = ""
):
    """Cheap eligibility/ranking only. Scores never authorize injection."""
    wanted = set(terms(query))
    contextual = set(terms(context_query))
    ranked, skipped = [], []
    for rank, memory in enumerate(candidates):
        overlap = wanted.intersection(terms(memory["summary"]))
        context_overlap = contextual.intersection(terms(memory["summary"]))
        subject_match = memory["subject_id"] in subjects
        score = memory.get("semantic_score")
        semantic_match = isinstance(score, (float, int)) and score >= semantic_min
        signals = {
            "topic_terms": sorted(overlap)[:12],
            "context_terms": sorted(context_overlap)[:12],
            "subject_match": subject_match,
            "semantic_score": score,
            "lexical_score": memory.get("lexical_score"),
        }
        if not (overlap or context_overlap or subject_match or semantic_match):
            skipped.append(
                {"id": memory["id"], "reason": "无文本、人物关联且语义匹配不足", **signals}
            )
            continue
        priority = (
            1.2 * subject_match
            + min(len(overlap), 6) * 0.18
            + min(len(context_overlap), 4) * 0.05
            + (score if isinstance(score, (int, float)) else 0)
            + 0.15 / (rank + 1)
        )
        ranked.append((priority, {**memory, "signals": signals}))
    ranked.sort(key=lambda item: -item[0])
    return [memory for _, memory in ranked], skipped


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
            previous = records.get(mid, {})
            merged = {**previous, **item}
            semantic = [r.get("semantic_score") for r in (previous, item)]
            scores_present = [s for s in semantic if isinstance(s, (int, float))]
            if scores_present:
                merged["semantic_score"] = max(scores_present)
            records[mid] = merged
            scores[mid] = scores.get(mid, 0) + 1 / (60 + rank)
    return [records[mid] for mid in sorted(records, key=lambda mid: -scores[mid])]
