"""Local extraction batching. Keep raw chat; select evidence-bearing episodes."""

from __future__ import annotations

import json

from .context import context_message
from .models import extraction_noise


def compact_message(row: dict) -> dict:
    message = context_message(row) if "time" not in row else row
    result = {
        k: message.get(k)
        for k in (
            "id",
            "sender_id",
            "sender_name",
            "text",
            "time",
            "time_kind",
            "reply_id",
            "is_source",
        )
    }
    # Retain transport attribution only where an explicit group alias was used.
    for key in ("origin_scope", "platform_message_id"):
        if key in message:
            result[key] = message[key]
    return result


def extraction_messages(batch: list[dict], context: list[dict], bot_ids: list[str]):
    """Select bounded text episodes; the model alone decides memory value."""
    scopes = {r["scope"] for r in batch}
    combined = {
        r["id"]: r
        for r in [*context, *batch]
        if r["scope"] in scopes and r["sender_id"] not in bot_ids
    }
    rows = sorted(combined.values(), key=lambda r: (r["created"], r["id"]))
    current = {r["id"] for r in batch}
    by_platform = {r["platform_id"]: r for r in rows}
    anchors = []
    seen = set()
    for index, row in enumerate(rows):
        text = row["text"].strip()
        key = (row["sender_id"], text, row.get("reply_id"))
        if row["id"] not in current or key in seen or extraction_noise(text, bool(row["reply_id"])):
            continue
        seen.add(key)
        anchors.append(index)
    if not anchors:
        return []
    selected = {}
    for index in anchors:
        for row in rows[max(0, index - 2) : index + 3]:
            if abs(row["created"] - rows[index]["created"]) <= 180:
                selected[row["id"]] = row
        row = rows[index]
        if reference := by_platform.get(row.get("reply_id")):
            selected[reference["id"]] = reference
    return [selected[r["id"]] for r in rows if r["id"] in selected]


def pack_extraction(batch, context, bot_ids, limit, max_chars=22000):
    """Fill a paid batch with useful episodes, not a fixed number of raw rows.

    Consume only a contiguous prefix. Rows beyond the payload budget remain
    pending, and skipped raw rows remain available for later source context.
    """
    consumed, selected = [], []
    for end in range(1, len(batch) + 1):
        prefix = batch[:end]
        useful = extraction_messages(prefix, context, bot_ids)
        ids = {r["id"] for r in prefix}
        if sum(r["id"] in ids for r in useful) > limit:
            break
        size = sum(len(json.dumps(compact_message(r), ensure_ascii=False)) + 2 for r in useful)
        if size > max_chars:
            break
        consumed, selected = prefix, useful
    return consumed, selected
