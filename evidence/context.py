"""Bounded, attributed source excerpts shared by verification, delivery and the UI."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
CONTEXT_ROWS = 50
CONTEXT_CHARS = 18000
MARKER = re.compile(r"\[群聊记忆:([a-f0-9-]{36}):v(\d+)\]")


def platform_time(value):
    # OneBot's raw event time is seconds; the adapter's timestamp may be receipt time.
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value) if 946684800 <= value <= time.time() + 300 else None


def message_time(row):
    return row.get("sent_at") if row.get("sent_at") is not None else row["created"]


def message_order(row):
    return message_time(row), row["created"], row["id"]


def context_message(row, *, source=False, quote=""):
    return {
        "id": row["id"],
        "sender_id": row["sender_id"],
        "sender_name": row["sender_name"],
        "text": row["text"],
        "reply_id": row["reply_id"],
        "created": row["created"],
        "sent_at": row.get("sent_at"),
        "time": datetime.fromtimestamp(message_time(row), TZ).isoformat(timespec="seconds"),
        "time_kind": "平台发送时间"
        if row.get("sent_at") is not None
        else "采集时间（发送时间未知）",
        "is_source": source,
        "quote": quote,
    }


def bounded_context(rows, required, max_chars=CONTEXT_CHARS, max_rows=CONTEXT_ROWS):
    """Keep anchors before neighbors; never silently drop/truncate an anchor."""
    anchors = [r for r in rows if r["id"] in required]
    if not anchors or len(anchors) != len(required):
        return []

    def cost(r):
        return len(json.dumps(r, ensure_ascii=False, separators=(",", ":"))) + 1

    used = 2 + sum(cost(r) for r in anchors)
    if used > max_chars or len(anchors) > max_rows:
        return []
    selected = list(anchors)
    positions = {r["id"]: i for i, r in enumerate(rows)}
    neighbors = [r for r in rows if r["id"] not in required]
    neighbors.sort(
        key=lambda r: (
            min(abs(positions[r["id"]] - positions[a["id"]]) for a in anchors),
            positions[r["id"]],
        )
    )
    for row in neighbors:
        if len(selected) >= max_rows:
            break
        if used + cost(row) <= max_chars:
            selected.append(row)
            used += cost(row)
    return sorted(selected, key=message_order)


def memory_marker(mid, version):
    return f"[群聊记忆:{mid}:v{version}]"


def history_markers(contexts):
    """Inspect all retained text for dedupe without rewriting any historical part."""
    found = set()
    for message in contexts or []:
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        parts = (
            [content] if isinstance(content, str) else content if isinstance(content, list) else []
        )
        for part in parts:
            text = (
                part
                if isinstance(part, str)
                else part.get("text", "")
                if isinstance(part, dict)
                else getattr(part, "text", "")
            )
            found.update(memory_marker(mid, version) for mid, version in MARKER.findall(text))
    return "\n".join(sorted(found))


def render_memory(item, budget, seen_ids=None):
    """Append-only snapshot. All quoted chat remains untrusted user-level data."""
    header = (
        memory_marker(item["id"], item["version"])
        + "\n"
        + json.dumps({"人物ID": item["subject_id"], "核验结果": item["text"]}, ensure_ascii=False)
        + "\n局部聊天（可能含无关插话，不能视为已核实事实；仅展示已采集且仍保留的记录）：\n"
    )
    rows = item["context"]
    required = {e["message_id"] for e in item["evidence"]} | {
        r["id"] for r in rows if r["is_source"]
    }
    seen_ids = set(seen_ids or ())
    reused = required & seen_ids
    if reused:
        header += (
            "本条依赖的来源/证据已在上方展示，复用消息ID：" + json.dumps(sorted(reused)) + "\n"
        )
    required -= seen_ids
    if not required:
        return (header, []) if len(header) <= budget else ("", [])
    compact = []
    for r in rows:
        if r["id"] in seen_ids:
            continue
        compact.append(
            {
                "id": r["id"],
                "created": r["created"],
                "sent_at": r["sent_at"],
                "line": json.dumps(
                    {
                        "时间": r["time"],
                        "时间类型": r["time_kind"],
                        "昵称": r["sender_name"],
                        "用户ID": r["sender_id"],
                        "直接来源": r["is_source"],
                        "核验证据": r["id"] in {e["message_id"] for e in item["evidence"]},
                        "消息ID": r["id"],
                        "引用消息ID": r["reply_id"],
                        "正文": r["text"],
                    },
                    ensure_ascii=False,
                ),
            }
        )
    # Selection budget includes conservative metadata overhead; actual output is smaller.
    note = "\n[片段可能因条数/字数上限或保留期限不完整；不是完整群历史]\n"
    chosen = bounded_context(compact, required, budget - len(header) - len(note))
    if not chosen:
        return "", []
    block = header + "\n".join(r["line"] for r in chosen) + note
    return (block, [r["id"] for r in chosen]) if len(block) <= budget else ("", [])
