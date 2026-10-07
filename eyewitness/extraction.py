"""Local extraction batching. Keep raw chat; select evidence-bearing episodes."""

from __future__ import annotations

import json
import re

from .context import context_message
from .models import extraction_noise

FACT_CUE = re.compile(
    r"喜欢|讨厌|偏好|过敏|习惯|计划|打算|准备|报名|约好|约定|决定|生日|毕业|入职|"
    r"考试|比赛|工作|职业|专业|服务器|显卡|设备|买了|换了|装了|住在|搬到|去了|"
    r"每天|每周|下个月|下周|下次|曾经|以前|我在|我是|我用|我有|我的|我想|我会|我不|"
    r"最爱|爱喝|爱吃|口味|上班|入学|结婚|养了|考研|考公|备考|在学|学了|我|不想|正在|用的|用了"
)
QUESTION = re.compile(r"[?？]|什么|哪个|多少|怎么|为何|吗[。！!~～]*$")
COMMAND = re.compile(r"^(?:@\S+\s*)?(?:帮我|给我|你去|看看|查一下|搜索|画一|生成|复读)")


def compact_message(row: dict) -> dict:
    message = context_message(row) if "time" not in row else row
    return {
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


def extraction_messages(batch: list[dict], context: list[dict], bot_ids: list[str]):
    """Retain facts, answers and their neighbors; never merge different senders."""
    combined = {r["id"]: r for r in [*context, *batch] if r["sender_id"] not in bot_ids}
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
        reference = by_platform.get(row.get("reply_id"))
        preceding = rows[max(0, index - 2) : index]
        # An explicit reply can refer outside the preceding window. Non-reply
        # short answers need a nearby question, not an arbitrary old question.
        answer = (
            len(text) >= 2
            and not QUESTION.search(text)
            and (
                bool(reference)
                or any(
                    QUESTION.search(r["text"])
                    and (FACT_CUE.search(r["text"]) or len(r["text"]) >= 12)
                    and row["created"] - r["created"] <= 180
                    for r in preceding
                )
            )
        )
        statement = (
            not QUESTION.search(text)
            and not COMMAND.search(text)
            # Short statements can contain real preferences or progress. Local
            # filtering is deliberately permissive; the model decides value.
            and (bool(FACT_CUE.search(text)) or len(text) >= 8)
        )
        if answer or statement:
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
