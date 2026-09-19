import json
import sqlite3
import time

import pytest

from eyewitness.context import (
    CONTEXT_CHARS,
    history_markers,
    memory_marker,
    platform_time,
    render_memory,
)
from eyewitness.models import Candidate, Settings
from eyewitness.store import Store
from tests.conftest import OTHER, SCOPE, seed


async def save_memory(store, ids):
    rows = await store.call("source_rows", SCOPE, ids)
    return (
        await store.call(
            "save_extraction",
            rows,
            [
                Candidate(
                    subject_id="alice",
                    summary="alice 自述准备参加绘画比赛",
                    kind="goal",
                    stance="self_report",
                    evidence=[{"message_id": r["id"], "quote": r["text"][:1000]} for r in rows],
                )
            ],
        )
    )[0]


async def populate(store, count=45, same_time=False):
    base = time.time() - 100000
    ids = []
    for i in range(count):
        ids.append(
            await store.call(
                "capture",
                SCOPE,
                f"msg-{i}",
                "alice",
                "小林",
                f"第{i:02d}条：我准备参加绘画比赛",
                created=base + count - i,
                sent_at=base if same_time else base + i * 300,
            )
        )
    return ids


async def test_context_ten_before_after_platform_time_and_scope(store):
    ids = await populate(store)
    mid = await save_memory(store, [ids[22]])
    await seed(store, OTHER, text="其他群的秘密不能出现")
    rows = await store.call("source_context", mid, SCOPE)
    assert [r["id"] for r in rows] == ids[12:33]
    assert [r["id"] for r in rows if r["is_source"]] == [ids[22]]
    assert all(r["time"].endswith("+08:00") and r["time_kind"] == "平台发送时间" for r in rows)
    assert await store.call("source_context", mid, OTHER) == []


async def test_context_overlap_deduplicated_and_ties_deterministic(store):
    ids = await populate(store, same_time=True)
    ordered = list(reversed(ids))
    mid = await save_memory(store, [ordered[20], ordered[22]])
    rows = await store.call("source_context", mid, SCOPE)
    assert [r["id"] for r in rows] == ordered[10:33]
    assert len({r["id"] for r in rows}) == len(rows)


async def test_context_cap_retains_all_source_anchors(store):
    ids = await populate(store, 100)
    anchors = [ids[i] for i in [10, 30, 50, 70, 90]]
    mid = await save_memory(store, anchors)
    rows = await store.call("source_context", mid, SCOPE)
    assert len(rows) <= 50
    assert set(anchors) <= {r["id"] for r in rows}
    assert len(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))) <= CONTEXT_CHARS


async def test_context_long_messages_preserve_anchor(store):
    ids = []
    base = time.time() - 100
    for i in range(25):
        ids.append(
            await store.call("capture", SCOPE, str(i), "alice", "A", "绘画" * 950, created=base + i)
        )
    mid = await save_memory(store, [ids[12]])
    rows = await store.call("source_context", mid, SCOPE)
    assert ids[12] in {r["id"] for r in rows}
    assert len(rows) < 21
    assert len(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))) <= CONTEXT_CHARS


async def test_context_keeps_remote_reply_parent(store):
    base = time.time() - 10000
    parent = await store.call(
        "capture", SCOPE, "parent", "bob", "B", "你要去参加什么比赛？", created=base
    )
    await populate(store, 45)
    msg = await store.call(
        "capture", SCOPE, "answer", "alice", "A", "我准备参加绘画比赛", "parent", base + 9999
    )
    mid = await save_memory(store, [msg])
    rows = await store.call("source_context", mid, SCOPE)
    assert parent in {r["id"] for r in rows}
    assert all(r["time_kind"].startswith("采集时间") for r in rows if r["sent_at"] is None)


async def test_old_schema_migration_keeps_unknown_send_time(tmp_path):
    path = tmp_path / "legacy.db"
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE messages (
      id TEXT PRIMARY KEY, scope TEXT, platform_id TEXT, sender_id TEXT, sender_name TEXT,
      text TEXT, reply_id TEXT, created REAL, processed INTEGER DEFAULT 0,
      UNIQUE(scope,platform_id)); PRAGMA user_version=1;""")
    db.execute(
        "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
        ("old", SCOPE, "1", "alice", "A", "旧消息", "", 1700000000, 0),
    )
    db.commit()
    db.close()
    store = Store(path)
    await store.call("open")
    row = (await store.call("source_rows", SCOPE, ["old"]))[0]
    assert row["created"] == 1700000000 and row["sent_at"] is None
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == 2
    await store.call("close")
    await store.call("open")
    assert (await store.call("source_rows", SCOPE, ["old"]))[0] == row
    await store.call("close")


@pytest.mark.parametrize(
    "value", [None, True, "123", float("nan"), float("inf"), -1, 1700000000000]
)
def test_invalid_platform_timestamp(value):
    assert platform_time(value) is None


def test_marker_scan_all_history_without_mutation():
    marker = memory_marker("12345678-1234-1234-1234-123456789abc", 1)
    history = [{"role": "user", "content": [{"type": "text", "text": marker}]}] + [
        {"role": "user", "content": "later"}
    ] * 20
    original = json.dumps(history)
    assert marker in history_markers(history)
    assert json.dumps(history) == original
    assert Settings().injection_chars == 8000


async def test_overlapping_memories_do_not_repeat_chat_text(store):
    mid, source = await seed(store)
    rows = await store.call("source_context", mid, SCOPE)
    item = {
        "id": mid,
        "version": 1,
        "subject_id": "alice",
        "text": "历史事实",
        "context": rows,
        "evidence": [{"message_id": source}],
    }
    first, ids = render_memory(item, 8000)
    second, more_ids = render_memory({**item, "version": 2}, 8000 - len(first), set(ids))
    assert "我计划下个月参加绘画比赛" in first
    assert "我计划下个月参加绘画比赛" not in second
    assert "复用消息ID" in second and not more_ids
