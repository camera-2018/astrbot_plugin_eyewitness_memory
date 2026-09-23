import asyncio
import json
import threading
import time

import pytest
from pydantic import ValidationError

from eyewitness.models import Candidate, Settings, extraction_noise, low_signal
from tests.conftest import OTHER, SCOPE, seed


async def test_defaults_no_capture(tmp_path):
    from eyewitness.store import Store

    s = Store(tmp_path / "empty.db")
    await s.call("open")
    assert (await s.call("get_settings")).mode == "off"
    assert await s.call("capture", SCOPE, "1", "u", "name", "喜欢画画") is None
    await s.call("close")


@pytest.mark.parametrize(
    "old_mode,expected", [("shadow", "off"), ("active", "active"), ("off", "off")]
)
async def test_upgrade_preserves_data_and_migrates_retired_mode(store, old_mode, expected):
    mid, _ = await seed(store)
    cfg = (await store.call("get_settings")).model_dump()
    cfg["mode"] = old_mode
    # Simulate an older persisted configuration, not a current API write.
    with store.db:
        store.db.execute("UPDATE settings SET data=? WHERE id=1", (json.dumps(cfg),))
    await store.call("close")
    await store.call("open")
    loaded = (await store.call("get_settings")).model_dump()
    assert loaded == {**cfg, "mode": expected}
    assert (await store.call("detail", mid))["status"] == "active"
    persisted = json.loads(store.db.execute("SELECT data FROM settings WHERE id=1").fetchone()[0])
    assert persisted["mode"] == expected
    if expected == "off":
        assert await store.call("capture", SCOPE, "disabled", "u", "u", "我喜欢绘画") is None


def test_retired_mode_cannot_be_configured():
    with pytest.raises(ValidationError):
        Settings(mode="shadow")


async def test_capture_idempotent_and_secret_redaction(store):
    a = await store.call("capture", SCOPE, "1", "u", "a", "password=secret123 hello")
    b = await store.call("capture", SCOPE, "1", "u", "b", "different")
    assert a == b
    rows = await store.call("recent", SCOPE)
    assert len(rows) == 1 and "secret123" not in rows[0]["text"]


@pytest.mark.parametrize(
    "text", ["🚀🚀", "[Image]", "[CQ:image,file=foo]", "https://example.com", "哈哈", "我去"]
)
async def test_noise_remains_in_context_but_not_extraction_queue(store, text):
    mid = await store.call("capture", SCOPE, text, "u", "小明", text)
    row = (await store.call("source_rows", SCOPE, [mid]))[0]
    assert row["text"] == text
    assert row["processed"] == 2
    assert extraction_noise(text)


@pytest.mark.parametrize("text", ["国企", "香芋味", "画风也很舒适", "我报名了摄影比赛"])
async def test_short_meaningful_message_is_still_extractable(store, text):
    mid = await store.call("capture", SCOPE, text, "u", "小明", text)
    row = (await store.call("source_rows", SCOPE, [mid]))[0]
    assert row["processed"] == 0
    assert not extraction_noise(text)


async def test_existing_pending_noise_is_backfilled_without_losing_source(store):
    mid = await store.call("capture", SCOPE, "old-emoji", "u", "小明", "🚀🚀")
    with store.db:
        store.db.execute("UPDATE messages SET processed=0 WHERE id=?", (mid,))
    assert await store.call("skip_existing_noise") == 1
    assert (await store.call("source_rows", SCOPE, [mid]))[0]["processed"] == 2
    assert await store.call("skip_existing_noise") == 0


async def test_sources_isolated_and_search_chinese(store):
    a, _ = await seed(store)
    b, _ = await seed(store, OTHER)
    assert [m["id"] for m in await store.call("search", SCOPE, "绘画比赛")] == [a]
    assert await store.call("detail", b, SCOPE) is None
    assert await store.call("source_context", b, SCOPE) == []


async def test_invalid_and_cross_scope_evidence_rejected(store):
    _, source = await seed(store, OTHER)
    msg = await store.call("capture", SCOPE, "new", "u", "u", "我喜欢绘画")
    batch = await store.call("source_rows", SCOPE, [msg])
    candidate = Candidate(
        summary="u 喜欢绘画",
        kind="preference",
        subject_id="u",
        stance="self_report",
        evidence=[{"message_id": source, "quote": "我喜欢绘画"}],
    )
    assert await store.call("save_extraction", batch, [candidate]) == []


@pytest.mark.parametrize(
    "stance,subject,expected",
    [
        ("joke", "u", None),
        ("uncertain", "u", None),
        ("hearsay", "u", "pending"),
        ("self_report", "other", "pending"),
        ("self_report", "u", "active"),
    ],
)
async def test_stance_and_subject_guard(store, stance, subject, expected):
    msg = await store.call("capture", SCOPE, "a", "u", "name", "我喜欢绘画")
    batch = await store.call("source_rows", SCOPE, [msg])
    c = Candidate(
        summary="这个人喜欢绘画",
        kind="preference",
        subject_id=subject,
        stance=stance,
        evidence=[{"message_id": msg, "quote": "喜欢绘画"}],
    )
    ids = await store.call("save_extraction", batch, [c])
    if expected is None:
        assert not ids
    else:
        assert (await store.call("detail", ids[0]))["status"] == expected


async def test_versions_conflict_disable_delete(store):
    mid, source = await seed(store)
    m = await store.call("edit", mid, 1, "alice 在本群自述计划参加绘画比赛", "disabled")
    assert m["version"] == 2 and len(m["versions"]) == 2
    assert not await store.call("search", SCOPE, "绘画")
    with pytest.raises(ValueError):
        await store.call("edit", mid, 1, m["summary"], "active")
    await store.call("delete", mid)
    assert await store.call("detail", mid) is None
    assert len(await store.call("source_rows", SCOPE, [source])) == 1


async def test_erase_also_removes_versions_and_sources(store):
    mid, source = await seed(store)
    other, _ = await seed(store, OTHER)
    assert await store.call("erase_subject", SCOPE, "alice") == 1
    assert await store.call("detail", mid) is None
    assert not await store.call("source_rows", SCOPE, [source])
    assert await store.call("detail", other)


async def test_call_usage_persisted(store):
    await store.call("record_call")
    await store.call("record_call")
    assert (await store.call("stats"))["calls_today"] == 2


async def test_deleted_source_cannot_be_resurrected(store):
    mid = await store.call("capture", SCOPE, "x", "alice", "a", "我喜欢绘画")
    batch = await store.call("source_rows", SCOPE, [mid])
    await store.call("erase_subject", SCOPE, "alice")
    candidate = Candidate(
        summary="alice 喜欢绘画",
        kind="preference",
        subject_id="alice",
        stance="self_report",
        evidence=[{"message_id": mid, "quote": "喜欢绘画"}],
    )
    assert await store.call("save_extraction", batch, [candidate]) == []


async def test_retention_preserves_linked_evidence(store):
    _, source = await seed(store)
    old = await store.call(
        "capture", SCOPE, "old", "b", "b", "普通短句", None, time.time() - 40 * 86400
    )
    await store.call("maintain", Settings())
    assert not await store.call("source_rows", SCOPE, [old])
    assert await store.call("source_rows", SCOPE, [source])


def test_settings_and_low_signal():
    with pytest.raises(ValidationError):
        Settings(allowed_scopes=["123"])
    with pytest.raises(ValidationError):
        Settings(collection="repeat_memory")
    assert low_signal("哈哈")
    assert not low_signal("后来呢", True)
    assert not low_signal("他之前的项目呢")


async def test_cancelled_database_call_keeps_serialization_lock(store):
    entered = threading.Event()
    release = threading.Event()

    def slow():
        entered.set()
        release.wait(timeout=2)

    store.slow = slow
    task = asyncio.create_task(store.call("slow"))
    await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    other = asyncio.create_task(store.call("stats"))
    await asyncio.sleep(0.02)
    assert not other.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await other)["memories"] == 0


async def test_outbox_retries_and_stale_completion(store):
    mid, _ = await seed(store)
    await store.call("edit", mid, 1, "alice 自述计划参加绘画比赛，尚未确认", "active")
    await store.call("index_done", mid, 1)
    assert any(j["memory_id"] == mid for j in await store.call("index_jobs"))
    await store.call("index_failed", mid, 0)
    assert not await store.call("index_jobs")
    await store.call("reindex")
    await store.call("index_done", mid, 2)
    assert not await store.call("index_jobs")
