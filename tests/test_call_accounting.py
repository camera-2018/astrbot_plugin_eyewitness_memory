"""Cancellation must not leave committed model-call rows running."""

import asyncio
import threading

import pytest


async def test_cancel_after_call_insert_settles_row_without_starting_model(store, monkeypatch):
    inserted = threading.Event()
    release = threading.Event()
    original = store.record_call

    def delayed_record(*args):
        cid = original(*args)
        inserted.set()
        assert release.wait(3)
        return cid

    monkeypatch.setattr(store, "record_call", delayed_record)
    task = asyncio.create_task(store.call("record_call", "System One 相关性", 123))
    try:
        assert await asyncio.to_thread(inserted.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = store.db.execute("SELECT * FROM model_calls").fetchone()
    assert row["status"] == "interrupted"
    assert row["elapsed_ms"] is None
    assert row["input_tokens"] is None
    assert "登记" in row["error"]
    assert not store.lock.locked()


@pytest.mark.parametrize("status", ["success", "interrupted"])
async def test_cancel_while_finalization_waits_for_database_preserves_outcome(store, status):
    cid = await store.call("record_call", "候选选择", 123)
    await store.lock.acquire()
    task = asyncio.create_task(store.call("finish_call", cid, status, 42, (10, 2, None), ""))
    try:
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
    finally:
        store.lock.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = store.db.execute("SELECT * FROM model_calls WHERE id=?", (cid,)).fetchone()
    assert row["status"] == status
    assert row["elapsed_ms"] == 42
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 2
    assert not store.lock.locked()
