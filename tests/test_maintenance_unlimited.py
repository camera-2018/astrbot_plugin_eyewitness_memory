import asyncio
import json
import time

import pytest
from pydantic import ValidationError

from eyewitness.engine import Engine
from eyewitness.models import Settings
from eyewitness.semantic_maintenance import budget_status, candidate_snapshot, prepare, reserve
from tests.conftest import seed
from tests.test_maintenance_engine import enable


def keep_response(prompt):
    targets = json.loads(prompt)["data"]["targets"]
    return json.dumps(
        {
            "decisions": [
                {"id": mid, "action": "keep", "category": "useful", "reason": "有原文支持"}
                for mid in targets
            ]
        }
    )


def test_zero_is_valid_unlimited_setting_but_negative_is_not():
    assert Settings(maintenance_review_limit=0).maintenance_review_limit == 0
    with pytest.raises(ValidationError):
        Settings(maintenance_review_limit=-1)


async def test_unlimited_run_exceeds_old_cap_and_retains_usage_and_seven_day_cooldown(store):
    cfg = await enable(store, maintenance_review_limit=0)
    mids = []
    for i in range(41):
        mid, _ = await seed(store, sender=f"u{i}", text=f"喜欢音乐{i}", summary=f"喜欢音乐{i}")
        mids.append(mid)
    calls, in_flight = [], 0

    async def generate(provider, system, prompt):
        nonlocal in_flight
        in_flight += 1
        assert in_flight == 1
        targets = json.loads(prompt)["data"]["targets"]
        assert 1 <= len(targets) <= 8
        calls.append(targets)
        await asyncio.sleep(0)
        in_flight -= 1
        return keep_response(prompt)

    engine = Engine(store, generate)
    result = await engine.maintenance_once(force=True)
    assert result["semantic"] == {"reviewed": 41, "forgotten": 0, "uncertain": 0, "calls": 6}
    assert sorted(mid for batch in calls for mid in batch) == sorted(mids)
    assert budget_status(store, cfg) == {
        "targets_used": 41,
        "calls_used": 6,
        "unlimited": True,
        "targets_remaining": None,
        "calls_remaining": None,
    }
    again = await Engine(store, generate).maintenance_once(force=True)
    assert again["semantic"]["calls"] == 0 and len(calls) == 6
    with store.db:
        store.db.execute(
            "UPDATE memory_semantic_checks SET checked_at=?", (time.time() - 7 * 86400 - 1,)
        )
    assert len(candidate_snapshot(store, cfg)["targets"]) == 41


async def test_frozen_run_does_not_follow_new_records_or_changed_versions(store):
    await enable(store, maintenance_review_limit=0)
    original = []
    for i in range(10):
        mid, _ = await seed(store, sender=f"u{i}", text=f"喜欢音乐{i}", summary=f"喜欢音乐{i}")
        original.append(mid)
    calls, added = [], []

    async def generate(provider, system, prompt):
        calls.append(json.loads(prompt)["data"]["targets"])
        if len(calls) == 1:
            added.append(
                (await seed(store, sender="new", text="喜欢新音乐", summary="喜欢新音乐"))[0]
            )
            await store.call("edit", original[-1], 1, "用户修订后的音乐偏好", "active")
        return keep_response(prompt)

    result = await Engine(store, generate).maintenance_once(force=True)
    reviewed = [mid for batch in calls for mid in batch]
    assert result["semantic"]["reviewed"] == 9 and len(calls) == 2
    assert set(reviewed) == set(original[:-1])
    next_run = await store.call("semantic_maintenance_candidates")
    assert {r["id"] for r in next_run["targets"]} == {original[-1], added[0]}


async def test_more_than_1024_invalid_prefix_records_do_not_hide_valid_tail(store):
    cfg = await enable(store, maintenance_review_limit=0)
    good, _ = await seed(store)
    base = dict(store.db.execute("SELECT * FROM memories WHERE id=?", (good,)).fetchone())
    rows = []
    for i in range(1030):
        row = {
            **base,
            "id": f"invalid-{i:04d}",
            "fingerprint": f"invalid-{i:04d}",
            "created": base["created"] - 1,
        }
        rows.append(tuple(row[k] for k in base))
    with store.db:
        store.db.executemany(
            "INSERT INTO memories("
            + ",".join(base)
            + ") VALUES("
            + ",".join("?" for _ in base)
            + ")",
            rows,
        )
    run = candidate_snapshot(store, cfg)
    assert len(run["targets"]) == 1031
    prepared = prepare(store, cfg, run=run)
    assert prepared["targets"] == [good]
    assert len(prepared["consumed"]) == 1031
    assert prepare(store, cfg, excluded=prepared["consumed"], run=run) is None


async def test_oversized_record_is_skipped_without_dropping_valid_record_after_it(store):
    cfg = await enable(store, maintenance_review_limit=0)
    large, source = await seed(store)
    good, _ = await seed(store, sender="bob", text="喜欢蓝色键盘", summary="喜欢蓝色键盘")
    with store.db:
        store.db.execute(
            "UPDATE messages SET text=text||? WHERE id=?", ("很长的原文" * 5000, source)
        )
    run = candidate_snapshot(store, cfg)
    prepared = prepare(store, cfg, run=run)
    assert prepared["targets"] == [good]
    assert set(prepared["consumed"]) == {large, good}


async def test_empty_or_all_invalid_unlimited_run_finishes_without_calls(store):
    cfg = await enable(store, maintenance_review_limit=0)
    calls = []

    async def generate(*args):
        calls.append(1)
        raise AssertionError("nothing eligible")

    assert (await Engine(store, generate).maintenance_once(force=True))["semantic"]["calls"] == 0
    mid, source = await seed(store)
    with store.db:
        store.db.execute(
            "UPDATE messages SET text=text||? WHERE id=?", ("很长的原文" * 5000, source)
        )
    assert (await Engine(store, generate).maintenance_once(force=True))["semantic"]["calls"] == 0
    assert calls == [] and budget_status(store, cfg)["calls_used"] == 0


async def test_changed_configuration_cancels_frozen_cohort_before_more_calls(store):
    cfg = await enable(store, maintenance_review_limit=0)
    for i in range(10):
        await seed(store, sender=f"u{i}", text=f"喜欢音乐{i}", summary=f"喜欢音乐{i}")
    calls = []

    async def generate(provider, system, prompt):
        calls.append(1)
        await store.call(
            "save_settings", cfg.model_copy(update={"maintenance_semantic_enabled": False})
        )
        return keep_response(prompt)

    engine = Engine(store, generate)
    assert (await engine.maintenance_once(force=True))["status"] == "failed"
    assert calls == [1]
    assert store.db.execute("SELECT count(*) FROM memory_semantic_checks").fetchone()[0] == 0
    assert (await engine.maintenance_once())["status"] == "not_due"
    assert calls == [1]


async def test_unlimited_failure_ends_run_and_does_not_retry_on_automatic_tick(store):
    cfg = await enable(store, maintenance_review_limit=0)
    await seed(store)
    calls = []

    async def generate(*args):
        calls.append(1)
        raise TimeoutError("model timeout")

    engine = Engine(store, generate)
    assert (await engine.maintenance_once(force=True))["status"] == "failed"
    assert (await engine.maintenance_once())["status"] == "not_due"
    assert calls == [1] and budget_status(store, cfg)["calls_used"] == 1


async def test_unlimited_reservation_still_enforces_one_batch_and_config(store):
    cfg = await enable(store, maintenance_review_limit=0)
    await seed(store)
    snapshot = prepare(store, cfg)
    assert reserve(store, cfg, snapshot)
    assert reserve(store, cfg, {**snapshot, "targets": snapshot["targets"] * 9}) is False
    await store.call("save_settings", cfg.model_copy(update={"maintenance_review_limit": 32}))
    with pytest.raises(ValueError):
        reserve(store, cfg, snapshot)


async def test_automatic_maintenance_does_not_block_worker_and_close_cancels_both(
    store, monkeypatch
):
    cfg = await enable(store, maintenance_review_limit=0)
    await seed(store)
    model_started, worker_continued = asyncio.Event(), asyncio.Event()
    index_cycles, extraction_cycles = [], []

    async def generate(*args):
        model_started.set()
        await asyncio.Event().wait()

    engine = Engine(store, generate)
    assert (await engine.maintenance_once())["status"] == "scheduled"
    with store.db:
        store.db.execute("UPDATE maintenance_state SET next_due=?", (time.time() - 1,))

    async def extract_once(cfg):
        extraction_cycles.append(1)
        return False

    async def index_once(cfg):
        index_cycles.append(1)

    pauses = []

    async def pause(seconds):
        assert seconds == 30
        pauses.append(1)
        if len(pauses) == 1:
            await model_started.wait()
        else:
            worker_continued.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(engine, "extract_once", extract_once)
    monkeypatch.setattr(engine, "index_once", index_once)
    monkeypatch.setattr("eyewitness.engine.asyncio.sleep", pause)
    engine.worker_task = asyncio.create_task(engine.worker())
    try:
        await asyncio.wait_for(worker_continued.wait(), timeout=2)
        assert len(extraction_cycles) == 2 and len(index_cycles) == 2
        assert engine.maintenance_lock.locked()
        assert engine.start_maintenance() is False
        assert store.db.execute("SELECT count(*) FROM model_calls").fetchone()[0] == 1
    finally:
        await engine.close()
    assert engine.worker_task.done() and engine.maintenance_task.done()
    assert not engine.maintenance_lock.locked()
    assert store.db.execute("SELECT status FROM model_calls").fetchone()[0] == "interrupted"
    assert budget_status(store, cfg)["calls_used"] == 1
