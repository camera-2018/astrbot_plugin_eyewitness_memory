import asyncio
import json
import threading
import time

import pytest

from eyewitness.engine import Engine
from eyewitness.models import QueryPlan
from tests.conftest import SCOPE, seed
from tests.test_planning import PlanningModel
from tests.test_recall_gate import setup
from tests.test_systemone import enable


@pytest.mark.parametrize(
    "form,expected",
    [
        ("standalone", ["GroundedReview"]),
        ("contextual", ["GroundedReview"]),
        ("uncertain", ["GroundedReview"]),
    ],
)
async def test_function_routing_removes_only_redundant_calls(store, form, expected):
    _, classify, calls = await setup(store, choice="needed", signal="person", form=form)
    await enable(store, recall_fast_path_enabled=True)
    mid, _ = await seed(store)
    model = PlanningModel(["alice 绘画比赛"])
    r = await Engine(store, model, systemone=classify).recall(SCOPE, "alice 绘画比赛")
    assert [x["schema"]["title"] for x in model.requests] == expected
    assert len(calls) == 1
    assert r["selected"][0]["id"] == mid and r["injection"]


async def test_low_confidence_standalone_still_expands_query(store):
    _, classify, _ = await setup(store, choice="needed", p=0.7, signal="person")
    await enable(store, recall_fast_path_enabled=True)
    await seed(store)
    model = PlanningModel(["alice 绘画比赛"])
    r = await Engine(store, model, systemone=classify).recall(
        SCOPE, "她的那个安排", conversation=[{"role": "user", "content": "alice要参加绘画比赛"}]
    )
    assert "alice要参加绘画比赛" in r["retrieval"]["queries"][1]
    assert model.requests[0]["schema"]["title"] == "GroundedReview"


async def test_wide_candidate_pool_uses_fastbrain_shortlist(store):
    _, classify, calls = await setup(store, choice="needed", signal="history")
    await enable(store, recall_fast_path_enabled=True)
    for i in range(5):
        await seed(store, sender=f"p{i}", text=f"我报名绘画比赛第{i}组", summary=f"绘画比赛第{i}组")
    model = PlanningModel()
    r = await Engine(store, model, systemone=classify).recall(SCOPE, "绘画比赛各组")
    stages = [x["schema"]["title"] for x in model.requests]
    assert all(s == "GroundedReview" for s in stages)
    assert len(calls) == 2 and r["injection"]
    assert r["selection"]["backend"] == "fastbrain" and r["selection"]["status"] == "success"


@pytest.mark.parametrize("kind", ["reject", "invented_quote"])
async def test_fast_path_cannot_bypass_grounding(store, kind):
    _, classify, _ = await setup(store, choice="needed", signal="person")
    await enable(store, recall_fast_path_enabled=True)
    await seed(store)
    model = PlanningModel()

    async def ground(provider, system, prompt):
        data = json.loads(await model(provider, system, prompt))
        for item in data["decisions"]:
            if kind == "reject":
                item.update(action="reject", reason="主体不符")
            else:
                item["evidence"][0]["quote"] = "模型自己编造的原文"
        return json.dumps(data)

    r = await Engine(store, ground, systemone=classify).recall(SCOPE, "alice 绘画比赛")
    assert not r["injection"] and not r["selected"]
    assert [x["schema"]["title"] for x in model.requests] == ["GroundedReview"]


async def test_gate_timeout_returns_while_sqlite_insert_drains_without_starting_model(
    store, monkeypatch
):
    cfg, classify, calls = await setup(store)
    inserted, release = threading.Event(), threading.Event()
    original = store.record_call

    def blocked(*args):
        cid = original(*args)
        inserted.set()
        assert release.wait(3)
        return cid

    monkeypatch.setattr(store, "record_call", blocked)
    engine = Engine(store, PlanningModel(), systemone=classify)
    result = {}
    started = time.monotonic()
    task = asyncio.create_task(
        engine.skip_recall(cfg, "你好", "", [], result, deadline=started + 5)
    )
    try:
        assert await asyncio.to_thread(inserted.wait, 1)
        assert not await asyncio.wait_for(task, 0.6)
        assert result["recall_gate"]["status"] == "fallback"
        assert not calls and store.lock.locked()
    finally:
        release.set()
        await engine.close()
    assert not store.lock.locked()
    assert store.db.execute("SELECT status FROM model_calls").fetchone()[0] == "interrupted"


async def test_known_model_result_returns_before_blocked_accounting_and_is_persisted(store):
    cfg = await store.call("get_settings")
    model = PlanningModel()

    async def generate(*args):
        result = await model(*args)
        await store.lock.acquire()
        return result

    engine = Engine(store, generate)
    try:
        r = await asyncio.wait_for(engine.ask(cfg, "test", {}, QueryPlan), 0.3)
        assert r.queries == []
        assert engine._pending
    finally:
        store.lock.release()
        await engine.close()
    row = store.db.execute("SELECT status FROM model_calls").fetchone()
    assert row[0] == "success"


async def test_outer_deadline_does_not_wait_for_db_or_inject_late_result(store, monkeypatch):
    engine = Engine(store, PlanningModel())
    monkeypatch.setattr(engine, "_RECALL_HARD_TIMEOUT", 0.05)
    entered, release = threading.Event(), threading.Event()
    original = store.get_settings

    def blocked():
        entered.set()
        assert release.wait(3)
        return original()

    monkeypatch.setattr(store, "get_settings", blocked)
    task = asyncio.create_task(engine.recall(SCOPE, "alice是谁"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        result = await asyncio.wait_for(task, 0.3)
        assert not result["injection"] and not result["selected"]
        assert store.lock.locked()
    finally:
        release.set()
        await engine.close()
    assert not engine._pending
    assert not store.db.execute("SELECT * FROM model_calls").fetchall()


async def test_cancel_between_registration_commit_and_result_delivery_settles_row(store):
    cfg = await store.call("get_settings")
    model = PlanningModel()
    engine = Engine(store, model)
    track = engine._track
    first = True

    async def run():
        parent = asyncio.current_task()

        def interrupt_registration(task):
            nonlocal first
            if first:
                first = False
                task.add_done_callback(lambda _: parent.cancel())
            return track(task)

        engine._track = interrupt_registration
        await engine.ask(cfg, "test", {}, QueryPlan)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(run())
    await engine.close()
    assert not model.requests
    row = store.db.execute("SELECT status,error FROM model_calls").fetchone()
    assert row["status"] == "interrupted" and "登记" in row["error"]
