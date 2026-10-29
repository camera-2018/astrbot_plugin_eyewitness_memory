import asyncio
import json

from eyewitness.admin import AdminAPI
from eyewitness.engine import Engine
from tests.conftest import seed


async def enable(store, **values):
    cfg = await store.call("get_settings")
    cfg = cfg.model_copy(
        update={"maintenance_enabled": True, "maintenance_semantic_enabled": True, **values}
    )
    await store.call("save_settings", cfg)
    return cfg


async def test_initial_schedule_and_idle_do_not_spend_model_calls(store):
    await enable(store)
    calls = []

    async def generate(*args):
        calls.append(args)
        raise AssertionError("Scheduled for tomorrow, no model now")

    engine = Engine(store, generate)
    first = await engine.maintenance_once()
    second = await engine.maintenance_once()
    assert first["status"] == "scheduled" and second["status"] == "not_due"
    assert calls == []


async def test_force_while_disabled_and_api_cannot_spend_or_modify(store):
    async def generate(*args):
        raise AssertionError("disabled")

    engine = Engine(store, generate)
    assert (await engine.maintenance_once(force=True))["status"] == "disabled"
    _, code = await AdminAPI(store, engine).handle({"path": "maintenance/run", "method": "POST"})
    assert code == 400
    assert store.db.execute("SELECT count(*) FROM model_calls").fetchone()[0] == 0


async def test_semantic_call_and_memory_limits_and_next_due(store):
    await enable(store, maintenance_review_limit=2)
    for i in range(4):
        await seed(store, sender=f"u{i}", text=f"我喜欢第{i}种音乐", summary=f"喜欢第{i}种音乐")
    calls = []

    async def generate(provider, system, prompt):
        data = json.loads(prompt)["data"]
        calls.append(data)
        return json.dumps(
            {
                "decisions": [
                    {"id": mid, "action": "keep", "category": "useful", "reason": "稳定偏好"}
                    for mid in data["targets"]
                ]
            }
        )

    engine = Engine(store, generate)
    result = await engine.maintenance_once(force=True)
    assert result["status"] == "completed"
    assert len(calls) == 1 and len(calls[0]["targets"]) == 2
    assert result["semantic"] == {"reviewed": 2, "forgotten": 0, "uncertain": 0, "calls": 1}
    assert (await engine.maintenance_once())["status"] == "not_due"
    assert len(calls) == 1
    assert store.db.execute("SELECT count(*) FROM memory_semantic_checks").fetchone()[0] == 2


async def test_semantic_failure_has_no_tight_retry_and_is_not_cleared_by_idle(store):
    await enable(store)
    await seed(store)
    calls = []

    async def generate(*args):
        calls.append(1)
        raise TimeoutError

    engine = Engine(store, generate)
    assert (await engine.maintenance_once(force=True))["status"] == "failed"
    assert engine.last_error.startswith("定时整理暂不可用：")
    assert (await engine.maintenance_once())["status"] == "not_due"
    assert engine.last_error.startswith("定时整理暂不可用：")
    assert len(calls) == 1


async def test_manual_run_is_single_flight_and_cancel_accounts_model(store):
    await enable(store)
    await seed(store)
    started = asyncio.Event()

    async def generate(*args):
        started.set()
        await asyncio.Event().wait()

    engine = Engine(store, generate)
    assert engine.start_maintenance() is True
    await started.wait()
    assert engine.start_maintenance() is False
    assert (await engine.maintenance_once(force=True))["status"] == "busy"
    await engine.close()
    assert not engine.maintenance_lock.locked()
    assert store.db.execute("SELECT status FROM model_calls").fetchone()[0] == "interrupted"
