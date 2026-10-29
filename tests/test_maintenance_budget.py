import json
import time

import pytest

from eyewitness.engine import Engine
from eyewitness.semantic_maintenance import budget_status, prepare, reserve
from tests.conftest import seed
from tests.test_maintenance_engine import enable


async def test_manual_runs_and_new_engine_share_rolling_daily_cap(store):
    cfg = await enable(store, maintenance_review_limit=2)
    for i in range(5):
        await seed(store, sender=f"u{i}", text=f"喜欢音乐{i}", summary=f"喜欢音乐{i}")
    calls = []

    async def generate(provider, system, prompt):
        data = json.loads(prompt)["data"]
        calls.append(data)
        return json.dumps({"decisions": []})

    await Engine(store, generate).maintenance_once(force=True)
    await Engine(store, generate).maintenance_once(force=True)
    assert len(calls) == 1 and len(calls[0]["targets"]) == 2
    assert budget_status(store, cfg)["targets_remaining"] == 0
    with store.db:
        store.db.execute("UPDATE memory_semantic_budget SET created=?", (time.time() - 86401,))
    await Engine(store, generate).maintenance_once(force=True)
    assert len(calls) == 2


async def test_failed_request_spends_budget_and_cannot_manual_retry(store):
    cfg = await enable(store, maintenance_review_limit=1)
    await seed(store)
    calls = []

    async def generate(*args):
        calls.append(1)
        raise TimeoutError("upstream")

    await Engine(store, generate).maintenance_once(force=True)
    await Engine(store, generate).maintenance_once(force=True)
    assert calls == [1]
    assert budget_status(store, cfg)["calls_remaining"] == 0


async def test_reservation_rechecks_shared_cap_and_config(store):
    cfg = await enable(store, maintenance_review_limit=1)
    await seed(store)
    a, b = prepare(store, cfg), prepare(store, cfg)
    assert reserve(store, cfg, a)
    assert reserve(store, cfg, b) is False
    assert prepare(store, cfg) is None
    await store.call(
        "save_settings", cfg.model_copy(update={"maintenance_semantic_enabled": False})
    )
    with pytest.raises(ValueError):
        reserve(store, cfg, a)


async def test_prepare_is_readonly_for_spending_and_limits_remaining_targets(store):
    cfg = await enable(store, maintenance_review_limit=10)
    for i in range(12):
        await seed(store, sender=f"u{i}", text=f"喜欢音乐{i}", summary=f"喜欢音乐{i}")
    first = prepare(store, cfg)
    assert budget_status(store, cfg)["targets_used"] == 0
    assert reserve(store, cfg, first)
    second = prepare(store, cfg, excluded=first["targets"])
    assert len(second["targets"]) == 2
    assert reserve(store, cfg, second)
    assert prepare(store, cfg) is None
