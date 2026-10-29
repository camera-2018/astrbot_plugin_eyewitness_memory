import asyncio
import sqlite3

import pytest

from eyewitness.engine import Engine
from eyewitness.errors import AuxiliaryModelFailure
from tests.conftest import seed


async def no_model(*args):
    pytest.fail("Health checks must not make model requests")


def fail_maintenance(store, monkeypatch, failed_cycles):
    original = store.call
    cycle = 0

    async def call(name, *args, **kwargs):
        nonlocal cycle
        if name == "maintain":
            cycle += 1
            if cycle in failed_cycles:
                error = sqlite3.OperationalError("private database path")
                error.sqlite_errorname = "SQLITE_BUSY"
                raise error
        return await original(name, *args, **kwargs)

    monkeypatch.setattr(store, "call", call)


async def run_cycles(engine, monkeypatch, count, after_cycle=None):
    snapshots = []

    async def pause(seconds):
        assert seconds == 30
        snapshots.append((engine.last_error, engine.last_cycle))
        if after_cycle:
            await after_cycle(len(snapshots))
        if len(snapshots) == count:
            raise asyncio.CancelledError

    monkeypatch.setattr("eyewitness.engine.asyncio.sleep", pause)
    with pytest.raises(asyncio.CancelledError):
        await engine.worker()
    return snapshots


async def test_busy_warning_clears_after_complete_healthy_idle_cycle(store, monkeypatch):
    fail_maintenance(store, monkeypatch, {1})
    reports = []
    engine = Engine(store, no_model, report=reports.append)

    first, recovered = await run_cycles(engine, monkeypatch, 2)

    assert first == ("后台任务异常：SQLite SQLITE_BUSY", 0)
    assert recovered[0] == "" and recovered[1] > 0
    assert reports == [first[0]]


async def test_busy_warning_is_retained_while_maintenance_still_fails(store, monkeypatch):
    fail_maintenance(store, monkeypatch, {1, 2})
    engine = Engine(store, no_model)

    snapshots = await run_cycles(engine, monkeypatch, 2)

    assert snapshots == [("后台任务异常：SQLite SQLITE_BUSY", 0)] * 2


@pytest.mark.parametrize(
    "unrecovered",
    ["提取失败：模型返回空内容", "向量索引暂不可用：上游 HTTP 503"],
)
async def test_busy_recovery_restores_older_unrecovered_stage_warning(
    store, monkeypatch, unrecovered
):
    fail_maintenance(store, monkeypatch, {1})
    engine = Engine(store, no_model)
    engine.last_error = unrecovered

    first, recovered = await run_cycles(engine, monkeypatch, 2)

    assert first[0] == "后台任务异常：SQLite SQLITE_BUSY"
    assert recovered[0] == unrecovered


async def test_busy_recovery_does_not_hide_new_extraction_failure(store, monkeypatch):
    fail_maintenance(store, monkeypatch, {1})
    engine = Engine(store, no_model)

    async def broken_extraction(cfg):
        raise AuxiliaryModelFailure("模型返回空内容")

    monkeypatch.setattr(engine, "extract_once", broken_extraction)
    _, recovered = await run_cycles(engine, monkeypatch, 2)

    assert recovered[0] == "提取失败：模型返回空内容"
    assert recovered[1] > 0


class FlakyVector:
    def __init__(self):
        self.calls = 0

    def enabled(self, cfg):
        return True

    async def upsert(self, cfg, memory):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("private endpoint")

    async def delete(self, cfg, mid):
        pytest.fail("The fixture contains a live memory")


async def test_busy_recovery_does_not_hide_new_index_failure(store, monkeypatch):
    await seed(store)
    fail_maintenance(store, monkeypatch, {1})
    engine = Engine(store, no_model, FlakyVector())

    _, recovered = await run_cycles(engine, monkeypatch, 2)

    assert recovered[0] == "向量索引暂不可用：连接失败"


async def test_index_recovery_keeps_unrecovered_extraction_failure(store, monkeypatch):
    await seed(store)
    engine = Engine(store, no_model, FlakyVector())
    extraction_calls = 0

    async def extraction(cfg):
        nonlocal extraction_calls
        extraction_calls += 1
        if extraction_calls == 1:
            raise AuxiliaryModelFailure("模型返回空内容")
        return False  # No later successful extraction has resolved the failure.

    async def retry_index(cycle):
        if cycle == 1:
            await store.call("reindex")

    monkeypatch.setattr(engine, "extract_once", extraction)
    first, recovered = await run_cycles(engine, monkeypatch, 2, retry_index)

    assert first[0] == "向量索引暂不可用：连接失败"
    assert recovered[0] == "提取失败：模型返回空内容"
    assert not await store.call("index_jobs")


async def test_extraction_recovery_keeps_index_failure_during_retry_backoff(store, monkeypatch):
    await seed(store)
    engine = Engine(store, no_model, FlakyVector())
    extraction_calls = 0

    async def extraction(cfg):
        nonlocal extraction_calls
        extraction_calls += 1
        if extraction_calls == 1:
            raise AuxiliaryModelFailure("模型返回空内容")
        return True

    monkeypatch.setattr(engine, "extract_once", extraction)
    first, recovered = await run_cycles(engine, monkeypatch, 2)

    assert first[0] == recovered[0] == "向量索引暂不可用：连接失败"
    assert engine.vector.calls == 1


async def test_other_index_job_success_does_not_clear_job_still_in_backoff(store):
    await seed(store)
    engine = Engine(store, no_model, FlakyVector())
    cfg = await store.call("get_settings")
    await engine.index_once(cfg)

    await seed(store, sender="bob", text="我喜欢烤面包", summary="bob 自述喜欢烤面包")
    await engine.index_once(cfg)

    assert engine.vector.calls == 2
    assert engine.last_error == "向量索引暂不可用：连接失败"
    await store.call("reindex")
    await engine.index_once(cfg)
    assert engine.last_error == ""


async def test_cancellation_before_healthy_cycle_completes_keeps_warning(store, monkeypatch):
    engine = Engine(store, no_model)
    engine.last_error = "后台任务异常：SQLite SQLITE_BUSY"

    async def cancelled_index(cfg):
        raise asyncio.CancelledError

    monkeypatch.setattr(engine, "index_once", cancelled_index)
    with pytest.raises(asyncio.CancelledError):
        await engine.worker()

    assert engine.last_cycle == 0
    assert engine.last_error == "后台任务异常：SQLite SQLITE_BUSY"
