import asyncio
import json
import sqlite3
import time

import pytest

from eyewitness.admin import AdminAPI
from eyewitness.engine import Engine
from eyewitness.errors import AuxiliaryModelFailure
from tests.conftest import OTHER, SCOPE


async def pending(store, scope=SCOPE, key="pending", text="我计划下个月参加绘画比赛"):
    return await store.call("capture", scope, key, "alice", "Alice", text, None, time.time() - 1000)


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("empty", "模型返回空内容"),
        ("json", "JSON 无法解析"),
        ("schema", "JSON 包含多余字段"),
        ("timeout", "请求超时"),
        ("rate_limit", "上游 HTTP 429"),
        ("sdk_empty", "EmptyModelOutputError"),
        ("save", "SQLite OperationalError"),
    ],
)
async def test_failure_is_durable_bounded_and_does_not_retry(store, monkeypatch, failure, reason):
    source = await pending(store)
    calls = 0

    class UpstreamError(Exception):
        status_code = 429

    class EmptyModelOutputError(Exception):
        pass

    async def generate(*args):
        nonlocal calls
        calls += 1
        if failure == "rate_limit":
            raise UpstreamError("sk-private-upstream-credential")
        if failure == "sdk_empty":
            raise EmptyModelOutputError("private-response-body")
        if failure == "timeout":
            await asyncio.Event().wait()
        return {"empty": "", "json": "private-invalid-json", "schema": '{"wrong":true}'}.get(
            failure, '{"candidates":[]}'
        )

    def save_error(*args):
        raise sqlite3.OperationalError("private-database-details")

    if failure == "save":
        monkeypatch.setattr(store, "save_extraction", save_error)
    engine = Engine(store, generate)
    cfg = await store.call("get_settings")
    cfg.extraction_timeout = 0.01  # Exercise the real deadline, without a slow test.
    with pytest.raises(AuxiliaryModelFailure, match=reason):
        await engine.extract_once(cfg)
    await store.call("close")
    await store.call("open")
    for _ in range(4):
        assert not await Engine(store, generate).extract_once(cfg)
    assert calls == 1
    row = (await store.call("source_rows", SCOPE, [source]))[0]
    assert row["processed"] == 3 and row["text"] == "我计划下个月参加绘画比赛"
    assert reason in row["extraction_error"]
    assert row["extraction_attempted_at"] > 0
    overview, status = await AdminAPI(store, Engine(store, generate)).handle({"path": "overview"})
    assert status == 200 and overview["extraction_failed_messages"] == 1
    assert reason in overview["last_extraction_failure"]["reason"]
    assert "private-" not in json.dumps(overview)
    assert not await store.call("search", SCOPE, "绘画比赛")


async def test_valid_empty_candidates_complete_without_failure(store):
    source = await pending(store)

    async def generate(*args):
        return '{"candidates":[]}'

    engine = Engine(store, generate)
    cfg = await store.call("get_settings")
    assert await engine.extract_once(cfg)
    assert (await store.call("source_rows", SCOPE, [source]))[0]["processed"] == 1
    assert (await store.call("stats"))["extraction_failed_messages"] == 0
    assert not await engine.extract_once(cfg)


async def test_only_submitted_bounded_batch_fails(store):
    for i in range(15):
        await pending(store, key=str(i), text=f"第{i}次的绘画计划" + "计划" * 990)

    submitted = []

    async def generate(provider, system, prompt):
        submitted.extend(m["id"] for m in json.loads(prompt)["data"]["messages"])
        return ""

    cfg = await store.call("get_settings")
    with pytest.raises(AuxiliaryModelFailure):
        await Engine(store, generate).extract_once(cfg)
    failed = [r[0] for r in store.db.execute("SELECT id FROM messages WHERE processed=3")]
    assert set(failed) == set(submitted)
    rest = await store.call("pending_batch", cfg)
    assert 0 < len(rest) < 15 and len(rest) + len(failed) == 15
    assert not set(failed).intersection(r["id"] for r in rest)


async def test_claim_is_atomic_and_does_not_steal_another_attempt(store):
    first = await pending(store, key="first")
    second = await pending(store, key="second")
    batch = await store.call("source_rows", SCOPE, [first, second])
    assert await store.call("claim_extraction", batch[:1])
    assert not await store.call("claim_extraction", batch)
    assert [
        r["id"] for r in await store.call("pending_batch", await store.call("get_settings"))
    ] == [second]
    await store.call("fail_extraction", batch[:1], "测试失败")
    assert await store.call("claim_extraction", batch[1:])


async def test_deleted_batch_is_not_claimed_or_recreated(store):
    source = await pending(store)
    batch = await store.call("source_rows", SCOPE, [source])
    await store.call("erase_subject", SCOPE, "alice")
    assert not await store.call("claim_extraction", batch)
    await store.call("fail_extraction", batch, "测试失败")
    assert not await store.call("source_rows", SCOPE, [source])


async def test_claim_cannot_cross_scopes(store):
    first = await pending(store)
    second = await pending(store, scope=OTHER)
    batch = await store.call("source_rows", SCOPE, [first])
    batch += await store.call("source_rows", OTHER, [second])
    with pytest.raises(ValueError, match="跨群"):
        await store.call("claim_extraction", batch)


async def test_cancelled_attempt_is_not_retried_after_reload(store):
    source = await pending(store)
    entered = asyncio.Event()
    calls = 0

    async def generate(*args):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()

    engine = Engine(store, generate)
    cfg = await store.call("get_settings")
    task = asyncio.create_task(engine.extract_once(cfg))
    await asyncio.wait_for(entered.wait(), 1)
    assert not await Engine(store, generate).extract_once(cfg)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await store.call("close")
    await store.call("open")
    assert not await Engine(store, generate).extract_once(cfg)
    row = (await store.call("source_rows", SCOPE, [source]))[0]
    assert row["processed"] == 3 and "中断" in row["extraction_error"]
    assert calls == 1


async def test_unfinished_attempt_after_crash_is_failed_not_requeued(store):
    source = await pending(store)
    batch = await store.call("source_rows", SCOPE, [source])
    assert await store.call("claim_extraction", batch)
    # Simulate a crash after committing the claim, without a completion/failure write.
    await store.call("close")
    await store.call("open")
    row = (await store.call("source_rows", SCOPE, [source]))[0]
    assert row["processed"] == 3 and "上次提取被中断" in row["extraction_error"]
    assert not await store.call("pending_batch", await store.call("get_settings"))


async def test_settings_disabled_in_flight_does_not_leave_running_rows(store):
    source = await pending(store)
    cfg = await store.call("get_settings")

    async def generate(*args):
        changed = cfg.model_copy(update={"mode": "off"})
        await store.call("save_settings", changed)
        return '{"candidates":[]}'

    with pytest.raises(AuxiliaryModelFailure, match="已关闭"):
        await Engine(store, generate).extract_once(cfg)
    row = (await store.call("source_rows", SCOPE, [source]))[0]
    assert row["processed"] == 3


async def test_idle_worker_does_not_erase_extraction_error(store):
    async def generate(*args):
        pytest.fail("An idle worker must not call the provider")

    engine = Engine(store, generate)
    engine.last_error = "提取失败：模型返回空内容"
    await engine.start()
    try:
        async with asyncio.timeout(1):
            while not engine.last_cycle:
                await asyncio.sleep(0.01)
        assert engine.last_error == "提取失败：模型返回空内容"
    finally:
        await engine.close()


async def test_manual_retry_is_once_scoped_and_protects_against_stale_clicks(store):
    first = await pending(store)
    foreign = await pending(store, scope=OTHER)
    cfg = await store.call("get_settings")
    calls = 0

    async def generate(*args):
        nonlocal calls
        calls += 1
        return ""

    engine = Engine(store, generate)
    for _ in range(2):
        with pytest.raises(AuxiliaryModelFailure):
            await engine.extract_once(cfg)
    assert calls == 2
    api = AdminAPI(store, engine)
    data, status = await api.handle({"path": "failed-batches?scope=" + SCOPE})
    assert status == 200 and data["total"] == 1
    batch = data["items"][0]
    operation = {
        "path": f"failed-batches/{batch['id']}/retry",
        "method": "POST",
        "body": {"confirm": "RETRY", "attempted_at": batch["attempted_at"]},
    }
    assert (await api.handle({**operation, "body": {}}))[1] == 400
    result, status = await api.handle(operation)
    assert status == 200 and result["queued"] == 1 and calls == 2
    assert (await store.call("source_rows", OTHER, [foreign]))[0]["processed"] == 3
    assert (await api.handle(operation))[1] == 409
    with pytest.raises(AuxiliaryModelFailure):
        await engine.extract_once(cfg)
    assert calls == 3
    assert (await api.handle(operation))[1] == 409
    assert not await engine.extract_once(cfg)
    assert (await store.call("source_rows", SCOPE, [first]))[0]["processed"] == 3


async def test_manual_retry_cannot_recreate_privacy_deleted_data_or_use_disabled_scope(store):
    source = await pending(store)
    batch = await store.call("source_rows", SCOPE, [source])
    await store.call("claim_extraction", batch)
    await store.call("fail_extraction", batch, "请求超时")
    failed = (await store.call("failed_batches"))["items"][0]
    api = AdminAPI(store, Engine(store, lambda *_: None))
    op = {
        "path": f"failed-batches/{failed['id']}/retry",
        "method": "POST",
        "body": {"confirm": "RETRY", "attempted_at": failed["attempted_at"]},
    }
    cfg = await store.call("get_settings")
    await store.call("save_settings", cfg.model_copy(update={"allowed_scopes": [OTHER]}))
    assert (await api.handle(op))[1] == 409
    await store.call("erase_subject", SCOPE, "alice")
    assert (await api.handle(op))[1] == 409
    assert not await store.call("source_rows", SCOPE, [source])


async def test_background_error_uses_injected_plugin_reporter(store):
    await pending(store)
    reported = []

    async def generate(*args):
        return ""

    engine = Engine(store, generate, report=reported.append)
    await engine.start()
    try:
        async with asyncio.timeout(2):
            while not engine.last_cycle:
                await asyncio.sleep(0.01)
        assert len(reported) == 1 and "提取失败" in reported[0]
    finally:
        await engine.close()
