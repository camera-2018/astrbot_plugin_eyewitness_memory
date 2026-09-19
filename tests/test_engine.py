import asyncio
import json
import time

from eyewitness.engine import Engine
from tests.conftest import OTHER, SCOPE, seed


class Model:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    async def __call__(self, *args):
        self.calls += 1
        value = self.answers.pop(0)
        return json.dumps(value, ensure_ascii=False)


def answers(mid, source):
    return [
        {"decisions": [{"id": mid, "action": "needs_source", "reason": "核对计划的主体"}]},
        {
            "supported": True,
            "text": "alice 曾自述计划参加绘画比赛，当前状态未知。",
            "reason": "本人原文支持",
            "evidence": [{"message_id": source, "quote": "我计划下个月参加绘画比赛"}],
        },
    ]


async def test_complete_active_flow(store):
    mid, source = await seed(store)
    model = Model(answers(mid, source))
    engine = Engine(store, model)
    result = await engine.recall(SCOPE, "之前绘画比赛的计划是什么", "alice")
    assert result["mode"] == "active" and len(result["selected"]) == 1
    assert "当前状态未知" in result["injection"] and model.calls == 2
    assert len(await store.call("traces", SCOPE)) == 1
    assert "我计划下个月参加绘画比赛" in result["injection"]
    assert "采集时间（发送时间未知）" in result["injection"]
    assert "+08:00" in result["injection"]
    assert source in result["selected"][0]["injected_message_ids"]


async def test_raw_context_reaches_verifier_and_main_model(store):
    mid, source = await seed(store)
    neighbor = "我可以跟你一起准备，明天再聊。"
    await store.call("capture", SCOPE, "neighbor", "bob", "小明", neighbor, sent_at=time.time())
    responses = answers(mid, source)
    prompts = []

    async def model(provider, system, prompt):
        prompts.append(json.loads(prompt))
        return json.dumps(responses.pop(0))

    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    messages = prompts[1]["data"]["messages"]
    assert any(
        r["text"] == neighbor and r["sender_name"] == "小明" and r["time_kind"] == "平台发送时间"
        for r in messages
    )
    assert neighbor in result["injection"] and "小明" in result["injection"]
    assert len(result["injection"]) <= (await store.call("get_settings")).injection_chars


async def test_injection_budget_never_emits_summary_without_required_source(store):
    mid, source = await seed(store)
    cfg = await store.call("get_settings")
    cfg.injection_chars = 200
    await store.call("save_settings", cfg)
    result = await Engine(store, Model(answers(mid, source))).recall(SCOPE, "绘画比赛")
    assert not result["injection"] and not result["selected"]
    assert "字数预算" in result["reason"]


async def test_retained_marker_prevents_duplicate_injection(store):
    from eyewitness.context import memory_marker

    mid, _ = await seed(store)
    model = Model([])
    result = await Engine(store, model).recall(SCOPE, "绘画比赛", visible=memory_marker(mid, 1))
    assert not result["injection"] and model.calls == 0


async def test_new_version_can_append_even_if_old_source_is_visible(store):
    from eyewitness.context import memory_marker

    mid, source = await seed(store)
    await store.call("edit", mid, 1, "alice 曾计划绘画比赛，但目前进度未知", "active")
    visible = memory_marker(mid, 1) + "我计划下个月参加绘画比赛"
    result = await Engine(store, Model(answers(mid, source))).recall(
        SCOPE, "绘画比赛", visible=visible
    )
    assert memory_marker(mid, 2) in result["injection"]


async def test_context_subject_erased_during_verification_cannot_leak(store):
    mid, source = await seed(store)
    await store.call("capture", SCOPE, "private-neighbor", "bob", "B", "不应泄露的邻近消息")
    responses = answers(mid, source)

    async def model(*args):
        if len(responses) == 1:
            await store.call("erase_subject", SCOPE, "bob")
        return json.dumps(responses.pop(0))

    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    assert not result["injection"] and not result["selected"]
    assert not await store.call("traces", SCOPE)


async def test_worker_shutdown(store):
    engine = Engine(store, Model([]))
    entered = asyncio.Event()

    async def slow(*args):
        entered.set()
        await asyncio.Event().wait()

    engine.worker = slow
    await engine.start()
    await entered.wait()
    await engine.close()
    assert engine.worker_task.done()


async def test_disabled_recall_does_not_call_model(store):
    await seed(store)
    cfg = await store.call("get_settings")
    cfg.mode = "off"
    await store.call("save_settings", cfg)
    engine = Engine(store, Model([]))
    result = await engine.recall(SCOPE, "绘画比赛")
    assert result["mode"] == "off" and not result["injection"]
    assert engine.generate.calls == 0


async def test_cross_group_review_cannot_select_foreign_memory(store):
    mid, source = await seed(store, OTHER)
    await seed(store, SCOPE)
    model = Model(answers(mid, source))
    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    assert not result["selected"] and model.calls == 1


async def test_low_signal_and_scope_disabled_no_calls(store):
    model = Model([])
    engine = Engine(store, model)
    assert not (await engine.recall(SCOPE, "哈哈"))["injection"]
    assert not (await engine.recall("default:GroupMessage:999", "绘画比赛"))["injection"]
    assert model.calls == 0


async def test_reject_no_verification(store):
    mid, _ = await seed(store)
    model = Model([{"decisions": [{"id": mid, "action": "reject", "reason": "只是接梗"}]}])
    assert not (await Engine(store, model).recall(SCOPE, "绘画比赛"))["selected"]
    assert model.calls == 1


async def test_already_visible_source_is_not_reinjected(store):
    await seed(store)
    model = Model([])
    result = await Engine(store, model).recall(
        SCOPE, "绘画比赛", visible="我计划下个月参加绘画比赛"
    )
    assert not result["selected"] and model.calls == 0


async def test_fabricated_quotes_fail_closed(store):
    mid, source = await seed(store)
    payload = answers(mid, source)
    payload[1]["evidence"][0]["quote"] = "不存在的原文"
    assert not (await Engine(store, Model(payload)).recall(SCOPE, "绘画比赛"))["selected"]


async def test_timeout_does_not_escape(store):
    await seed(store)
    cfg = await store.call("get_settings")
    cfg.online_timeout = 0.5
    await store.call("save_settings", cfg)

    async def slow(*args):
        await asyncio.sleep(5)
        return "{}"

    result = await Engine(store, slow).recall(SCOPE, "绘画比赛")
    assert "时间预算" in result["reason"] and not result["injection"]


async def test_edit_during_verification_invalidates_result(store):
    mid, source = await seed(store)
    responses = answers(mid, source)

    async def model(*args):
        if len(responses) == 1:
            await store.call("edit", mid, 1, "alice 已经放弃绘画比赛计划", "disabled")
        return json.dumps(responses.pop(0))

    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    assert not result["selected"]


async def test_delete_during_verification_does_not_recreate_private_trace(store):
    mid, source = await seed(store)
    responses = answers(mid, source)

    async def model(*args):
        if len(responses) == 1:
            await store.call("delete", mid)
        return json.dumps(responses.pop(0))

    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    assert not result["selected"] and not result["candidates"]
    assert not await store.call("traces", SCOPE)


async def test_invalid_json_and_no_budget_degrade(store):
    await seed(store)
    model = Model([{"nonsense": True}])
    assert not (await Engine(store, model).recall(SCOPE, "绘画比赛"))["injection"]
    cfg = await store.call("get_settings")
    cfg.daily_calls = 0
    await store.call("save_settings", cfg)
    model = Model([])
    assert not (await Engine(store, model).recall(SCOPE, "绘画比赛"))["selected"]
    assert model.calls == 0


async def test_background_extraction_real_store_and_retry(store):
    text = "我计划下个月参加绘画比赛"
    source = await store.call(
        "capture", SCOPE, "background", "alice", "Alice", text, None, time.time() - 1000
    )
    cfg = await store.call("get_settings")
    # Failed parse must leave pending messages for a later retry.
    engine = Engine(store, Model([{"wrong": True}]))
    import pytest

    with pytest.raises(Exception):
        await engine.extract_once(cfg)
    assert len(await store.call("pending_batch", cfg)) == 1
    engine.generate = Model(
        [
            {
                "candidates": [
                    {
                        "summary": "alice 自述计划参加绘画比赛",
                        "kind": "goal",
                        "subject_id": "alice",
                        "stance": "self_report",
                        "evidence": [{"message_id": source, "quote": text}],
                    }
                ]
            }
        ]
    )
    await engine.extract_once(cfg)
    assert not await store.call("pending_batch", cfg)
    assert len(await store.call("search", SCOPE, "绘画比赛")) == 1
