import json

import pytest

from eyewitness.engine import Engine
from eyewitness.models import extraction_noise, low_signal
from tests.conftest import SCOPE, seed
from tests.test_engine import Model, answers, planning_passthrough


@pytest.mark.parametrize(
    "query",
    [
        "有道理",
        "不是",
        "哈哈",
        "我好饿",
        "必须发语音",
        "降智了",
        "打开电视",
        "教我",
        "继续",
        "啤酒烧烤是什么",
        "嗯",
        "666",
        "Tell me more",
        "はい",
    ],
)
async def test_utterance_shape_cannot_override_a_semantic_match_or_model_decision(store, query):
    mid, source = await seed(store, text=query, summary="alice 的一段需要核对上下文的原话")

    class Vector:
        calls = 0

        def enabled(self, cfg):
            return True

        async def search(self, *args):
            self.calls += 1
            return [{"id": mid, "score": 0.9, "payload": {"version": 1}}]

    payload = answers(mid, source)
    payload[0]["decisions"][0].update(
        text="alice 曾在原文中作出这一回应，具体含义由上下文决定。",
        evidence=[{"message_id": source, "quote": query}],
    )
    vector, model = Vector(), Model(payload)
    result = await Engine(store, model, vector).recall(SCOPE, query, "bob")
    assert vector.calls == model.calls == 1
    assert result["selected"][0]["id"] == mid
    assert not result["retrieval"].get("intent_skip")
    assert not low_signal(query) and not extraction_noise(query)


async def test_weak_candidate_score_still_saves_model_calls_without_classifying_query(store):
    mid, _ = await seed(store)

    class Vector:
        def enabled(self, cfg):
            return True

        async def search(self, *args):
            return [{"id": mid, "score": 0.3, "payload": {"version": 1}}]

    model = Model([])
    result = await Engine(store, model, Vector()).recall(SCOPE, "天气如何")
    assert not result["injection"] and model.calls == 0
    assert result["retrieval"]["gate_skipped"]


async def test_model_can_reject_a_casual_response_after_reading_its_context(store):
    mid, _ = await seed(store)
    prompts = []

    @planning_passthrough
    async def generate(provider, system, prompt):
        data = json.loads(prompt)["data"]
        prompts.append(data)
        return json.dumps(
            {"decisions": [{"id": mid, "action": "reject", "reason": "没有新的背景信息需要补充"}]}
        )

    result = await Engine(store, generate).recall(
        SCOPE,
        "有道理",
        "bob",
        conversation=[{"role": "assistant", "content": "我们刚才讨论了绘画比赛"}],
    )
    assert len(prompts) == 1 and not result["injection"]
    assert prompts[0]["query"] == "有道理"
    assert prompts[0]["conversation"][0]["text"] == "我们刚才讨论了绘画比赛"
    assert not result["retrieval"].get("intent_skip")


async def test_quote_scoped_to_current_group_still_receives_review(store):
    mid, source = await seed(store)
    await store.call("capture", SCOPE, "quote", "alice", "Alice", "我计划下个月参加绘画比赛")
    model = Model(answers(mid, source))
    result = await Engine(store, model).recall(SCOPE, "比赛是什么", "bob", reply_id="quote")
    assert result["injection"] and model.calls == 1


async def test_quoted_presentation_keeps_historical_topic_recall(store):
    mid, source = await seed(store)
    await store.call("capture", SCOPE, "quote", "alice", "Alice", "我的绘画比赛计划")
    model = Model(answers(mid, source))
    result = await Engine(store, model).recall(SCOPE, "发语音", "bob", reply_id="quote")
    assert result["injection"] and model.calls == 1


async def test_named_person_definition_can_still_recall_own_history(store):
    mid, source = await seed(store)
    result = await Engine(store, Model(answers(mid, source))).recall(
        SCOPE, "测试用户的比赛是什么", "bob"
    )
    assert result["selected"][0]["id"] == mid


async def test_recommendation_keeps_personal_preference_recall(store):
    mid, source = await seed(store)
    result = await Engine(store, Model(answers(mid, source))).recall(
        SCOPE, "给我推荐绘画比赛的准备方法", "alice"
    )
    assert result["selected"][0]["id"] == mid
