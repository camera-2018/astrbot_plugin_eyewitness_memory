import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from eyewitness.engine import BASE, Engine
from eyewitness.errors import AuxiliaryModelFailure
from eyewitness.extraction import extraction_messages, pack_extraction
from eyewitness.models import Extraction
from eyewitness.retrieval import compact_review, direct_operation, merge_rankings
from eyewitness.usage import ModelReply
from tests.conftest import OTHER, SCOPE, seed
from tests.test_engine import Model, answers


async def pending(store, key, text, sender="alice", scope=SCOPE, reply="", offset=0):
    return await store.call(
        "capture", scope, key, sender, sender, text, reply, time.time() - 1000 + offset
    )


async def test_only_generic_overlap_does_not_call_llm(store):
    await seed(store, text="我今天打算练习钢琴", summary="alice 今天计划练习钢琴")
    model = Model([])
    result = await Engine(store, model).recall(SCOPE, "今天怎么样")
    assert not result["selected"] and model.calls == 0
    assert result["retrieval"]["gate_skipped"]
    assert "本地筛选" in result["reason"]


@pytest.mark.parametrize("query", ["哭哭", "还真是", "，教我", "继续", "那怎么办"])
async def test_reactions_and_topicless_followups_use_no_embedding_or_llm(store, query):
    await seed(store)

    class Vector:
        def enabled(self, cfg):
            return True

        async def search(self, *args):
            pytest.fail("A reaction or topicless followup must not contact embeddings")

    model = Model([])
    result = await Engine(store, model, Vector()).recall(SCOPE, query)
    assert not result["injection"] and model.calls == 0


async def test_short_followup_searches_current_topic_and_can_recall(store):
    target, source = await seed(store)
    model = Model(answers(target, source))
    result = await Engine(store, model).recall(
        SCOPE,
        "，教我",
        conversation=[{"role": "assistant", "content": "我们聊的是绘画比赛的准备。"}],
    )
    assert result["injection"] and model.calls == 1
    assert result["retrieval"]["expanded_query"]


async def test_followup_does_not_use_unrelated_group_recent_as_its_topic(store):
    await seed(store)
    await store.call("capture", SCOPE, "interjection", "bob", "Bob", "我在准备绘画比赛")
    model = Model([])
    result = await Engine(store, model).recall(SCOPE, "教我")
    assert not result["injection"] and model.calls == 0


@pytest.mark.parametrize(
    "query",
    ["打开电视", "睦头，把空调开到16度", "看看电视开了吗", "帮我打开浏览器", "把电视调到少儿频道"],
)
async def test_immediate_operations_use_no_model_even_with_matching_old_memory(store, query):
    await seed(store, text="我喜欢大屏电视", summary="alice 喜欢大屏电视")
    model = Model([])
    result = await Engine(store, model).recall(SCOPE, query)
    assert model.calls == 0 and not result["injection"]
    assert result["retrieval"]["intent_skip"]


@pytest.mark.parametrize(
    "query",
    [
        "按我上次说的设置打开电视",
        "打开之前讨论的项目",
        "模仿小林说三句话",
        "快祝群友生日快乐",
        "那个项目进展怎么样",
        "我应该买什么显卡",
    ],
)
def test_person_background_and_history_dependent_commands_are_not_shortcut(query):
    assert not direct_operation(query)


async def test_weak_semantic_only_skips_but_synonym_above_floor_gets_one_review(store):
    target, source = await seed(store)

    class Vector:
        score = 0.3

        def enabled(self, cfg):
            return True

        async def search(self, *args):
            return [{"id": target, "score": self.score, "payload": {"version": 1}}]

    vector = Vector()
    model = Model(answers(target, source))
    engine = Engine(store, model, vector)
    assert not (await engine.recall(SCOPE, "画画的安排"))["injection"]
    assert model.calls == 0
    vector.score = 0.7
    result = await engine.recall(SCOPE, "画画的安排")
    assert result["injection"] and model.calls == 1
    assert result["retrieval"]["signals"][target]["semantic_score"] == 0.7


async def test_named_subject_retrieval_does_not_require_summary_name_or_questioner_match(store):
    target, source = await seed(store, summary="曾计划参加绘画比赛")
    # Original sender nickname is 测试用户; summary contains neither this name nor ID.
    result = await Engine(store, Model(answers(target, source))).recall(
        SCOPE, "评价一下测试用户", sender_id="bob"
    )
    assert result["selected"][0]["id"] == target
    assert result["retrieval"]["subject"] == 1


async def test_contextual_birthday_background_is_not_gated_as_non_history_query(store):
    target, _ = await seed(store, text="今天是我生日", summary="alice 表示十月三日是自己的生日")
    model = Model([{"decisions": [{"id": target, "action": "reject", "reason": "检查时间"}]}])
    result = await Engine(store, model).recall(SCOPE, "快祝群友生日快乐", "bob")
    assert model.calls == 1 and target in result["retrieval"]["reviewed_ids"]


async def test_top_three_share_one_grounded_call_and_two_selections_max(store):
    for i in range(8):
        await seed(
            store, sender=f"user{i}", text=f"我报名绘画比赛第{i}组", summary=f"绘画比赛第{i}组"
        )
    prompts = []

    async def generate(provider, system, prompt):
        data = json.loads(prompt)["data"]
        prompts.append(data)
        messages = {r["id"]: r for r in data["messages"]}
        return json.dumps(
            {
                "decisions": [
                    {
                        "id": m["id"],
                        "action": "accept",
                        "reason": "原文支持",
                        "text": m["summary"],
                        "evidence": [
                            {
                                "message_id": m["source_ids"][0],
                                "quote": messages[m["source_ids"][0]]["text"],
                            }
                        ],
                    }
                    for m in data["candidates"]
                ]
            }
        )

    result = await Engine(store, generate).recall(SCOPE, "绘画比赛报名情况")
    assert len(prompts) == 1 and len(prompts[0]["candidates"]) == 3
    assert len(result["selected"]) == 2
    assert len({m["id"] for m in prompts[0]["messages"]}) == len(prompts[0]["messages"])


async def test_neighbor_quote_alone_cannot_substitute_for_direct_source(store):
    target, source = await seed(store)
    neighbor = await store.call("capture", SCOPE, "neighbor", "bob", "bob", "我报名了游泳比赛")
    payload = answers(target, source)
    payload[0]["decisions"][0]["evidence"] = [{"message_id": neighbor, "quote": "我报名了游泳比赛"}]
    result = await Engine(store, Model(payload)).recall(SCOPE, "绘画比赛")
    assert not result["selected"] and "未覆盖直接来源" in result["candidates"][0]["verification"]


def test_fusion_preserves_both_retrieval_signals():
    record = merge_rankings(
        [{"id": "one", "lexical_score": -3}], [{"id": "one", "semantic_score": 0.7}]
    )[0]
    assert record["lexical_score"] == -3 and record["semantic_score"] == 0.7


async def test_reaction_and_unanswered_question_batch_uses_no_llm(store):
    ids = [
        await pending(store, str(i), text, offset=i)
        for i, text in enumerate(["你们在聊什么？", "要不要继续？", "谁知道啊", "太离谱了"])
    ]
    model = Model([])
    assert not await Engine(store, model).extract_once(await store.call("get_settings"))
    assert model.calls == 0
    assert all(r["processed"] == 1 for r in await store.call("source_rows", SCOPE, ids))


@pytest.mark.parametrize("reply", [True, False])
async def test_short_answer_keeps_previous_batch_question_and_attributes_source(store, reply):
    question = await pending(store, "question", "你最喜欢什么口味？", sender="bob")
    rows = await store.call("source_rows", SCOPE, [question])
    await store.call("save_extraction", rows, [])
    answer = await pending(store, "answer", "香芋味", reply="question" if reply else "", offset=1)
    prompts = []

    async def generate(provider, system, prompt):
        data = json.loads(prompt)["data"]
        prompts.append(data)
        return json.dumps(
            {
                "candidates": [
                    {
                        "summary": "alice 表示最喜欢香芋味",
                        "kind": "preference",
                        "subject_id": "alice",
                        "stance": "hearsay",
                        "evidence": [
                            {"message_id": question, "quote": "你最喜欢什么口味？"},
                            {"message_id": answer, "quote": "香芋味"},
                        ],
                    }
                ]
            }
        )

    assert await Engine(store, generate).extract_once(await store.call("get_settings"))
    assert {r["id"] for r in prompts[0]["messages"]} == {question, answer}
    memories = (await store.call("list_memories", SCOPE))["items"]
    # Attribution involving two authors must not be automatically self_report.
    assert len(memories) == 1 and memories[0]["status"] == "pending"
    assert len((await store.call("detail", memories[0]["id"]))["sources"]) == 2


async def test_context_from_other_group_cannot_create_memory(store):
    foreign = await pending(store, "question", "你最喜欢什么口味？", sender="bob", scope=OTHER)
    answer = await pending(store, "answer", "香芋味", reply="question", offset=1)
    batch = await store.call("source_rows", SCOPE, [answer])
    assert await store.call("extraction_context", batch) == []
    assert extraction_messages(batch, [], []) == []
    assert foreign != answer


async def test_busy_group_does_not_starve_second_scope(store):
    for i in range(7):
        await pending(store, str(i), "我计划参加绘画比赛", offset=i)
    await pending(store, "other", "我计划参加摄影比赛", scope=OTHER)
    cfg = await store.call("get_settings")
    cfg.batch_size = 5
    first = await store.call("pending_batch", cfg)
    await store.call("save_extraction", first, [])
    second = await store.call("pending_batch", cfg)
    assert first[0]["scope"] == SCOPE and second[0]["scope"] == OTHER


async def test_same_short_statement_by_two_people_is_not_deduplicated_across_subjects(store):
    a = await pending(store, "a", "我喜欢绘画", sender="alice")
    b = await pending(store, "b", "我喜欢绘画", sender="bob", offset=1)
    rows = await store.call("source_rows", SCOPE, [a, b])
    assert {r["id"] for r in extraction_messages(rows, [], [])} == {a, b}


async def test_usage_counts_provider_tokens_not_character_estimates(store):
    async def generate(*args):
        return ModelReply('{"candidates":[]}', 120, 7, 80)

    cfg = await store.call("get_settings")
    await Engine(store, generate).ask(cfg, "test", {}, Extraction)
    stat = (await store.call("stats"))["llm_usage_24h"][0]
    assert stat["input_tokens"] == 120 and stat["output_tokens"] == 7
    assert stat["usage_known"] == 1 and stat["succeeded"] == 1
    assert "test" not in json.dumps(stat) and BASE not in json.dumps(stat)


@pytest.mark.parametrize("raw", ["", "{invalid", '{"private":true}'])
async def test_failed_output_still_records_returned_usage(store, raw):
    async def generate(*args):
        return ModelReply(raw, 100, 0)

    with pytest.raises(AuxiliaryModelFailure):
        await Engine(store, generate).ask(await store.call("get_settings"), "test", {}, Extraction)
    stat = (await store.call("stats"))["llm_usage_24h"][0]
    assert stat["failed"] == 1 and stat["avg_input_tokens"] == 100
    assert "private" not in store.db.execute("SELECT error FROM model_calls").fetchone()[0]


async def test_timeout_records_interruption_and_unknown_usage(store):
    async def generate(*args):
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            Engine(store, generate).ask(await store.call("get_settings"), "test", {}, Extraction),
            0.02,
        )
    stat = (await store.call("stats"))["llm_usage_24h"][0]
    assert stat["failed"] == 1 and stat["running"] == 0
    assert stat["avg_input_tokens"] is None and stat["usage_known"] == 0


def test_sdk_usage_missing_is_not_zero():
    assert ModelReply.from_response(SimpleNamespace(completion_text="ok")).input_tokens is None
    response = SimpleNamespace(
        completion_text="ok", usage=SimpleNamespace(input=100, output=2, input_cached=80)
    )
    assert ModelReply.from_response(response) == ModelReply("ok", 100, 2, 80)


async def test_sparse_episodes_share_a_single_paid_batch_without_deleting_raw_chat(store):
    for i in range(12):
        await pending(
            store, str(i), "我报名了摄影课" if i in (3, 10) else "挺有意思", offset=i * 240
        )
    cfg = await store.call("get_settings")
    cfg.batch_size = 5
    model = Model([{"candidates": []}])
    assert await Engine(store, model).extract_once(cfg)
    assert model.calls == 1
    assert store.db.execute("SELECT count(*) FROM messages WHERE processed=1").fetchone()[0] == 12
    assert (await store.call("stats"))["messages"] == 12


async def test_packing_limit_keeps_unsubmitted_tail_pending(store):
    for i in range(12):
        await pending(store, str(i), f"我计划参加第{i}次比赛", offset=i)
    cfg = await store.call("get_settings")
    cfg.batch_size = 5
    model = Model([{"candidates": []}])
    assert await Engine(store, model).extract_once(cfg)
    assert store.db.execute("SELECT count(*) FROM messages WHERE processed=1").fetchone()[0] == 5
    assert store.db.execute("SELECT count(*) FROM messages WHERE processed=0").fetchone()[0] == 7


async def test_packing_does_not_use_later_rows_as_unpaid_context(store):
    for i in range(12):
        await pending(store, str(i), f"我计划参加第{i}次比赛", offset=i)
    batch = await store.call("pending_batch", await store.call("get_settings"))
    consumed, selected = pack_extraction(batch, [], [], 5)
    assert {r["id"] for r in selected}.issubset({r["id"] for r in consumed})


async def test_usage_survives_reload_and_running_is_marked_interrupted(store):
    await store.call("record_call", "联合核验", 30)
    await store.call("close")
    await store.call("open")
    stat = (await store.call("stats"))["llm_usage_24h"][0]
    assert stat["failed"] == 1 and stat["running"] == 0 and stat["avg_input_tokens"] is None


def test_request_aliases_are_lossless_without_mutating_or_renumbering_history():
    data = {
        "query": "sample",
        "messages": [{"id": "real-source", "text": "原文"}],
        "candidates": [
            {"id": "real-memory", "source_ids": ["real-source"], "context_ids": ["real-source"]}
        ],
    }
    payload, memories, sources = compact_review(data)
    assert data["messages"][0]["id"] == "real-source"
    assert payload["messages"][0]["id"] == "s1"
    assert payload["candidates"][0]["source_ids"] == ["s1"]
    assert memories == {"m1": "real-memory"} and sources == {"s1": "real-source"}


async def test_alias_response_is_resolved_before_injection(store):
    mid, source = await seed(store)

    async def generate(provider, system, prompt):
        data = json.loads(prompt)["data"]
        candidate = data["candidates"][0]
        return json.dumps(
            {
                "decisions": [
                    {
                        "id": candidate["id"],
                        "action": "accept",
                        "reason": "有原文",
                        "text": "alice 计划参加绘画比赛",
                        "evidence": [
                            {
                                "message_id": candidate["source_ids"][0],
                                "quote": "我计划下个月参加绘画比赛",
                            }
                        ],
                    }
                ]
            }
        )

    result = await Engine(store, generate).recall(SCOPE, "绘画比赛")
    assert result["selected"][0]["id"] == mid
    assert result["selected"][0]["evidence"][0]["message_id"] == source
    assert "[群聊记忆:m1" not in result["injection"]


async def test_privacy_epoch_is_checked_inside_serialized_extraction_save(store):
    mid = await pending(store, "pending", "我计划参加绘画比赛")
    batch = await store.call("source_rows", SCOPE, [mid])
    epoch = await store.call("epoch")
    await store.call("erase_subject", SCOPE, "bob")
    with pytest.raises(AuxiliaryModelFailure, match="隐私删除"):
        await store.call("save_extraction", batch, [], [], epoch)
    assert (await store.call("source_rows", SCOPE, [mid]))[0]["processed"] == 0
