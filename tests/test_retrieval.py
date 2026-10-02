import copy
import json

from eyewitness.engine import Engine
from eyewitness.retrieval import conversation_hint, merge_rankings
from tests.conftest import OTHER, SCOPE, seed
from tests.test_engine import Model, answers


def test_context_is_bounded_and_excludes_injected_memory_and_non_chat_content():
    messages = [
        {"role": "system", "content": "secret system"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "alice 在聊什么？[历史记忆参考：untrusted]旧记忆"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "在聊绘画比赛"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "thinking", "text": "private reasoning"},
            ],
        },
    ]
    original = copy.deepcopy(messages)
    assert conversation_hint(messages) == [
        {"role": "user", "text": "alice 在聊什么？"},
        {"role": "assistant", "text": "在聊绘画比赛"},
    ]
    assert messages == original
    assert sum(len(r["text"]) for r in conversation_hint(messages, 7)) <= 7
    assert (
        sum(
            len(r["text"])
            for r in conversation_hint(
                [{"role": "user", "content": "长" * 3000} for _ in range(12)]
            )
        )
        <= 3600
    )


def test_rank_fusion_preserves_semantic_hits_and_deduplicates():
    lexical = [{"id": f"word-{i}"} for i in range(12)]
    semantic = [{"id": "semantic"}, lexical[3], {"id": "semantic"}]
    ranked = merge_rankings(lexical, semantic)
    assert ranked[0] == lexical[3]
    assert {"id": "semantic"} in ranked[:12]
    assert len(ranked) == 13


async def test_full_lexical_results_do_not_discard_semantic_match(store, monkeypatch):
    lexical = []
    for i in range(12):
        mid, _ = await seed(store, sender=f"user{i}", text=f"我喜欢设备{i}", summary=f"喜欢设备{i}")
        lexical.append(await store.call("detail", mid, SCOPE))
    target, source = await seed(store)
    foreign, _ = await seed(store, scope=OTHER)
    stale, _ = await seed(store, sender="stale", text="我喜欢数学", summary="喜欢数学")
    monkeypatch.setattr(store, "search", lambda scope, query: lexical)

    class Vector:
        def enabled(self, cfg):
            return True

        async def search(self, cfg, scope, query):
            return [
                {"id": foreign, "payload": {"version": 1}},
                {"id": stale, "payload": {"version": 0}},
                {"id": target, "payload": {"version": 1}},
            ]

    result = await Engine(store, Model(answers(target, source)), Vector()).recall(
        SCOPE, "画画的安排", "bob"
    )
    assert result["selected"][0]["id"] == target
    assert result["retrieval"]["lexical"] == 12
    assert result["retrieval"]["semantic"] == 1
    assert len(result["retrieval"]["reviewed_ids"]) == 12
    assert foreign not in result["retrieval"]["reviewed_ids"]
    assert stale not in result["retrieval"]["reviewed_ids"]


async def test_filtering_ineligible_candidates_happens_before_limit(store, monkeypatch):
    target, source = await seed(store)
    detail = await store.call("detail", target, SCOPE)
    # These have no stored sources, so none should consume a review slot.
    decoys = [{**detail, "id": f"missing-{i}"} for i in range(12)]
    monkeypatch.setattr(store, "search", lambda scope, query: decoys + [detail])
    result = await Engine(store, Model(answers(target, source))).recall(SCOPE, "绘画比赛")
    assert result["selected"][0]["id"] == target


async def test_followup_context_reaches_search_review_and_verifier(store, monkeypatch):
    target, source = await seed(store)
    responses = answers(target, source)
    prompts, searches = [], []
    original_search = store.search

    def search(scope, query):
        searches.append(query)
        return original_search(scope, query)

    monkeypatch.setattr(store, "search", search)

    async def model(provider, system, prompt):
        prompts.append(json.loads(prompt))
        return json.dumps(responses.pop(0), ensure_ascii=False)

    contexts = [{"role": "user", "content": "alice 的绘画比赛[群聊记忆:old:v1]过期描述"}]
    result = await Engine(store, model).recall(
        SCOPE, "他后来怎么样了", "bob", conversation=contexts
    )
    assert "alice 的绘画比赛" in searches[0]
    assert "过期描述" not in searches[0]
    assert result["injection"]
    for prompt in prompts:
        assert prompt["data"]["conversation"] == [{"role": "user", "text": "alice 的绘画比赛"}]


async def test_quoted_message_outside_recent_window_is_scope_bound(store):
    target, source = await seed(store)
    await store.call("capture", SCOPE, "quote", "alice", "Alice", "我提过绘画比赛")
    await store.call("capture", OTHER, "quote", "eve", "Eve", "不可见的其他群原文")
    for i in range(10):
        await store.call("capture", SCOPE, str(i), "bob", "Bob", f"插话{i}")
    responses = answers(target, source)
    prompts = []

    async def model(provider, system, prompt):
        prompts.append(json.loads(prompt))
        return json.dumps(responses.pop(0))

    result = await Engine(store, model).recall(SCOPE, "后来呢", "bob", reply_id="quote")
    assert result["injection"]
    for prompt in prompts:
        assert prompt["data"]["quoted_message"]["text"] == "我提过绘画比赛"
        assert "不可见的其他群原文" not in json.dumps(prompt, ensure_ascii=False)
    assert await store.call("referenced_message", SCOPE, "missing") is None


async def test_new_topic_does_not_expand_search_with_old_conversation(store, monkeypatch):
    queries = []

    def search(scope, query):
        queries.append(query)
        return []

    monkeypatch.setattr(store, "search", search)
    await Engine(store, Model([])).recall(
        SCOPE, "今天天气怎么样", conversation=[{"role": "user", "content": "绘画比赛"}]
    )
    assert queries == ["今天天气怎么样"]
