import asyncio
import json

import pytest

from eyewitness.engine import Engine
from eyewitness.models import CandidateShortlist, QueryPlan
from eyewitness.planning import search_queries, shortlist_payload
from eyewitness.retrieval import merge_rankings
from tests.conftest import OTHER, SCOPE, seed


class PlanningModel:
    """Exercise every real stage; no production understanding is mocked out."""

    def __init__(self, queries=None, choose=None):
        self.queries = queries or []
        self.choose = choose
        self.requests = []

    async def __call__(self, provider, system, prompt):
        body = json.loads(prompt)
        self.requests.append(body)
        data = body["data"]
        if body["schema"]["title"] == "QueryPlan":
            return json.dumps({"queries": self.queries, "reason": "依据当前问题及会话"})
        if body["schema"]["title"] == "CandidateShortlist":
            chosen = self.choose(data) if self.choose else data["candidates"][: data["limit"]]
            return json.dumps(
                {"selected": [{"id": m["id"], "reason": "值得查原文"} for m in chosen]}
            )
        messages = {r["id"]: r for r in data["messages"]}
        return json.dumps(
            {
                "decisions": [
                    {
                        "id": m["id"],
                        "action": "accept",
                        "reason": "直接原文支持",
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


def test_query_expansion_never_replaces_original_or_accepts_unbounded_strings():
    assert search_queries("原问题", ["改写", "原问题", "改写", "", "另一个", "额外"], 2) == [
        "原问题",
        "改写",
        "另一个",
    ]
    assert search_queries("原问题", ["改写"], 0) == ["原问题"]
    with pytest.raises(ValueError):
        QueryPlan(queries=["a" * 401], reason="长输出")
    with pytest.raises(ValueError):
        CandidateShortlist(selected=[{"id": "m1", "reason": "x"}] * 13)


async def test_model_understanding_adds_search_branch_and_preserves_current_question(store):
    target, _ = await seed(store)
    model = PlanningModel(["alice 绘画比赛"])
    result = await Engine(store, model).recall(
        SCOPE,
        "再聊聊那个安排",
        "bob",
        conversation=[{"role": "assistant", "content": "刚才在讨论 alice 提过的活动。"}],
    )
    assert result["retrieval"]["queries"] == ["再聊聊那个安排", "alice 绘画比赛"]
    assert result["selected"][0]["id"] == target
    assert [r["schema"]["title"] for r in model.requests] == [
        "QueryPlan",
        "CandidateShortlist",
        "GroundedReview",
    ]
    assert all(r["data"]["query"] == "再聊聊那个安排" for r in model.requests)
    assert "candidates" not in model.requests[0]["data"]
    assert model.requests[-1]["data"]["understanding"]["queries"] == ["alice 绘画比赛"]


async def test_understanding_hints_cannot_be_used_as_grounding_evidence(store):
    await seed(store)
    model = PlanningModel(["原文里没有的细节"])

    async def invent_evidence(provider, system, prompt):
        body = json.loads(prompt)
        if body["schema"]["title"] != "GroundedReview":
            return await model(provider, system, prompt)
        assert body["data"]["understanding"]["queries"] == ["原文里没有的细节"]
        return json.dumps(
            {
                "decisions": [
                    {
                        "id": "m1",
                        "action": "accept",
                        "reason": "错误地把模型的线索当来源",
                        "text": "原文里没有的细节",
                        "evidence": [{"message_id": "understanding", "quote": "原文里没有的细节"}],
                    }
                ]
            }
        )

    result = await Engine(store, invent_evidence).recall(SCOPE, "绘画比赛")
    assert not result["selected"] and not result["injection"]
    assert result["candidates"][0]["verification"] == "逐字证据缺失、越界或未覆盖直接来源"


async def test_incorrect_rewrite_cannot_erase_original_retrieval(store):
    target, _ = await seed(store)
    model = PlanningModel(["不相关的天气预报"])
    result = await Engine(store, model).recall(SCOPE, "绘画比赛的计划")
    assert result["retrieval"]["queries"][0] == "绘画比赛的计划"
    assert result["selected"][0]["id"] == target


async def test_wide_summary_pool_can_select_candidate_beyond_old_top_three(store):
    ids = []
    for i in range(10):
        mid, _ = await seed(
            store,
            sender=f"person{i}",
            text=f"我报名绘画比赛第{i}组",
            summary=f"绘画比赛第{i}组的安排",
        )
        ids.append(mid)
    model = PlanningModel(choose=lambda data: data["candidates"][-1:])
    result = await Engine(store, model).recall(SCOPE, "绘画比赛各组安排")
    assert len(result["selection"]["pool_ids"]) == 10
    selected = result["selected"][0]["id"]
    assert result["selection"]["pool_ids"].index(selected) >= 3
    assert selected in ids
    assert "messages" not in model.requests[1]["data"]


async def test_more_than_three_candidates_are_grounded_without_cross_batch_alias_leaks(store):
    ids = []
    for i in range(8):
        mid, _ = await seed(
            store, sender=f"person{i}", text=f"我报名绘画比赛第{i}组", summary=f"绘画比赛第{i}组"
        )
        ids.append(mid)
    model = PlanningModel()
    result = await Engine(store, model).recall(SCOPE, "绘画比赛各组")
    reviews = [r for r in model.requests if r["schema"]["title"] == "GroundedReview"]
    assert len(reviews) == 2 and all(len(r["data"]["candidates"]) == 3 for r in reviews)
    assert len(result["retrieval"]["reviewed_ids"]) == 6
    assert len({c["id"] for c in result["candidates"]}) == 6
    assert len(result["selected"]) == 2
    assert all(c["id"] in ids for c in result["candidates"])
    assert all(
        not e["message_id"].startswith("s") for m in result["selected"] for e in m["evidence"]
    )
    calls = store.db.execute(
        "SELECT stage,status,count(*) FROM model_calls GROUP BY stage,status"
    ).fetchall()
    assert {(r[0], r[1]): r[2] for r in calls} == {
        ("问题理解", "success"): 1,
        ("候选选择", "success"): 1,
        ("联合核验", "success"): 2,
    }


async def test_unknown_or_foreign_shortlist_id_never_escapes_current_pool(store):
    target, _ = await seed(store)
    foreign, _ = await seed(store, scope=OTHER)
    model = PlanningModel(choose=lambda data: [{"id": foreign}])
    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    assert result["selection"]["status"] == "fallback"
    assert result["selected"][0]["id"] == target
    assert foreign not in result["selection"]["pool_ids"]


async def test_successful_empty_shortlist_is_not_replaced_by_forced_injection(store):
    await seed(store)
    model = PlanningModel(choose=lambda data: [])
    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    assert not result["injection"]
    assert result["selection"]["status"] == "success"
    assert [r["schema"]["title"] for r in model.requests] == ["QueryPlan", "CandidateShortlist"]


async def test_understanding_timeout_falls_back_without_retry_or_losing_original(store):
    target, _ = await seed(store)
    model = PlanningModel()
    plan_attempts = 0

    async def slow_plan(provider, system, prompt):
        nonlocal plan_attempts
        if json.loads(prompt)["schema"]["title"] == "QueryPlan":
            plan_attempts += 1
            await asyncio.sleep(1)
        return await model(provider, system, prompt)

    cfg = await store.call("get_settings")
    cfg.understanding_timeout = 0.5
    await store.call("save_settings", cfg)
    result = await Engine(store, slow_plan).recall(SCOPE, "绘画比赛")
    assert plan_attempts == 1 and result["understanding"]["status"] == "fallback"
    assert result["selected"][0]["id"] == target
    assert result["retrieval"]["queries"] == ["绘画比赛"]


async def test_shortlist_failure_falls_back_to_bounded_original_candidates(store):
    target, _ = await seed(store)
    model = PlanningModel()
    attempts = 0

    async def fail_selection(provider, system, prompt):
        nonlocal attempts
        if json.loads(prompt)["schema"]["title"] == "CandidateShortlist":
            attempts += 1
            return "invalid JSON"
        return await model(provider, system, prompt)

    result = await Engine(store, fail_selection).recall(SCOPE, "绘画比赛")
    assert attempts == 1 and result["selection"]["status"] == "fallback"
    assert result["selected"][0]["id"] == target


@pytest.mark.parametrize("status", [401, 403, 429, 503])
@pytest.mark.parametrize("stage", ["QueryPlan", "CandidateShortlist"])
async def test_http_failure_stops_this_run_without_cascading_model_calls(store, status, stage):
    await seed(store)
    model = PlanningModel()
    requests = []

    class ProtocolError(Exception):
        status_code = status

    async def fail_once(provider, system, prompt):
        title = json.loads(prompt)["schema"]["title"]
        requests.append(title)
        if title == stage:
            raise ProtocolError("provider body must not escape")
        return await model(provider, system, prompt)

    result = await Engine(store, fail_once).recall(SCOPE, "绘画比赛")
    assert requests == (["QueryPlan"] if stage == "QueryPlan" else ["QueryPlan", stage])
    assert not result["injection"]
    assert f"HTTP {status}" in result["reason"]
    assert "provider body" not in result["reason"]
    trace = result["understanding"] if stage == "QueryPlan" else result["selection"]
    assert trace["status"] == "failed"


@pytest.mark.parametrize("response", ["", "   ", None])
async def test_empty_provider_output_does_not_cascade_into_later_model_calls(store, response):
    await seed(store)
    attempts = 0

    async def empty(provider, system, prompt):
        nonlocal attempts
        attempts += 1
        return response

    result = await Engine(store, empty).recall(SCOPE, "绘画比赛")
    assert attempts == 1 and not result["injection"]
    assert "模型返回空内容或非文本" in result["reason"]
    assert result["understanding"]["status"] == "failed"


async def test_successful_grounding_batch_survives_another_batch_failure(store):
    ids = []
    for i in range(6):
        mid, _ = await seed(
            store, sender=f"person{i}", text=f"我报名绘画比赛第{i}组", summary=f"绘画比赛第{i}组"
        )
        ids.append(mid)
    model = PlanningModel()
    batches = 0

    async def fail_one_batch(provider, system, prompt):
        nonlocal batches
        if json.loads(prompt)["schema"]["title"] == "GroundedReview":
            batches += 1
            if batches == 1:
                return "invalid JSON"
        return await model(provider, system, prompt)

    result = await Engine(store, fail_one_batch).recall(SCOPE, "绘画比赛")
    assert batches == 2
    assert result["selected"] and all(m["id"] in ids for m in result["selected"])
    assert result["review_errors"] and "部分候选核验失败" in result["reason"]


@pytest.mark.parametrize("query", ["哈哈", "今天天气如何", "发语音", "What's next?", "どう思う？"])
async def test_every_text_shape_reaches_model_understanding_without_keyword_gates(store, query):
    model = PlanningModel()
    result = await Engine(store, model).recall(SCOPE, query)
    assert model.requests[0]["schema"]["title"] == "QueryPlan"
    assert result["retrieval"]["queries"][0] == query


def test_multibranch_fusion_preserves_strongest_semantic_hit():
    merged = merge_rankings(
        [{"id": "m", "semantic_score": 0.8}], [{"id": "m", "semantic_score": 0.3}]
    )
    assert merged[0]["semantic_score"] == 0.8


def test_summary_pool_is_bounded_and_never_contains_raw_neighbor_chats():
    memories = [
        {
            "id": str(i),
            "subject_id": "alice",
            "summary": "画" * 600,
            "kind": "goal",
            "sources": [{"text": "不应在排序时发送"}],
        }
        for i in range(48)
    ]
    payload, aliases = shortlist_payload(memories, {"query": "比赛"}, 48, 6, max_chars=2000)
    assert 0 < len(aliases) < 48
    assert len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) <= 2000
    assert "不应在排序时发送" not in json.dumps(payload, ensure_ascii=False)
