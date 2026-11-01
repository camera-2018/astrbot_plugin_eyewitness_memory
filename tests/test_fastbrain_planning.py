import asyncio
import json
import time

import pytest

from eyewitness.engine import Engine
from eyewitness.fastbrain_planning import (
    diverse_verified,
    plan_queries,
    select_ranked,
    selection_requests,
)
from eyewitness.models import Settings
from eyewitness.systemone import parse_reply
from tests.conftest import SCOPE, seed
from tests.test_planning import PlanningModel
from tests.test_recall_gate import setup
from tests.test_systemone import enable


def test_context_selection_uses_only_existing_text_and_keeps_original():
    cfg = Settings()
    context = [{"role": "user", "text": "alice的绘画比赛"}, {"role": "user", "text": "刚下雨了"}]
    gate = {"status": "success", "context_choice": "c1", "context_probabilities": {"c1": 0.9}}
    q, trace = plan_queries(cfg, "那个报名到哪天", context, gate)
    assert q == ["那个报名到哪天", "alice的绘画比赛 那个报名到哪天"]
    assert trace["status"] == "fastbrain"
    q, _ = plan_queries(cfg, "原问题", context, {"status": "fallback"})
    assert q[0] == "原问题" and len(q) == 3
    assert plan_queries(
        cfg.model_copy(update={"query_expansion_limit": 0}), "原问题", context, gate
    )[0] == ["原问题"]


def test_unrecognized_context_alias_cannot_invent_a_query():
    q, _ = plan_queries(
        Settings(),
        "他呢",
        [],
        {"context_choice": "恶意关键词", "context_probabilities": {"恶意关键词": 1}},
    )
    assert q == ["他呢"]


def test_ranking_retains_local_anchor_and_includes_later_high_score():
    pool = [{"id": str(i)} for i in range(20)]
    decisions = {
        str(i): {"probabilities": {"relevant": 0.01, "uncertain": 0.01, "irrelevant": 0.98}}
        for i in range(20)
    }
    decisions["19"] = {"probabilities": {"relevant": 0.99, "uncertain": 0.005, "irrelevant": 0.005}}
    selected = select_ranked(pool, decisions, 6)
    assert selected[0]["id"] == "19" and pool[0] in selected
    assert len(selected) == 6 and len({r["id"] for r in selected}) == 6
    assert select_ranked(pool, {}, 6) == pool[:6]
    assert select_ranked(pool, decisions, 1) == pool[:1]


async def test_bounded_batches_use_summary_only_and_scope_local_ids(store):
    pool = []
    for i in range(12):
        mid, _ = await seed(store, sender=f"p{i}")
        pool.append(await store.call("detail", mid, SCOPE))
    cfg = Settings(systemone_model="FastBrain-9B", candidate_pool_size=10)
    batches = selection_requests(cfg, pool, {"query": "绘画比赛", "conversation": []}, batch_size=8)
    assert [len(a) for _, a in batches] == [8, 2]
    assert set(mid for _, a in batches for mid in a.values()) == {m["id"] for m in pool[:10]}
    for body, _ in batches:
        assert "sources" not in json.dumps(body)
        assert len(json.dumps(body, ensure_ascii=False).encode()) <= 24000


@pytest.mark.parametrize("failure", ["network", "invalid", "timeout", "unconfigured"])
async def test_selector_failure_retains_local_candidates_without_cloud_planning(store, failure):
    cfg, _, _ = await setup(store, choice="needed")
    cfg = await enable(store, recall_fast_path_enabled=True, systemone_timeout=0.2)
    pool = []
    for i in range(7):
        mid, _ = await seed(store, sender=f"p{i}")
        pool.append(await store.call("detail", mid, SCOPE))

    async def broken(cfg, body):
        if failure == "timeout":
            await asyncio.sleep(10)
        if failure == "invalid":
            return parse_reply({}, body)
        raise ConnectionError("offline")

    model = PlanningModel()
    engine = Engine(store, model, systemone=None if failure == "unconfigured" else broken)
    result = {}
    selected = await engine.fastbrain_candidates(
        cfg, pool, {"query": "绘画比赛"}, result, deadline=time.monotonic() + 3
    )
    assert selected == pool[: cfg.source_review_limit]
    assert not model.requests and result["selection"]["status"] == "fallback"
    await engine.close()


async def test_gate_failure_does_not_start_gemini_plan_or_selection(store):
    _, classifier, _ = await setup(store, error="network")
    await enable(store, recall_fast_path_enabled=True)
    await seed(store)
    model = PlanningModel()
    r = await Engine(store, model, systemone=classifier).recall(SCOPE, "绘画比赛")
    assert r["injection"]
    assert [x["schema"]["title"] for x in model.requests] == ["GroundedReview"]


def test_multi_subject_questions_keep_original_and_split_source_clauses():
    query = "请问，林夏是谁？陈默又是谁？"
    queries, trace = plan_queries(
        Settings(), query, [], {"status": "success", "search_focus": "identity"}
    )
    assert queries[0] == query
    assert any("林夏是谁 身份" in q for q in queries[1:])
    assert any("陈默又是谁 身份" in q for q in queries[1:])
    assert all("相洋" not in q for q in queries)
    assert trace["search_focus"] == "identity"


async def test_partial_selector_timeout_keeps_completed_scores_and_unknowns(store):
    cfg = await enable(store, systemone_timeout=0.2, source_review_limit=6, candidate_pool_size=48)
    pool = []
    for i in range(26):
        mid, _ = await seed(store, sender=f"p{i}")
        pool.append(await store.call("detail", mid, SCOPE))

    async def classify(cfg, body):
        options = body["questions"]["select"]["criteria"]
        if len(options) == 25:
            await asyncio.sleep(10)
        answers = {
            "select": {
                "type": "choice",
                "choice": "m2",
                "confidence": 0.99,
                "probabilities": {
                    k: 0.99 if k == "m2" else 0.01 / (len(options) - 1) for k in options
                },
            }
        }
        return parse_reply(
            {
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": 50, "output_tokens": 0},
            },
            body,
        )

    engine = Engine(store, PlanningModel(), systemone=classify)
    result = {}
    chosen = await engine.fastbrain_candidates(
        cfg, pool, {"query": "绘画比赛"}, result, deadline=time.monotonic() + 3
    )
    assert result["selection"]["status"] == "partial"
    assert set(result["selection"]["decisions"]) == {m["id"] for m in pool[24:]}
    assert len(chosen) == 6 and pool[0] in chosen and pool[25] in chosen
    await engine.close()


def test_two_identity_slots_do_not_drop_another_verified_person():
    items = [
        {"id": "a-identity", "subject_id": "alice"},
        {"id": "a-hobby", "subject_id": "alice"},
        {"id": "b-identity", "subject_id": "bob"},
    ]
    assert diverse_verified(items) == [items[0], items[2]]
    assert diverse_verified(items[:2]) == items[:2]
    assert diverse_verified([]) == []
