import asyncio
import time

import pytest

from eyewitness.engine import Engine
from eyewitness.systemone import parse_reply
from tests.conftest import SCOPE, seed
from tests.test_engine import Model, answers
from tests.test_systemone import enable


def gate_response(body, choice="unneeded", p=0.99, signal="none", form="standalone"):
    choices = {
        "recall": choice,
        "memory_signal": signal,
        "query_form": form,
        "context_source": "uncertain",
        "search_focus": "other",
    }
    return {
        "model": body["model"],
        "answers": {
            key: {
                "type": "choice",
                "choice": choices[key],
                "confidence": p,
                "probabilities": {
                    o: p if o == choices[key] else (1 - p) / (len(q["criteria"]) - 1)
                    for o in q["criteria"]
                },
            }
            for key, q in body["questions"].items()
        },
        "usage": {"input_tokens": 123, "output_tokens": 0},
    }


async def setup(store, choice="unneeded", p=0.99, error=None, signal="none", form="standalone"):

    cfg = await enable(store, recall_gate_enabled=True, recall_gate_timeout=0.2)
    calls = []

    async def systemone(cfg, body):
        calls.append(body)
        if error == "timeout":
            await asyncio.sleep(2)
        if error == "cancel":
            raise asyncio.CancelledError()
        if error == "invalid":
            return parse_reply({}, body)
        if error == "network":
            raise ConnectionError("test")
        if "recall" in body["questions"]:
            raw = gate_response(body, choice, p, signal, form)
        else:
            options = body["questions"]["select"]["criteria"]
            chosen = next(k for k in options if k != "none")
            raw = {
                "model": body["model"],
                "answers": {
                    "select": {
                        "type": "choice",
                        "choice": chosen,
                        "confidence": 0.9,
                        "probabilities": {
                            k: 0.9 if k == chosen else 0.1 / (len(options) - 1) for k in options
                        },
                    }
                },
                "usage": {"input_tokens": 123, "output_tokens": 0},
            }
        return parse_reply(raw, body)

    return cfg, systemone, calls


@pytest.mark.parametrize("p", [0.85, 0.89, 0.99])
async def test_negative_skips_all_gemini_and_still_records_trace(store, p):
    cfg, systemone, calls = await setup(store, p=p)

    async def forbidden(*args):
        pytest.fail("Gemini must not be called")

    r = await Engine(store, forbidden, systemone=systemone).recall(SCOPE, "哈哈哈")
    assert r["recall_gate"]["skip"] and not r["injection"]
    assert len(calls) == 1
    assert len(await store.call("traces", SCOPE)) == 1


@pytest.mark.parametrize(
    "choice,p,error",
    [
        ("needed", 0.99, None),
        ("uncertain", 0.99, None),
        ("unneeded", 0.99, "timeout"),
        ("unneeded", 0.99, "invalid"),
        ("unneeded", 0.99, "network"),
    ],
)
async def test_uncertain_and_failures_keep_original_recall(store, choice, p, error):
    _, systemone, calls = await setup(store, choice, p, error)
    mid, source = await seed(store)
    # Disable advisory ranking to focus on the routing failure fallback.
    engine = Engine(store, Model(answers(mid, source)), systemone=systemone)

    async def no_rank(cfg, candidates, *args, **kwargs):
        return candidates

    engine.prioritize = no_rank
    r = await engine.recall(SCOPE, "绘画比赛", "alice")
    assert not r["recall_gate"]["skip"]
    assert r["selected"][0]["id"] == mid
    assert r["injection"] and len(calls) == 1


async def test_cancellation_is_not_swallowed(store):
    _, systemone, _ = await setup(store, error="cancel")
    with pytest.raises(asyncio.CancelledError):
        await Engine(store, Model([]), systemone=systemone).recall(SCOPE, "哈哈哈")


@pytest.mark.parametrize(
    "query,reply,conversation",
    [
        ("我以前用什么模型", "", []),
        ("他用啥", "", []),
        ("哈哈哈", "quote-id", []),
        ("我是谁", "", []),
        ("How did it go last time?", "", []),
        ("x" * 1500, "", []),
        ("哈哈哈", "", [{"text": "x" * 1200}]),
    ],
)
async def test_keywords_and_quotes_are_classified_not_rule_bypassed(
    store, query, reply, conversation
):
    cfg, systemone, calls = await setup(store, p=0.51)
    engine = Engine(store, Model([]), systemone=systemone)
    r = {}
    assert not await engine.skip_recall(
        cfg, query, reply, conversation, r, deadline=time.monotonic() + 30
    )
    assert len(calls) == 1 and r["recall_gate"]["choice"] == "unneeded"
    assert calls[0]["state"]["current_message"] == query
    assert calls[0]["state"]["recent_conversation"] == conversation
    if reply:
        assert calls[0]["state"]["has_reference"] is True
        assert calls[0]["state"]["quoted_message"] is None


async def test_quote_text_reaches_classifier(store):
    cfg, systemone, calls = await setup(store, choice="uncertain")
    r = {}
    assert not await Engine(store, Model([]), systemone=systemone).skip_recall(
        cfg,
        "看看这个",
        "reply",
        [],
        r,
        deadline=time.monotonic() + 30,
        quoted={"text": "我计划参加比赛", "sender_id": "alice"},
    )
    assert calls[0]["state"]["quoted_message"]["text"] == "我计划参加比赛"


async def test_disabled_and_missing_provider_fail_open(store):
    cfg, systemone, calls = await setup(store)
    engine = Engine(store, Model([]), systemone=systemone)
    for patch in ({"recall_gate_enabled": False}, {"systemone_provider_id": ""}):
        r = {}
        assert not await engine.skip_recall(
            cfg.model_copy(update=patch), "哈哈哈", "", [], r, deadline=time.monotonic() + 30
        )
    assert not calls


@pytest.mark.parametrize(
    "p,signal,form,reply",
    [
        (0.34, "none", "standalone", ""),
        (0.84, "none", "standalone", ""),
        (0.99, "person", "standalone", ""),
        (0.99, "config", "standalone", ""),
        (0.99, "history", "standalone", ""),
        (0.99, "uncertain", "standalone", ""),
        (0.99, "none", "contextual", ""),
        (0.99, "none", "uncertain", ""),
        (0.99, "none", "standalone", "missing-quote"),
    ],
)
async def test_conflicting_heads_weak_negative_and_missing_quote_keep_recall(
    store, p, signal, form, reply
):
    cfg, systemone, calls = await setup(store, p=p, signal=signal, form=form)
    r = {}
    assert not await Engine(store, Model([]), systemone=systemone).skip_recall(
        cfg, "你还记得吗", reply, [], r, deadline=time.monotonic() + 30
    )
    assert len(calls) == 1
    assert r["recall_gate"]["memory_signal"] == signal
    assert r["recall_gate"]["query_form"] == form


async def test_incomplete_multihead_response_fails_open(store):
    cfg, _, _ = await setup(store)

    async def missing_head(cfg, body):
        raw = gate_response(body)
        del raw["answers"]["memory_signal"]
        return parse_reply(raw, body)

    r = {}
    assert not await Engine(store, Model([]), systemone=missing_head).skip_recall(
        cfg, "你好", "", [], r, deadline=time.monotonic() + 30
    )
    assert r["recall_gate"]["status"] == "fallback"


async def test_specific_search_focus_overrides_negative_recall_gate(store):
    cfg, _, _ = await setup(store)
    cfg = cfg.model_copy(update={"recall_fast_path_enabled": True})

    async def classify(cfg, body):
        raw = gate_response(body)
        options = body["questions"]["search_focus"]["criteria"]
        raw["answers"]["search_focus"] = {
            "type": "choice",
            "choice": "identity",
            "confidence": 0.9,
            "probabilities": {
                k: 0.9 if k == "identity" else 0.1 / (len(options) - 1) for k in options
            },
        }
        return parse_reply(raw, body)

    r = {}
    assert not await Engine(store, Model([]), systemone=classify).skip_recall(
        cfg, "介绍这位群友的身世", "", [], r, deadline=time.monotonic() + 3
    )
    assert r["recall_gate"]["search_focus"] == "identity"


@pytest.mark.parametrize("focus", ["other", "uncertain"])
async def test_no_specific_search_focus_does_not_force_clear_chitchat_to_recall(store, focus):
    cfg, _, _ = await setup(store)
    cfg = cfg.model_copy(update={"recall_fast_path_enabled": True})

    async def classify(cfg, body):
        raw = gate_response(body)
        options = body["questions"]["search_focus"]["criteria"]
        raw["answers"]["search_focus"] = {
            "type": "choice",
            "choice": focus,
            "confidence": 0.9,
            "probabilities": {k: 0.9 if k == focus else 0.1 / (len(options) - 1) for k in options},
        }
        return parse_reply(raw, body)

    r = {}
    assert await Engine(store, Model([]), systemone=classify).skip_recall(
        cfg, "晚上好", "", [], r, deadline=time.monotonic() + 3
    )


def test_compact_gate_preserves_context_and_reference_indices():
    from eyewitness.fastbrain_planning import plan_queries
    from eyewitness.models import Settings
    from eyewitness.recall_gate import gate_request

    cfg = Settings(recall_fast_path_enabled=True)
    conversation = [
        {"role": "user", "text": "小王买了显卡，预算不超过三千。"},
        {"role": "assistant", "text": "相洋提过自己的 NAS 升级计划。"},
    ]
    quoted = {"text": "他说还要保留旧模型", "sender_id": "alice"}
    body = gate_request(cfg, "他后来怎么改的？", conversation, "quote1", quoted)
    assert body["state"]["recent_conversation"] == conversation
    assert body["state"]["quoted_message"] == quoted
    assert (
        body["questions"]["context_source"]["criteria"]["c2"] == "第2条：" + conversation[1]["text"]
    )
    queries, _ = plan_queries(
        cfg,
        "他后来怎么改的？",
        conversation,
        {"context_choice": "c2", "context_probabilities": {"c2": 0.99}},
    )
    assert any(conversation[1]["text"] in q for q in queries)
    assert not any(conversation[0]["text"] in q for q in queries)


def test_gate_default_timeout_matches_schema():
    import json
    from pathlib import Path

    from eyewitness.models import Settings

    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert Settings().recall_gate_timeout == schema["recall_gate_timeout"]["default"] == 3.0


async def test_three_second_gate_accepts_reply_after_old_two_second_limit(store):
    cfg, _, _ = await setup(store)
    cfg = cfg.model_copy(update={"recall_gate_timeout": 3.0})

    async def slow_reply(cfg, body):
        await asyncio.sleep(2.1)
        return parse_reply(gate_response(body, choice="needed", signal="person"), body)

    result = {}
    engine = Engine(store, Model([]), systemone=slow_reply)
    assert not await engine.skip_recall(
        cfg, "我是谁", "", [], result, deadline=time.monotonic() + 30
    )
    assert result["recall_gate"]["status"] == "success"
    assert result["recall_gate"]["elapsed_ms"] >= 2000
