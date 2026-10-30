import asyncio
import time

import pytest

from eyewitness.engine import Engine
from eyewitness.recall_gate import guard_reason
from eyewitness.systemone import parse_reply
from tests.conftest import SCOPE, seed
from tests.test_engine import Model, answers
from tests.test_systemone import enable, response


async def setup(store, choice="unneeded", p=0.99, error=None):
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
        raw = response(body, choice)
        raw["answers"]["recall"]["probabilities"] = {
            o: p if o == choice else (1 - p) / 2 for o in body["questions"]["recall"]["criteria"]
        }
        return parse_reply(raw, body)

    return cfg, systemone, calls


async def test_negative_skips_all_gemini_and_still_records_trace(store):
    cfg, systemone, calls = await setup(store)

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
        ("unneeded", 0.89, None),
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
def test_history_quotes_and_truncation_protected(query, reply, conversation):
    assert guard_reason(query, reply, conversation)


async def test_protected_query_does_not_call_gate(store):
    _, systemone, calls = await setup(store)
    engine = Engine(store, Model([]), systemone=systemone)
    result = {}
    skip = await engine.skip_recall(
        await store.call("get_settings"), "我是谁", "", [], result, deadline=time.monotonic() + 30
    )
    assert not skip and not calls and result["recall_gate"]["status"] == "protected"


async def test_disabled_and_missing_provider_fail_open(store):
    cfg, systemone, calls = await setup(store)
    engine = Engine(store, Model([]), systemone=systemone)
    for patch in ({"recall_gate_enabled": False}, {"systemone_provider_id": ""}):
        r = {}
        assert not await engine.skip_recall(
            cfg.model_copy(update=patch), "哈哈哈", "", [], r, deadline=time.monotonic() + 30
        )
    assert not calls
