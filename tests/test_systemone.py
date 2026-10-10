import asyncio
import copy
import json
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from pydantic import ValidationError

from eyewitness.engine import Engine
from eyewitness.errors import AuxiliaryModelFailure
from eyewitness.models import Settings
from eyewitness.systemone import (
    SystemOneClient,
    candidate_request,
    endpoint,
    parse_reply,
)
from tests.conftest import SCOPE, seed
from tests.test_engine import Model, answers


def body(phase="relevance"):
    options = (
        ["relevant", "irrelevant", "uncertain"]
        if phase == "relevance"
        else ["supported", "unsupported", "uncertain"]
    )
    return {
        "model": "jev-1.13.0",
        "state": "test",
        "questions": {
            phase: {
                "type": "choice",
                "instructions": "test",
                "criteria": dict.fromkeys(options, "test"),
            },
        },
    }


def response(request, choice=None):
    phase, question = next(iter(request["questions"].items()))
    options = list(question["criteria"])
    choice = choice or options[0]
    return {
        "model": request["model"],
        "answers": {
            phase: {
                "type": "choice",
                "choice": choice,
                "confidence": 0.8,
                "probabilities": {o: 0.9 if o == choice else 0.05 for o in options},
            },
        },
        "usage": {"input_tokens": 123, "output_tokens": 10},
    }


async def enable(store, **patch):
    cfg = await store.call("get_settings")
    cfg = cfg.model_copy(
        update={"systemone_provider_id": "gateway", "systemone_model": "jev-1.13.0", **patch}
    )
    await store.call("save_settings", cfg)
    return cfg


def test_endpoint_keeps_prefix_and_rejects_cross_origin_paths():
    assert (
        endpoint("http://localhost:3067/v1/", "/typesafe/v1/systemone")
        == "http://localhost:3067/typesafe/v1/systemone"
    )
    assert (
        endpoint("https://example.test/proxy/v1", "/clef/v1/systemone")
        == "https://example.test/proxy/clef/v1/systemone"
    )
    assert endpoint("https://example.test", "/v1/systemone") == "https://example.test/v1/systemone"
    for path in (
        "//evil.test/v1/systemone",
        "https://evil.test/v1/systemone",
        "/../v1/systemone",
        "/x%2f/v1/systemone",
        "/v1/systemone?key=x",
    ):
        with pytest.raises(ValidationError):
            Settings(systemone_api_path=path)
    for url in (
        "https://user:password@example.test/v1",
        "file:///tmp/x",
        "https://example.test/v1?key=x",
    ):
        with pytest.raises(AuxiliaryModelFailure):
            endpoint(url, "/v1/systemone")


def test_parse_validates_model_alias_and_cloudflare_envelope():
    request = body()
    assert parse_reply(response(request), request).input_tokens == 123
    request["model"] = "jev-latest"
    result = response(request)
    result["model"] = "jev-1.13.0"
    assert parse_reply(result, request).model == "jev-1.13.0"
    request["model"] = "clef-flash"
    result = response(request)
    result["model"] = "@cf/cloudflare/clef-flash"
    assert parse_reply({"success": True, "result": result}, request).output_tokens == 10


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(model="jev-0.0.0"),
        lambda b: b.update(answers={}),
        lambda b: b["answers"]["relevance"].update(type="noul"),
        lambda b: b["answers"]["relevance"].update(choice="unknown"),
        lambda b: b["answers"]["relevance"].update(choice="irrelevant"),
        lambda b: b["answers"]["relevance"].update(confidence=True),
        lambda b: b["answers"]["relevance"]["probabilities"].update(relevant=float("nan")),
        lambda b: b["answers"]["relevance"]["probabilities"].update(relevant=0.4),
        lambda b: b["usage"].update(input_tokens=True),
        lambda b: b["usage"].update(input_tokens=65537),
        lambda b: b.pop("usage"),
    ],
)
def test_malformed_success_is_not_a_decision(change):
    request = body()
    result = response(request)
    change(result)
    with pytest.raises(AuxiliaryModelFailure):
        parse_reply(result, request)


async def test_candidate_payload_is_local_and_support_has_no_current_query(store):
    mid, source = await seed(store)
    memory = await store.call("detail", mid, SCOPE)
    rows = await store.call("source_context", mid, SCOPE)
    cfg = Settings(systemone_model="jev-1.13.0")
    kwargs = {
        "query": "当前问题的秘密话题",
        "sender_id": "bob",
        "conversation": [{"text": "当前会话"}],
    }
    rel = candidate_request(cfg, memory, rows, phase="relevance", **kwargs)
    support = candidate_request(cfg, memory, rows, phase="support", **kwargs)
    assert rel["state"]["current"]["query"] == kwargs["query"]
    assert "current" not in support["state"]
    assert "当前问题的秘密话题" not in json.dumps(support, ensure_ascii=False)
    assert support["state"]["memory"]["source_ids"] == [source]
    assert support["state"]["source_chat"][0]["is_source"]
    assert "+08:00" in support["state"]["source_chat"][0]["time"]
    assert "采集时间" in support["state"]["source_chat"][0]["time_kind"]
    with pytest.raises(AuxiliaryModelFailure):
        candidate_request(cfg, memory, [], phase="support")


async def test_negative_systemone_never_discards_gemini_accepted_candidate(store):
    mid, source = await seed(store)
    await enable(store)
    calls = []

    async def decide(cfg, request):
        calls.append(copy.deepcopy(request))
        return parse_reply(response(request, "irrelevant"), request)

    gemini = Model(answers(mid, source))
    result = await Engine(store, gemini, systemone=decide).recall(SCOPE, "之前绘画比赛", "alice")
    assert result["injection"] and result["selected"][0]["id"] == mid
    assert len(calls) == 1 and gemini.calls == 1
    assert result["systemone"]["decisions"][mid]["relevance"]["choice"] == "irrelevant"
    usage = next(s for s in await store.call("usage_stats") if s["stage"] == "System One 相关性")
    assert usage["calls"] == 1 and usage["input_tokens"] == 123


async def test_positive_systemone_never_overrides_gemini_rejection(store):
    mid, _ = await seed(store)
    await enable(store)
    phases = []

    async def decide(cfg, request):
        phases.extend(request["questions"])
        return parse_reply(response(request), request)

    gemini = Model([{"decisions": [{"id": mid, "action": "reject", "reason": "原文不支持"}]}])
    result = await Engine(store, gemini, systemone=decide).recall(SCOPE, "绘画比赛", "alice")
    assert not result["injection"] and not result["selected"]
    assert phases == ["relevance", "support"] and gemini.calls == 1


async def test_failed_source_initial_check_still_goes_to_gemini(store):
    mid, source = await seed(store)
    await enable(store)

    async def decide(cfg, request):
        phase = next(iter(request["questions"]))
        choice = "unsupported" if phase == "support" else "relevant"
        return parse_reply(response(request, choice), request)

    result = await Engine(store, Model(answers(mid, source)), systemone=decide).recall(
        SCOPE, "绘画比赛"
    )
    assert result["injection"]
    assert result["systemone"]["decisions"][mid]["support"]["choice"] == "unsupported"


async def test_priority_sort_preserves_all_candidates_and_isolates_their_context(store):
    candidates, pool, contexts = [], [], {}
    cfg = await enable(store)
    for sender in ("alice", "bob", "carol"):
        mid, source = await seed(store, sender=sender)
        m = await store.call("detail", mid, SCOPE)
        rows = await store.call("source_rows", SCOPE, [source])
        candidates.append(m)
        pool.extend(rows)
        contexts[mid] = {source}
    requests = []

    async def decide(cfg, request):
        requests.append(request)
        if next(iter(request["questions"])) == "relevance":
            choice = (
                "relevant" if request["state"]["memory"]["subject_id"] == "carol" else "irrelevant"
            )
        else:
            choice = "supported"
        return parse_reply(response(request, choice), request)

    result = {}
    ranked = await Engine(store, Model([]), systemone=decide).prioritize(
        cfg,
        candidates,
        pool,
        contexts,
        "绘画比赛",
        "bob",
        [],
        None,
        result,
        deadline=asyncio.get_running_loop().time() + 3,
    )
    assert [m["subject_id"] for m in ranked] == ["carol", "alice", "bob"]
    assert {m["id"] for m in ranked} == {m["id"] for m in candidates}
    assert len(requests) == 4
    for request in requests:
        state = request["state"]
        assert len(state["source_chat"]) == 1
        assert state["source_chat"][0]["sender_id"] == state["memory"]["subject_id"]


async def test_systemone_failure_falls_back_without_retry(store):
    mid, source = await seed(store)
    await enable(store)
    count = 0

    async def fail(cfg, request):
        nonlocal count
        count += 1
        raise AuxiliaryModelFailure("System One：上游 HTTP 429，未重试")

    result = await Engine(store, Model(answers(mid, source)), systemone=fail).recall(
        SCOPE, "绘画比赛"
    )
    assert result["injection"] and count == 1
    assert "429" in result["systemone"]["decisions"][mid]["error"]


async def test_advice_timeout_preserves_final_verification_budget_and_cancels_job(store):
    mid, source = await seed(store)
    await enable(store, online_timeout=0.8, systemone_timeout=0.2)
    stopped = asyncio.Event()

    async def slow(cfg, request):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    start = time.monotonic()
    result = await Engine(store, Model(answers(mid, source)), systemone=slow).recall(
        SCOPE, "绘画比赛"
    )
    assert result["injection"] and stopped.is_set()
    assert time.monotonic() - start < 0.8
    assert result["systemone"]["status"] == "partial"
    usage = next(s for s in await store.call("usage_stats") if s["stage"] == "System One 相关性")
    assert usage["running"] == 0 and usage["failed"] == 1


async def test_disabling_systemone_makes_no_advisory_calls(store):
    mid, source = await seed(store)

    async def unexpected(*args):
        pytest.fail("Disabled System One called")

    result = await Engine(store, Model(answers(mid, source)), systemone=unexpected).recall(
        SCOPE, "绘画比赛"
    )
    assert result["injection"] and "systemone" not in result


async def test_http_reuses_native_credentials_with_bounded_concurrency(aiohttp_server):
    calls, running, peak = [], 0, 0

    async def handler(request):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        calls.append((request.path, request.headers["Authorization"], await request.json()))
        try:
            await asyncio.sleep(0.02)
            return web.json_response(response(calls[-1][2]))
        finally:
            running -= 1

    app = web.Application()
    app.router.add_post("/typesafe/v1/systemone", handler)
    server = await aiohttp_server(app)
    keys = ["synthetic-one"]
    provider = SimpleNamespace(
        provider_config={"api_base": str(server.make_url("/v1"))}, get_keys=lambda: keys
    )
    client = SystemOneClient(lambda _: provider)
    cfg = Settings(systemone_provider_id="gateway", systemone_model="jev-1.13.0")
    try:
        await asyncio.gather(*(client.call(cfg, body()) for _ in range(4)))
        keys[0] = "synthetic-two"
        await client.call(cfg, body())
        assert peak == 2
        assert all(p == "/typesafe/v1/systemone" for p, _, _ in calls)
        assert calls[-1][1] == "Bearer synthetic-two"
        assert all("messages" not in b for _, _, b in calls)
    finally:
        await client.close()
    assert client.session.closed


@pytest.mark.parametrize("kind", ["redirect", "html", "large", "rate_limit"])
async def test_http_failure_does_not_follow_redirect_or_retry(aiohttp_server, kind):
    hits = []

    async def handler(request):
        hits.append(request.path)
        if kind == "redirect":
            raise web.HTTPFound("/leak")
        if kind == "html":
            return web.Response(text="<html>upstream frontend</html>", content_type="text/html")
        if kind == "large":
            return web.Response(body=b" " * 70000, content_type="application/json")
        return web.json_response({"detail": "secret-provider-error"}, status=429)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    server = await aiohttp_server(app)
    provider = SimpleNamespace(
        provider_config={"api_base": str(server.make_url("/v1"))},
        get_keys=lambda: ["synthetic-secret"],
    )
    client = SystemOneClient(lambda _: provider)
    try:
        with pytest.raises(AuxiliaryModelFailure) as caught:
            await client.call(Settings(systemone_provider_id="gateway"), body())
        assert len(hits) == 1 and hits[0] != "/leak"
        assert "synthetic-secret" not in str(caught.value)
        assert "secret-provider-error" not in str(caught.value)
    finally:
        await client.close()
