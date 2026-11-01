"""Exercise the real AstrBot 4.29.4 service and adapter, without network/messages."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "astrbot.core.provider.decision", reason="native path requires AstrBot 4.29.4 SDK"
)
import httpx
from astrbot.core.provider.decision import DecisionService
from astrbot.core.provider.sources.system_one_decision_source import SystemOneDecisionProvider

from eyewitness.models import Settings
from eyewitness.native_decision import NativeDecisionClient, NativeDecisionFailure
from eyewitness.systemone import SystemOneClient

BODY = {
    "model": "FastBrain-9B",
    "state": {"text": "木头帮忙看看", "fewshot": ["原样保留"]},
    "questions": {
        "should_reply": {
            "type": "choice",
            "instructions": "Decide without changing IDs",
            "criteria": {"True": "Participate", "False": "Stay silent"},
        }
    },
}


def response_for(body):
    return {
        "model": body["model"],
        "answers": {
            head: {
                "type": "choice",
                "choice": next(iter(q["criteria"])),
                "confidence": 0.8,
                "probabilities": {key: 1 / len(q["criteria"]) for key in q["criteria"]},
            }
            for head, q in body["questions"].items()
        },
        "usage": {"input_tokens": 123, "output_tokens": 4},
    }


@pytest.fixture
async def native():
    seen = []

    async def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=response_for(body))

    provider = SystemOneDecisionProvider(
        {
            "id": "native",
            "type": "system_one_decision",
            "api_base": "http://test/v1/systemone",
            "model": "FastBrain-9B",
            "key": "dummy",
            "timeout": 2,
            "total_timeout": 2,
        },
        {},
        transport=httpx.MockTransport(handler),
    )

    async def resolve(provider_id):
        assert provider_id == "native"
        return provider

    service = DecisionService(SimpleNamespace(get_provider_by_id=resolve))
    yield NativeDecisionClient(service, resolve), provider, seen, service, resolve
    await provider.terminate()


@pytest.mark.asyncio
async def test_real_service_preserves_payload_and_unknown_usage(native):
    bridge, provider, seen, service, resolve = native
    client = SystemOneClient(resolve, service)
    cfg = Settings(
        systemone_decision_provider_id="native",
        systemone_model="FastBrain-9B",
        fastbrain_priority_enabled=True,
    )
    reply = await client.call(cfg, BODY)
    assert seen == [BODY]
    assert reply.answers["should_reply"]["probabilities"] == {"True": 0.5, "False": 0.5}
    assert reply.input_tokens is None and reply.output_tokens is None
    assert reply.priority == "normal"
    await client.close()
    assert not provider.client.is_closed  # The core owns this client.


@pytest.mark.asyncio
async def test_config_mismatch_does_not_send_or_fallback(native):
    bridge, _, seen, _, _ = native
    with pytest.raises(NativeDecisionFailure, match="configured_model_mismatch"):
        await bridge.ask("native", {**BODY, "model": "other"}, timeout=1)
    assert seen == []


@pytest.mark.asyncio
async def test_http_backoff_preserved_no_retries(native):
    bridge, provider, _, _, _ = native
    calls = []

    async def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "120"}, text="private-response")

    await provider.client.aclose()
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(NativeDecisionFailure) as exc:
        await bridge.ask("native", BODY, timeout=1, priority_enabled=True)
    assert exc.value.reason == "http_429" and exc.value.retry_after == 120
    assert "private-response" not in str(exc.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_timeout_releases_shared_lane(native):
    bridge, provider, _, _, _ = native

    async def slow(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json=response_for(json.loads(request.content)))

    await provider.client.aclose()
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    with pytest.raises(NativeDecisionFailure, match="timeout"):
        await bridge.ask("native", BODY, timeout=0.01, priority_enabled=True)
    reply = await bridge.ask("native", BODY, timeout=1, priority_enabled=True)
    assert reply.answers


@pytest.mark.asyncio
async def test_native_urgent_bypasses_waiting_normal_and_cancelled_request(native):
    bridge, provider, _, service, resolve = native
    entered, release = asyncio.Event(), asyncio.Event()
    order = []

    async def handler(request):
        body = json.loads(request.content)
        name = body["state"]["name"]
        order.append(name)
        if name == "active":
            entered.set()
            await release.wait()
        return httpx.Response(200, json=response_for(body))

    await provider.client.aclose()
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def call(name, urgent=False):
        return await bridge.ask(
            "native",
            {**BODY, "state": {"name": name}},
            timeout=1,
            urgent=urgent,
            priority_enabled=True,
        )

    active = asyncio.create_task(call("active"))
    await entered.wait()
    normal = asyncio.create_task(call("normal"))
    cancelled = asyncio.create_task(call("cancelled"))
    urgent = asyncio.create_task(call("gate", True))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    await asyncio.gather(active, normal, urgent)
    assert order == ["active", "gate", "normal"]
