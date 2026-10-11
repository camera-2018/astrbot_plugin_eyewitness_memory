"""Exercise AstrBot + its real SDK against loopback only; no real model or data."""

import asyncio
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

from eyewitness.engine import Engine
from eyewitness.generation import generate_once
from eyewitness.models import Candidate, Settings
from eyewitness.store import Store
from eyewitness.usage import ModelReply


async def check():
    with tempfile.TemporaryDirectory(prefix="eyewitness-retry-check-") as root:
        previous = os.environ.get("ASTRBOT_ROOT")
        os.environ["ASTRBOT_ROOT"] = root
        try:
            await exercise()
        finally:
            if previous is None:
                os.environ.pop("ASTRBOT_ROOT", None)
            else:
                os.environ["ASTRBOT_ROOT"] = previous


async def exercise():
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial

    calls = Counter()
    entered, release = asyncio.Event(), asyncio.Event()

    async def endpoint(request):
        payload = await request.json()
        prompt = payload["messages"][-1]["content"]
        assert isinstance(prompt, str)
        assert "request_max_retries" not in payload and "max_retries" not in payload
        assert payload["model"] == "retry-test"
        assert payload["metadata"] == {"source": "retry-check"}
        key = prompt
        if prompt.startswith("{"):
            key = json.loads(prompt)["data"]["query"].split()[-1]
        calls[key] += 1
        status = {
            "limited": 429,
            "unavailable": 503,
            "guard-limited": 429,
            "guard-unavailable": 503,
        }.get(key, 200)
        if prompt == "normal" and calls[prompt] < 3:
            status = 503
        if prompt == "cancel":
            entered.set()
            await release.wait()
        if status != 200:
            return web.json_response(
                {"error": {"message": "synthetic failure", "type": "upstream_error"}},
                status=status,
            )
        return web.json_response(
            {
                "id": "synthetic-retry-check",
                "object": "chat.completion",
                "created": 1,
                "model": "retry-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "" if key == "guard-empty" else "ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", endpoint)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    server = await asyncio.get_running_loop().create_server(runner.server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    provider = ProviderOpenAIOfficial(
        {
            "id": "retry-test",
            "type": "openai_chat_completion",
            "model": "retry-test",
            "key": ["synthetic-key-one", "synthetic-key-two"],
            "api_base": f"http://127.0.0.1:{port}/v1",
            "custom_extra_body": {"metadata": {"source": "retry-check"}},
            "timeout": 5,
        },
        {},
    )
    context = SimpleNamespace(get_provider_by_id=lambda _: provider)
    original_client, original_keys = provider.client, provider.api_keys
    sdk_retries = original_client.max_retries
    try:
        for prompt, status in (("limited", 429), ("unavailable", 503)):
            try:
                await generate_once(context, "retry-test", "synthetic system", prompt)
            except Exception as exc:
                assert getattr(exc, "status_code", None) == status, type(exc).__name__
            else:
                raise AssertionError("Expected synthetic error")
            assert calls[prompt] == 1, dict(calls)
        print("PASS: 429/503 each issue one real HTTP request, even with two API keys")

        with tempfile.TemporaryDirectory(prefix="eyewitness-phase-guard-") as root:
            store = Store(Path(root) / "synthetic.sqlite3")
            await store.call("open")
            try:
                scope = "test:GroupMessage:100"
                await store.call(
                    "save_settings",
                    Settings(mode="active", allowed_scopes=[scope], provider_id="retry-test"),
                )
                text = "我准备参加绘画比赛"
                mid = await store.call("capture", scope, "synthetic-1", "alice", "Alice", text)
                rows = await store.call("source_rows", scope, [mid])
                await store.call(
                    "save_extraction",
                    rows,
                    [
                        Candidate(
                            summary=text,
                            kind="goal",
                            subject_id="alice",
                            stance="self_report",
                            evidence=[{"message_id": mid, "quote": text}],
                        )
                    ],
                )

                async def generate(pid, system, prompt):
                    return ModelReply.from_response(
                        await generate_once(context, pid, system, prompt)
                    )

                engine = Engine(store, generate, report=lambda _: None)
                for key in ("guard-limited", "guard-unavailable", "guard-empty"):
                    result = await engine.recall(scope, "绘画比赛 " + key)
                    assert calls[key] == 1, dict(calls)
                    assert result["understanding"]["status"] == "failed", result
                    assert not result["injection"] and not result.get("selection"), result
                print(
                    "PASS: entire recall stops after one HTTP 429/503 or empty response; no phase retries"
                )
            finally:
                await store.call("close")

        memory, normal = await asyncio.gather(
            generate_once(context, "retry-test", "synthetic system", "memory"),
            provider.text_chat(prompt="normal", max_tokens=3000, request_max_retries=1),
        )
        assert calls["memory"] == 1 and calls["normal"] == 3, dict(calls)
        assert memory.completion_text == normal.completion_text == "ok"
        assert ModelReply.from_response(memory).input_tokens == 12
        print("PASS: concurrent normal chat retains SDK retries; response/usage/options preserved")

        task = asyncio.create_task(
            generate_once(context, "retry-test", "synthetic system", "cancel")
        )
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation must propagate")
        release.set()
        assert calls["cancel"] == 1
        assert provider.client is original_client and original_client.max_retries == sdk_retries
        assert provider.api_keys is original_keys and len(original_keys) == 2
        assert not original_client.is_closed()
        await generate_once(context, "retry-test", "synthetic system", "after-cancel")
        assert calls["after-cancel"] == 1
        print("PASS: cancellation does not retry, close, or mutate the shared provider/client")
        print(json.dumps({"http_requests": dict(calls)}, sort_keys=True))
    finally:
        release.set()
        await provider.terminate()
        server.close()
        await server.wait_closed()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(check())
