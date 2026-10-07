import asyncio
from types import SimpleNamespace

import pytest

from eyewitness.errors import AuxiliaryModelFailure
from eyewitness.generation import generate_once


class RateLimitError(Exception):
    status_code = 429


class Client:
    def __init__(self, calls, max_retries=2):
        self.calls = calls
        self.max_retries = max_retries
        self.api_key = "original-key"

    def with_options(self, *, max_retries):
        return Client(self.calls, max_retries)

    async def request(self, prompt):
        for _ in range(self.max_retries + 1):
            self.calls.append((prompt, self.max_retries, self.api_key))
            await asyncio.sleep(0)
        if prompt == "429":
            raise RateLimitError()
        return SimpleNamespace(completion_text="ok", usage=None)


class Provider:
    def __init__(self):
        self.client = Client([])
        self.api_keys = ["key-one", "key-two"]

    async def text_chat(self, *, prompt, system_prompt, max_tokens, request_max_retries=None):
        assert system_prompt == "system" and max_tokens == 3000
        for key in self.api_keys:
            self.client.api_key = key
            for _ in range(request_max_retries or 5):
                try:
                    return await self.client.request(prompt)
                except RateLimitError:
                    pass
        raise RateLimitError()


def context_for(provider):
    def get_provider_by_id(pid):
        assert pid == "selected"
        return provider

    return SimpleNamespace(get_provider_by_id=get_provider_by_id)


async def test_429_disables_framework_sdk_and_key_rotation_retries():
    provider = Provider()
    original_client = provider.client
    original_keys = provider.api_keys
    with pytest.raises(RateLimitError):
        await generate_once(context_for(provider), "selected", "system", "429")
    assert len(original_client.calls) == 1
    assert original_client.calls[0][1] == 0
    assert provider.client is original_client and original_client.max_retries == 2
    assert original_client.api_key == "original-key"
    assert provider.api_keys is original_keys and len(original_keys) == 2


async def test_concurrent_chat_keeps_its_original_sdk_retry_policy():
    provider = Provider()
    result, normal = await asyncio.gather(
        generate_once(context_for(provider), "selected", "system", "memory"),
        provider.text_chat(prompt="normal", system_prompt="system", max_tokens=3000),
    )
    assert result.completion_text == normal.completion_text == "ok"
    assert [r[1] for r in provider.client.calls if r[0] == "memory"] == [0]
    assert [r[1] for r in provider.client.calls if r[0] == "normal"] == [2, 2, 2]


async def test_async_provider_lookup_and_cancellation_preserve_shared_client():
    provider = Provider()
    entered = asyncio.Event()

    async def blocked(self, prompt):
        entered.set()
        await asyncio.Event().wait()

    async def lookup(pid):
        return provider

    original = Client.request
    Client.request = blocked
    try:
        task = asyncio.create_task(
            generate_once(SimpleNamespace(get_provider_by_id=lookup), "selected", "system", "x")
        )
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.client.max_retries == 2 and provider.client.api_key == "original-key"
        assert provider.api_keys == ["key-one", "key-two"]
    finally:
        Client.request = original


@pytest.mark.parametrize("supported", [True, False])
async def test_other_providers_do_not_receive_unsupported_request_options(supported):
    calls = []

    async def modern(*, request_max_retries=None, **kwargs):
        calls.append((request_max_retries, kwargs))
        return "modern"

    async def custom(**kwargs):
        assert "request_max_retries" not in kwargs
        calls.append((None, kwargs))
        return "custom"

    provider = SimpleNamespace(text_chat=modern if supported else custom)
    result = await generate_once(context_for(provider), "selected", "system", "prompt")
    assert result == ("modern" if supported else "custom")
    assert calls == [
        (
            1 if supported else None,
            {"prompt": "prompt", "system_prompt": "system", "max_tokens": 3000},
        )
    ]


async def test_missing_provider_fails_without_falling_back_to_default():
    with pytest.raises(AuxiliaryModelFailure, match="Provider 不可用"):
        await generate_once(context_for(None), "selected", "system", "prompt")
