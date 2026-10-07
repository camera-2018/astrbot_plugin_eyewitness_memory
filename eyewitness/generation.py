"""Request-local retry controls; never mutate AstrBot's shared provider/client."""

import inspect
import random
from copy import copy

from .errors import AuxiliaryModelFailure


async def generate_once(context, provider_id: str, system: str, prompt: str):
    provider = context.get_provider_by_id(provider_id)
    if inspect.isawaitable(provider):
        provider = await provider
    if provider is None or not callable(getattr(provider, "text_chat", None)):
        raise AuxiliaryModelFailure("辅助模型 Provider 不可用")

    options = {"prompt": prompt, "system_prompt": system, "max_tokens": 3000}
    # Do not leak a framework-only option into older/custom providers' API bodies.
    if "request_max_retries" in inspect.signature(provider.text_chat).parameters:
        options["request_max_retries"] = 1

    client = getattr(provider, "client", None)
    if hasattr(client, "max_retries") and callable(getattr(client, "with_options", None)):
        # OpenAI-compatible (including Azure) and Anthropic SDKs copy their
        # configuration while sharing the transport. Do not close this copy:
        # AstrBot still owns the shared HTTP connection pool and its lifecycle.
        isolated = copy(provider)
        if isolated is provider:
            raise AuxiliaryModelFailure("辅助模型无法隔离重试配置")
        isolated.client = client.with_options(max_retries=0)
        keys = getattr(provider, "api_keys", None)
        if isinstance(keys, list) and keys:
            # OpenAI's outer recovery loop otherwise retries 429 with other keys.
            isolated.api_keys = [random.choice(keys)]
        provider = isolated

    return await provider.text_chat(**options)
