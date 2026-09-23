"""Conservative request-time memory budget using AstrBot's token estimator."""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from typing import Callable

from .context import render_memory


@dataclass(frozen=True)
class InjectionBudget:
    context_limit: int
    estimated_input: int
    reserved: int
    available: int
    reason: str = ""


def plan_budget(context_limit: int, estimated_input: int, tool_tokens: int = 0) -> InjectionBudget:
    """Leave 20% model-window slack plus an output allowance before adding memory."""
    output_reserve = min(4096, max(512, context_limit // 10))
    usable = max(0, int(context_limit * 0.8) - output_reserve)
    occupied = estimated_input + tool_tokens
    available = max(0, usable - occupied)
    return InjectionBudget(
        context_limit=context_limit,
        estimated_input=occupied,
        reserved=context_limit - usable,
        available=available,
        reason="" if available else "本轮输入和工具已占满保守上下文预算",
    )


def unavailable(reason: str) -> InjectionBudget:
    return InjectionBudget(0, 0, 0, 0, reason)


async def request_budget(context, event, req) -> tuple[InjectionBudget, Callable[[str], int]]:
    """Estimate a fresh request; never reuse the previous response's token_usage."""
    from astrbot.core.agent.context.token_counter import EstimateTokenCounter
    from astrbot.core.agent.message import (
        AudioURLPart,
        ImageURLPart,
        Message,
        TextPart,
        ThinkPart,
    )

    counter = EstimateTokenCounter()

    def text_cost(value: str) -> int:
        native = counter.count_tokens([Message(role="user", content=value)])
        return max(math.ceil(native * 1.5), len(value))

    if req.image_urls or req.audio_urls or req.tool_calls_result:
        return unavailable("本轮含媒体或工具结果，无法安全估算"), text_cost
    for part in req.extra_user_content_parts or []:
        if not isinstance(part, TextPart) and not (
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ):
            return unavailable("本轮附加内容含非文本，无法安全估算"), text_cost

    try:
        selected_id = event.get_extra("selected_provider") if hasattr(event, "get_extra") else None
        if selected_id:
            provider = context.get_provider_by_id(selected_id)
            if inspect.isawaitable(provider):
                provider = await provider
        else:
            provider = await context.get_using_provider_async(event.unified_msg_origin)
        config = getattr(provider, "provider_config", {}) if provider else {}
        limit = config.get("max_context_tokens", 0)
    except Exception:
        return unavailable("无法读取当前模型的上下文上限"), text_cost
    if type(limit) is not int or not 4096 <= limit <= 2_000_000:
        return unavailable("模型上下文上限未知"), text_cost

    try:
        messages = []
        if req.system_prompt:
            messages.append(Message(role="system", content=req.system_prompt))
        for raw in req.contexts or []:
            role = raw.get("role") if isinstance(raw, dict) else getattr(raw, "role", None)
            if role == "_checkpoint":
                continue
            messages.append(Message.model_validate(raw))
        content = []
        if req.prompt:
            content.append(TextPart(text=req.prompt))
        for part in req.extra_user_content_parts or []:
            content.append(TextPart(text=part["text"]) if isinstance(part, dict) else part)
        if content:
            messages.append(Message(role="user", content=content))

        native_tokens = counter.count_tokens(messages)
        text_chars = 0
        media_tokens = 0
        for message in messages:
            parts = message.content
            if isinstance(parts, str):
                text_chars += len(parts)
            elif isinstance(parts, list):
                for part in parts:
                    if isinstance(part, TextPart):
                        text_chars += len(part.text)
                    elif isinstance(part, ThinkPart):
                        text_chars += len(part.think)
                    elif isinstance(part, ImageURLPart):
                        media_tokens += 2300
                    elif isinstance(part, AudioURLPart):
                        media_tokens += 1500
                    else:
                        return unavailable("历史含未知内容类型，无法安全估算"), text_cost
            if message.tool_calls:
                text_chars += len(
                    json.dumps(
                        [
                            tc if isinstance(tc, dict) else tc.model_dump()
                            for tc in message.tool_calls
                        ],
                        ensure_ascii=False,
                    )
                )
        estimated_input = max(math.ceil(native_tokens * 1.5), text_chars + media_tokens)
        tool_tokens = 0
        if req.func_tool and not req.func_tool.empty():
            tools_json = json.dumps(req.func_tool.openai_schema(), ensure_ascii=False)
            tool_tokens = text_cost(tools_json)
        return plan_budget(limit, estimated_input, tool_tokens), text_cost
    except Exception:
        return unavailable("请求结构无法安全估算"), text_cost


def fit_memory_block(
    item: dict,
    char_budget: int,
    token_budget: int | None,
    existing_text: str,
    count_tokens: Callable[[str], int] | None,
    seen_ids: set[str],
) -> tuple[str, list[str], str]:
    """Keep all required source lines; discard neighbors before omitting a memory."""
    block, ids = render_memory(item, char_budget, seen_ids)
    if not block:
        return "", [], "chars"
    if token_budget is None:
        return block, ids, ""
    if count_tokens is None:
        return "", [], "tokens"

    separator = "\n" if existing_text and not existing_text.endswith("\n") else ""
    if count_tokens(existing_text + separator + block) <= token_budget:
        return block, ids, ""

    # render_memory never clips an anchor. Search only the optional-neighbor budget.
    low, high = 0, char_budget - 1
    best: tuple[str, list[str]] | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate, candidate_ids = render_memory(item, middle, seen_ids)
        if not candidate:
            low = middle + 1
        elif count_tokens(existing_text + separator + candidate) <= token_budget:
            best = candidate, candidate_ids
            low = middle + 1
        else:
            high = middle - 1
    if best:
        return best[0], best[1], ""
    return "", [], "tokens"
