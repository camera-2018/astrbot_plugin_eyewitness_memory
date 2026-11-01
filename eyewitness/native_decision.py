"""AstrBot 4.29.4 DecisionService bridge; no independent HTTP or credentials.

Keep this file identical in both plugins. The shared admission queue sits before
DecisionService.ask, so the official provider never sees lower-priority backlog.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

from .fastbrain_queue import priority_slot


class NativeDecisionFailure(Exception):
    def __init__(self, reason: str, retry_after: float = 60):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


def _failure(exc):
    # Official adapters wrap HTTP failures. Retain only status and retry delay,
    # never provider exception text, which may contain response data or URLs.
    cause = exc.__cause__
    response = getattr(cause, "response", None)
    if response is not None:
        delay = 60.0
        value = response.headers.get("Retry-After")
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                delay = max(delay, seconds)
        except (TypeError, ValueError):
            try:
                delay = max(delay, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
        return NativeDecisionFailure(f"http_{response.status_code}", delay)
    return NativeDecisionFailure("native_decision_failed")


@dataclass(frozen=True)
class NativeReply:
    model: str
    answers: dict
    queue_ms: int
    request_ms: int
    priority: str


class NativeDecisionClient:
    def __init__(self, service, resolve_provider):
        self.service = service
        self.resolve_provider = resolve_provider

    async def ask(self, provider_id, body, *, timeout, urgent=False, priority_enabled=False):
        # Lazy import keeps explicitly configured legacy HTTP mode usable on old SDKs.
        from astrbot.core.provider.entities import DecisionChoiceAnswer, DecisionChoiceQuestion

        if self.service is None or self.resolve_provider is None:
            raise NativeDecisionFailure("native_decision_unavailable")
        provider = self.resolve_provider(provider_id)
        if inspect.isawaitable(provider):
            provider = await provider
        if provider is None or provider.provider_config.get("type") != "system_one_decision":
            raise NativeDecisionFailure("not_system_one_provider")
        model = provider.get_model()
        if model != body.get("model"):
            raise NativeDecisionFailure("configured_model_mismatch")
        if len(json.dumps(body, ensure_ascii=False).encode()) > 48000:
            raise NativeDecisionFailure("request_too_large")
        questions = {}
        for head, question in body["questions"].items():
            if question.get("type") != "choice":
                raise NativeDecisionFailure("unsupported_question_type")
            # Preserve the original option IDs, descriptions, and instructions.
            # yes_no()/choose() would change their wire representation.
            questions[head] = DecisionChoiceQuestion(
                instructions=question["instructions"], criteria=question["criteria"]
            )
        async with priority_slot(
            provider.base_url, model, urgent=urgent, enabled=priority_enabled
        ) as ticket:
            started = time.monotonic()
            try:
                async with asyncio.timeout(timeout):
                    answers = await self.service.ask(
                        questions, body.get("state", {}), provider_id=provider_id
                    )
            except TimeoutError:
                raise NativeDecisionFailure("timeout") from None
            except Exception as exc:
                raise _failure(exc) from None
            if set(answers) != set(questions) or any(
                not isinstance(answer, DecisionChoiceAnswer) for answer in answers.values()
            ):
                raise NativeDecisionFailure("invalid_answer_type")
            normalized = {
                head: {
                    "type": "choice",
                    "choice": answer.choice,
                    "probabilities": dict(answer.probabilities),
                    "confidence": answer.confidence,
                }
                for head, answer in answers.items()
            }
            return NativeReply(
                model,
                normalized,
                ticket.wait_ms,
                round((time.monotonic() - started) * 1000),
                ticket.priority,
            )
