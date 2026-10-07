"""Bounded diagnostics that never copy model output or provider exception text."""

from __future__ import annotations

import json
import sqlite3

from pydantic import ValidationError


class AuxiliaryModelFailure(Exception):
    """A safe-to-display explanation of one auxiliary model failure."""


class RetryConflict(ValueError):
    """A manual retry refers to an outdated or unavailable failed batch."""


def upstream_category(exc: Exception) -> str:
    """Classify only known structured fields; never display provider response text."""
    fragments = []

    def collect(value, depth=0):
        if depth > 4:
            return
        if isinstance(value, dict):
            for key, item in list(value.items())[:20]:
                if key in {"message", "code", "type", "status"} and isinstance(item, str):
                    fragments.append(item[:2000].lower())
                elif isinstance(item, (dict, list)):
                    collect(item, depth + 1)
        elif isinstance(value, list):
            for item in value[:6]:
                collect(item, depth + 1)

    collect(getattr(exc, "body", None))
    text = " ".join(fragments)
    if any(
        word in text
        for word in (
            "insufficient_quota",
            "quota reached",
            "quota exhausted",
            "credit",
            "balance",
            "billing",
            "daily limit",
            "subscription limit",
            "额度",
            "余额",
            "配额耗尽",
        )
    ):
        return "额度不足或已耗尽"
    if any(
        word in text
        for word in (
            "rate_limit",
            "rate limit",
            "too many requests",
            "per minute",
            "rpm",
            "tpm",
            "限流",
            "频率",
        )
    ):
        return "请求频率或 Token 速率受限"
    if any(
        word in text
        for word in (
            "model_cooldown",
            "cooling down",
            "no_available_channels",
            "all channels unavailable",
            "冷却",
        )
    ):
        return "模型或渠道冷却中，暂不可用"
    return ""


def failure_detail(exc: Exception) -> str:
    if isinstance(exc, AuxiliaryModelFailure):
        return str(exc)
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_input=False)
        codes = {error["type"] for error in errors}
        if "json_invalid" in codes:
            return "模型返回的 JSON 无法解析"
        if "missing" in codes:
            return "模型返回的 JSON 缺少必需字段"
        if "extra_forbidden" in codes:
            return "模型返回的 JSON 包含多余字段"
        first = errors[0] if errors else {}
        location = ".".join(str(part) for part in first.get("loc", ())) or "根对象"
        error_type = str(first.get("type") or "unknown")
        return f"模型返回字段 {location} 校验失败（{error_type}）"
    if isinstance(exc, TimeoutError):
        return "请求超时"
    if isinstance(exc, json.JSONDecodeError):
        return "上游响应不是有效 JSON"
    for name in ("status_code", "status"):
        status = getattr(exc, name, None)
        if type(status) is int and 400 <= status <= 599:
            category = upstream_category(exc)
            return f"上游 HTTP {status}" + ("：" + category if category else "")
    if type(exc).__name__ == "EmptyModelOutputError":
        return "模型返回空内容（EmptyModelOutputError）"
    if isinstance(exc, sqlite3.Error):
        return "SQLite " + str(getattr(exc, "sqlite_errorname", type(exc).__name__))[:48]
    if isinstance(exc, ValueError):
        known = {
            "Embedding provider 不可用": "Embedding Provider 未配置或不可用",
            "Invalid embedding dimensions": "Embedding 向量维度无效",
        }
        if str(exc) in known:
            return known[str(exc)]
    if isinstance(exc, ConnectionError):
        return "连接失败"
    if isinstance(exc, OSError) and exc.errno is not None:
        return f"I/O errno {exc.errno}"
    return "异常类型 " + type(exc).__name__[:64]
