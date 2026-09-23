"""Bounded diagnostics that never copy model output or provider exception text."""

from __future__ import annotations

import json
import sqlite3

from pydantic import ValidationError


class AuxiliaryModelFailure(Exception):
    """A safe-to-display explanation of one auxiliary model failure."""


def failure_detail(exc: Exception) -> str:
    if isinstance(exc, AuxiliaryModelFailure):
        return str(exc)
    if isinstance(exc, ValidationError):
        codes = {error["type"] for error in exc.errors(include_input=False)}
        if "json_invalid" in codes:
            return "模型返回的 JSON 无法解析"
        if "missing" in codes:
            return "模型返回的 JSON 缺少必需字段"
        if "extra_forbidden" in codes:
            return "模型返回的 JSON 包含多余字段"
        return "模型返回的 JSON 字段类型或取值不符"
    if isinstance(exc, TimeoutError):
        return "请求超时"
    if isinstance(exc, json.JSONDecodeError):
        return "上游响应不是有效 JSON"
    for name in ("status_code", "status"):
        status = getattr(exc, name, None)
        if type(status) is int and 400 <= status <= 599:
            return f"上游 HTTP {status}"
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
