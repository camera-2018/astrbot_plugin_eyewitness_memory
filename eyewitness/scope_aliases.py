"""Explicit group equivalence and collision-free, reversible transport keys.

These aliases apply only to this plugin's memory namespace. They never rewrite
AstrBot events, platform group IDs, sender IDs, or outbound message destinations.
"""

from __future__ import annotations

import base64
import json
import re

GROUP_SCOPE = re.compile(r"[^:\s=]+:GroupMessage:[^:\s=]+")
MESSAGE_KEY_PREFIX = "~eyewitness-origin-v1:"


def parse_scope_aliases(entries: list[str]) -> dict[str, str]:
    """Validate a one-hop list of `source_umo=canonical_umo` declarations."""
    if not isinstance(entries, list) or len(entries) > 100:
        raise ValueError("群映射必须是最多100项的列表")
    result = {}
    for entry in entries:
        if not isinstance(entry, str) or entry.count("=") != 1:
            raise ValueError("群映射格式为 来源群UMO=统一群UMO")
        source, target = (part.strip() for part in entry.split("=", 1))
        if not GROUP_SCOPE.fullmatch(source) or not GROUP_SCOPE.fullmatch(target):
            raise ValueError("群映射只接受完整 GroupMessage 会话标识")
        if source == target:
            raise ValueError("群映射的来源和目标不能相同")
        if source in result and result[source] != target:
            raise ValueError("同一个来源群不能映射到多个目标")
        result[source] = target
    if set(result) & set(result.values()):
        raise ValueError("群映射必须直接指向最终群，不允许多跳或循环")
    return result


def normalize_scope_aliases(entries: list[str]) -> list[str]:
    return [f"{source}={target}" for source, target in parse_scope_aliases(entries).items()]


def canonical_scope(scope: str, entries: list[str]) -> str:
    return parse_scope_aliases(entries).get(scope, scope)


def platform_message_key(origin: str, canonical: str, platform_id: str) -> str:
    """Keep legacy IDs, namespace aliased IDs, and escape the reserved prefix.

    UNIQUE(scope, platform_id) remains safe without rebuilding the messages
    table. The actual origin and platform ID are retained verbatim in the key.
    """
    if not isinstance(platform_id, str) or not platform_id:
        raise ValueError("消息ID不能为空")
    if origin == canonical and not platform_id.startswith(MESSAGE_KEY_PREFIX):
        return platform_id
    if not GROUP_SCOPE.fullmatch(origin) or not GROUP_SCOPE.fullmatch(canonical):
        raise ValueError("消息引用只接受群会话标识")
    data = json.dumps([origin, platform_id], ensure_ascii=False, separators=(",", ":")).encode()
    return MESSAGE_KEY_PREFIX + base64.urlsafe_b64encode(data).decode().rstrip("=")


def decode_message_key(key: str | None, default_scope: str = "") -> tuple[str, str | None]:
    """Decode only canonical, well-formed internal keys; retain other IDs intact."""
    if not isinstance(key, str) or not key.startswith(MESSAGE_KEY_PREFIX):
        return default_scope, key
    token = key[len(MESSAGE_KEY_PREFIX) :]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        return default_scope, key
    try:
        payload = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
        values = json.loads(payload)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or not isinstance(values[0], str)
            or not GROUP_SCOPE.fullmatch(values[0])
            or not isinstance(values[1], str)
            or not values[1]
        ):
            return default_scope, key
        encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
        if base64.urlsafe_b64encode(encoded).decode().rstrip("=") != token:
            return default_scope, key
        return values[0], values[1]
    except (ValueError, UnicodeError):
        return default_scope, key
