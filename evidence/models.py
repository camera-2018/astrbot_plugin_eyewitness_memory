from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["off", "active"] = "off"
    allowed_scopes: list[str] = Field(default_factory=list, max_length=100)
    bot_ids: list[str] = Field(default_factory=list, max_length=500)
    provider_id: str = ""
    embedding_provider_id: str = ""
    qdrant_url: str = ""
    collection: str = "evidence_memory_v1"
    online_timeout: float = Field(default=3.0, ge=0.5, le=30)
    extraction_timeout: float = Field(default=60.0, ge=5, le=180)
    batch_size: int = Field(default=40, ge=5, le=100)
    batch_age_seconds: int = Field(default=900, ge=60, le=86400)
    daily_calls: int = Field(default=200, ge=0, le=10000)
    retention_days: int = Field(default=30, ge=1, le=365)
    trace_days: int = Field(default=7, ge=1, le=90)
    max_db_mb: int = Field(default=512, ge=16, le=4096)
    injection_chars: int = Field(default=8000, ge=200, le=16000)

    @field_validator("allowed_scopes")
    @classmethod
    def valid_scopes(cls, values: list[str]) -> list[str]:
        for v in values:
            if not re.fullmatch(r"[^:\s]+:GroupMessage:[^:\s]+", v):
                raise ValueError("必须使用完整群作用域，如 default:GroupMessage:123456")
        return list(dict.fromkeys(values))

    @field_validator("collection")
    @classmethod
    def owned_collection(cls, value: str) -> str:
        if not re.fullmatch(r"evidence_memory_[a-zA-Z0-9_-]+", value):
            raise ValueError("collection 必须以 evidence_memory_ 开头，不能复用其他插件索引")
        return value

    @field_validator("qdrant_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        if value:
            u = urlsplit(value)
            if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password:
                raise ValueError("Qdrant 地址必须为无内嵌凭据的 HTTP(S) URL")
            if u.query or u.fragment:
                raise ValueError("Qdrant 地址不能包含 query 或 fragment")
        return value.rstrip("/")


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_id: str
    quote: str = Field(min_length=2, max_length=1000)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    summary: str = Field(min_length=4, max_length=600)
    kind: Literal["preference", "goal", "event", "agreement"]
    subject_id: str
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=5)
    stance: Literal["self_report", "hearsay", "joke", "uncertain"]
    expires_in_days: int = Field(default=30, ge=1, le=365)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    candidates: list[Candidate] = Field(default_factory=list, max_length=8)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    action: Literal["accept", "reject", "needs_source"]
    reason: str = Field(max_length=300)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decisions: list[Decision] = Field(default_factory=list, max_length=12)


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    supported: bool
    text: str = Field(max_length=600)
    reason: str = Field(max_length=300)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=5)


def terms(text: str) -> list[str]:
    """Explicit Chinese bigrams + Latin words, never SQL/FTS syntax from users."""
    found: list[str] = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_]{2,}", text.lower()[:2000]):
        found.extend(
            [part]
            if part.isascii() or len(part) < 2
            else [part[i : i + 2] for i in range(len(part) - 1)]
        )
    return list(dict.fromkeys(found))[:80]


SECRET = re.compile(
    r"(?:sk-[a-zA-Z0-9_-]{12,}|(?i:SESSDATA|bili_jct|authorization|api[_-]?key|password|密码)"
    r"\s*[=:：]\s*[^\s;]+)"
)


def safe_text(text: str, limit: int = 2000) -> str:
    return SECRET.sub("[已脱敏]", text)[:limit]


def low_signal(text: str, has_reference: bool = False) -> bool:
    if has_reference:
        return False
    stripped = re.sub(r"\[[A-Za-z]+\]|[\s，。！？!?~～（）()]+", "", text)
    return not stripped or stripped in {
        "哈哈",
        "哈哈哈",
        "笑死",
        "确实",
        "好",
        "嗯",
        "哦",
        "牛逼",
        "草",
        "嘻嘻",
        "复读",
        "收到",
        "笑了",
    }
