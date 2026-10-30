from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["off", "active"] = "off"
    allowed_scopes: list[str] = Field(default_factory=list, max_length=100)
    scope_aliases: list[str] = Field(default_factory=list, max_length=100)
    bot_ids: list[str] = Field(default_factory=list, max_length=500)
    provider_id: str = ""
    json_mode: bool = False
    embedding_provider_id: str = ""
    systemone_provider_id: str = ""
    systemone_model: str = ""
    systemone_api_path: str = "/typesafe/v1/systemone"
    systemone_timeout: float = Field(default=3.0, ge=0.2, le=10)
    systemone_check_support: bool = True
    recall_gate_enabled: bool = False
    recall_gate_timeout: float = Field(default=1.5, ge=0.2, le=3)
    recall_gate_skip_probability: float = Field(default=0.9, ge=0.9, le=1.0)
    qdrant_url: str = ""
    collection: str = "eyewitness_memory_v1"
    online_timeout: float = Field(default=30.0, ge=0.5, le=30)
    understanding_timeout: float = Field(default=8.0, ge=0.5, le=15)
    query_expansion_limit: int = Field(default=3, ge=0, le=4)
    candidate_pool_size: int = Field(default=24, ge=3, le=48)
    source_review_limit: int = Field(default=6, ge=1, le=12)
    extraction_timeout: float = Field(default=60.0, ge=5, le=180)
    batch_size: int = Field(default=80, ge=5, le=100)
    batch_age_seconds: int = Field(default=900, ge=60, le=86400)
    retention_days: int = Field(default=30, ge=1, le=365)
    trace_days: int = Field(default=7, ge=1, le=90)
    max_db_mb: int = Field(default=512, ge=16, le=4096)
    injection_chars: int = Field(default=8000, ge=200, le=16000)
    semantic_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    maintenance_enabled: bool = False
    maintenance_interval_hours: int = Field(default=24, ge=1, le=720)
    maintenance_semantic_enabled: bool = False
    maintenance_review_limit: int = Field(default=32, ge=0, le=128)
    maintenance_timeout: float = Field(default=45.0, ge=5, le=120)

    @field_validator("scope_aliases")
    @classmethod
    def valid_scope_aliases(cls, values: list[str]) -> list[str]:
        from .scope_aliases import parse_scope_aliases

        aliases = parse_scope_aliases(values)
        return [f"{source}={target}" for source, target in aliases.items()]

    @model_validator(mode="after")
    def canonical_allowlist(self):
        from .scope_aliases import parse_scope_aliases

        if set(parse_scope_aliases(self.scope_aliases)) & set(self.allowed_scopes):
            raise ValueError("白名单请填写映射后的统一群，不能同时列入映射来源群")
        return self

    @field_validator("systemone_model")
    @classmethod
    def valid_systemone_model(cls, value: str) -> str:
        if value and not re.fullmatch(r"[A-Za-z0-9_@/.-]{1,128}", value):
            raise ValueError("System One 模型名只能包含字母、数字、_ @ / . -")
        return value

    @field_validator("systemone_api_path")
    @classmethod
    def valid_systemone_path(cls, value: str) -> str:
        if not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*/v1/systemone", value):
            raise ValueError("System One 路径必须为相对路径，如 /typesafe/v1/systemone")
        return value

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
        if not re.fullmatch(r"eyewitness_memory_[a-zA-Z0-9_-]+", value):
            raise ValueError("collection 必须以 eyewitness_memory_ 开头，不能复用其他插件索引")
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
    # A one-character message can still be the exact source of a memory.
    # Keep the non-empty guard, but do not reject valid evidence such as "X".
    quote: str = Field(min_length=1, max_length=1000)


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


class MaintenanceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    action: Literal["keep", "forget", "uncertain"]
    reason: str = Field(min_length=1, max_length=300)
    category: Literal["useful", "low_value", "superseded", "duplicate", "uncertain"]
    related_id: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=5)


class MaintenanceReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decisions: list[MaintenanceDecision] = Field(default_factory=list, max_length=16)


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


class GroundedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    action: Literal["accept", "reject"]
    reason: str = Field(max_length=300)
    text: str = Field(default="", max_length=600)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=5)


class GroundedReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decisions: list[GroundedDecision] = Field(default_factory=list, max_length=3)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    queries: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(max_length=4)
    reason: str = Field(max_length=400)


class ShortlistItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    reason: str = Field(max_length=300)


class CandidateShortlist(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selected: list[ShortlistItem] = Field(max_length=12)


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
    """Only empty/non-text content; never classify the meaning of an utterance."""
    if has_reference:
        return False
    stripped = re.sub(r"\[[A-Za-z]+\]|[\s，。！？!?~～（）()]+", "", text)
    return not any(char.isalnum() for char in stripped)


EXTRACTION_MARKUP = re.compile(
    r"\[CQ:[^\]]+\]|\[At:[^\]]+\]|\[[A-Za-z]+\]|"
    r"@[^\s@()]{1,40}\(\d+\)|https?://\S+"
)


def extraction_noise(text: str, has_reference: bool = False) -> bool:
    """Skip only contentless transport markup, never a word/intent category."""
    visible = EXTRACTION_MARKUP.sub("", text).strip()
    return not any(char.isalnum() for char in visible)
