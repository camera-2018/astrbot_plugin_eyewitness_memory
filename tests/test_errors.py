import sqlite3

import pytest
from pydantic import ValidationError

from eyewitness.errors import failure_detail
from eyewitness.models import EvidenceRef, Review, Verification


def test_diagnostic_classifies_invalid_json_without_echoing_it():
    with pytest.raises(ValidationError) as caught:
        Review.model_validate_json("not-secret-json")
    assert failure_detail(caught.value) == "模型返回的 JSON 无法解析"


def test_diagnostic_classifies_missing_field():
    with pytest.raises(ValidationError) as caught:
        Verification.model_validate_json("{}")
    assert failure_detail(caught.value) == "模型返回的 JSON 缺少必需字段"


def test_single_character_evidence_is_valid():
    assert EvidenceRef(message_id="message-1", quote="X").quote == "X"


def test_diagnostic_includes_invalid_field_location():
    with pytest.raises(ValidationError) as caught:
        EvidenceRef.model_validate({"message_id": "message-1", "quote": 1})
    assert "模型返回字段 quote 校验失败" in failure_detail(caught.value)


def test_diagnostic_classifies_sqlite_error_without_query():
    db = sqlite3.connect(":memory:")
    with pytest.raises(sqlite3.Error) as caught:
        db.execute("SELECT secret FROM missing_table")
    assert failure_detail(caught.value) == "SQLite SQLITE_ERROR"
    db.close()


@pytest.mark.parametrize(
    "body,category",
    [
        ({"error": {"code": "insufficient_quota", "message": "sk-private-credential"}}, "额度"),
        (
            {"error": {"message": "Individual quota reached. Please upgrade your subscription"}},
            "额度",
        ),
        ({"error": {"type": "rate_limit_exceeded"}}, "速率"),
        ({"error": {"code": "model_cooldown"}}, "冷却"),
        ({"error": {"message": "sk-private-credential arbitrary-private-message"}}, "HTTP 429"),
    ],
)
def test_http_error_classification_does_not_echo_credentials_or_response_text(body, category):
    class Error(Exception):
        status_code = 429

    exc = Error("raw-private-exception")
    exc.body = body
    detail = failure_detail(exc)
    assert category in detail and "private" not in detail and "sk-" not in detail
