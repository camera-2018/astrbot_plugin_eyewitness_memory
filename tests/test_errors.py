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
