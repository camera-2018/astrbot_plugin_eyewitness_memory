import sqlite3

import pytest
from pydantic import ValidationError

from eyewitness.errors import failure_detail
from eyewitness.models import Review, Verification


def test_diagnostic_classifies_invalid_json_without_echoing_it():
    with pytest.raises(ValidationError) as caught:
        Review.model_validate_json("not-secret-json")
    assert failure_detail(caught.value) == "模型返回的 JSON 无法解析"


def test_diagnostic_classifies_missing_field():
    with pytest.raises(ValidationError) as caught:
        Verification.model_validate_json("{}")
    assert failure_detail(caught.value) == "模型返回的 JSON 缺少必需字段"


def test_diagnostic_classifies_sqlite_error_without_query():
    db = sqlite3.connect(":memory:")
    with pytest.raises(sqlite3.Error) as caught:
        db.execute("SELECT secret FROM missing_table")
    assert failure_detail(caught.value) == "SQLite SQLITE_ERROR"
    db.close()
