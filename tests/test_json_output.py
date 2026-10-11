import json

import pytest
from pydantic import ValidationError

from eyewitness.engine import Engine
from eyewitness.errors import AuxiliaryModelFailure
from eyewitness.json_output import parse_output
from eyewitness.models import Extraction, GroundedReview, Settings
from eyewitness.usage import ModelReply
from tests.conftest import SCOPE, seed
from tests.test_engine import Model, answers
from tests.test_extraction_failures import pending


def review():
    return {
        "decisions": [
            {
                "id": "m1",
                "action": "accept",
                "reason": "本人原文支持",
                "text": "参加绘画比赛",
                "evidence": [{"message_id": "s1", "quote": "我报名了绘画比赛"}],
            }
        ],
    }


def test_extra_metadata_is_ignored_at_root_decision_and_evidence_only():
    raw = review()
    raw["metadata"] = {"secret": "must-not-survive"}
    raw["decisions"][0]["confidence"] = 0.99
    raw["decisions"][0]["evidence"][0]["sender_name"] = "untrusted"
    parsed, ignored = parse_output(GroundedReview, json.dumps(raw))
    assert parsed.model_dump() == GroundedReview.model_validate(review()).model_dump()
    assert ignored == 3
    assert "must-not-survive" not in parsed.model_dump_json()
    assert "confidence" not in parsed.model_dump_json()
    assert GroundedReview.model_json_schema()["additionalProperties"] is False


def test_valid_reply_uses_original_strict_validation():
    parsed, ignored = parse_output(GroundedReview, json.dumps(review()))
    assert len(parsed.decisions) == 1 and ignored == 0


@pytest.mark.parametrize("raw", ["not-json", '{"private":true}', "[]", '"string"'])
def test_invalid_or_unrelated_payload_does_not_become_an_empty_success(raw):
    with pytest.raises(ValidationError):
        parse_output(Extraction, raw)


@pytest.mark.parametrize("failure", ["missing", "enum", "type", "length", "evidence", "count"])
def test_extra_fields_do_not_hide_protocol_errors(failure):
    raw = review()
    raw["note"] = "extra-but-not-a-repair"
    decision = raw["decisions"][0]
    if failure == "missing":
        del decision["id"]
    elif failure == "enum":
        decision["action"] = "maybe"
    elif failure == "type":
        decision["text"] = 10
    elif failure == "length":
        decision["text"] = "长" * 601
    elif failure == "evidence":
        decision["evidence"][0]["quote"] = ""
    else:
        raw["decisions"] *= 4
    with pytest.raises(ValidationError):
        parse_output(GroundedReview, json.dumps(raw))


def test_plugin_configuration_stays_strict():
    with pytest.raises(ValidationError):
        Settings.model_validate({"unknown_option": True})


async def test_compatible_reply_is_one_successful_call_with_real_usage_and_safe_log(store, caplog):
    async def generate(*args):
        return ModelReply('{"candidates":[],"note":"private-provider-body"}', 100, 9)

    caplog.set_level("INFO", logger="eyewitness.engine")
    parsed = await Engine(store, generate).ask(
        await store.call("get_settings"), "test", {}, Extraction
    )
    assert parsed.candidates == []
    row = store.db.execute("SELECT * FROM model_calls").fetchone()
    assert row["status"] == "success" and row["input_tokens"] == 100 and row["output_tokens"] == 9
    assert row["error"] == ""
    assert "兼容忽略 1 个" in caplog.text and "private-provider-body" not in caplog.text
    assert store.db.execute("SELECT count(*) FROM model_calls").fetchone()[0] == 1


async def test_extraction_extra_fields_do_not_fail_or_requeue_the_batch(store):
    source = await pending(store)
    candidate = {
        "summary": "alice 计划参加绘画比赛",
        "kind": "goal",
        "subject_id": "alice",
        "stance": "self_report",
        "confidence": 0.9,
        "evidence": [{"message_id": source, "quote": "我计划下个月参加绘画比赛", "time": "today"}],
    }
    model = Model([{"candidates": [candidate], "reason": "metadata"}])
    engine = Engine(store, model)
    cfg = await store.call("get_settings")
    assert await engine.extract_once(cfg)
    assert not await engine.extract_once(cfg)
    row = (await store.call("source_rows", SCOPE, [source]))[0]
    assert row["processed"] == 1 and row["extraction_error"] == ""
    assert model.calls == 1 and (await store.call("stats"))["active"] == 1


@pytest.mark.parametrize("fabrication", [None, "quote", "id", "neighbor"])
async def test_extra_fields_cannot_bypass_grounded_source_checks(store, fabrication):
    mid, source = await seed(store)
    payload = answers(mid, source)
    payload[0]["note"] = "compatible"
    decision = payload[0]["decisions"][0]
    decision["confidence"] = 1.0
    evidence = decision["evidence"][0]
    evidence["supported"] = True
    if fabrication == "quote":
        evidence["quote"] = "不存在的原文"
    elif fabrication == "id":
        evidence["message_id"] = "not-a-source"
    elif fabrication == "neighbor":
        neighbor = await store.call("capture", SCOPE, "neighbor", "bob", "Bob", "我报名了游泳比赛")
        evidence.update(message_id=neighbor, quote="我报名了游泳比赛")
    model = Model(payload)
    result = await Engine(store, model).recall(SCOPE, "绘画比赛")
    assert bool(result["injection"]) == (fabrication is None)
    assert model.calls == 1
    assert store.db.execute("SELECT status FROM model_calls").fetchone()[0] == "success"


async def test_invalid_protocol_with_extra_fields_is_still_failed_without_retry(store):
    source = await pending(store)
    model = Model([{"candidates": [{"confidence": 1.0}], "note": "extra"}])
    engine = Engine(store, model)
    cfg = await store.call("get_settings")
    with pytest.raises(AuxiliaryModelFailure, match="缺少必需字段"):
        await engine.extract_once(cfg)
    assert not await engine.extract_once(cfg)
    assert model.calls == 1
    assert (await store.call("source_rows", SCOPE, [source]))[0]["processed"] == 3
