import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from eyewitness.engine import Engine
from eyewitness.evidence import source_quote
from eyewitness.generation import generate_once
from eyewitness.json_output import parse_output
from eyewitness.models import Candidate, CandidateShortlist, GroundedReview, QueryPlan
from tests.conftest import SCOPE, seed
from tests.test_engine import Model, answers


@pytest.mark.parametrize(
    "text,quote,expected",
    [
        ("项目\n请试用\n给点建议", "项目 请试用 给点建议", "项目\n请试用\n给点建议"),
        ("before\r\nproject\tfeedback\nafter", "project feedback", "project\tfeedback"),
        ("项目\n请试用", "项目请试用", None),
        ("项目\n请试用", "项目 请购买", None),
        ("项目\n请试用", "项目，请试用", None),
        ("a\nb xx a\tb", "a b", None),
        ("abc", "  ", None),
    ],
)
def test_quote_restoration_is_contiguous_unambiguous_and_only_whitespace(text, quote, expected):
    assert source_quote(text, quote) == expected


async def test_grounding_restores_original_quote_and_keeps_source_guards(store):
    text = "我做了客户端\n希望大家试用\n并提供建议"
    mid, source = await seed(store, text=text, summary="alice 开发客户端，希望试用和建议")
    reply = answers(mid, source)
    reply[0]["decisions"][0]["evidence"][0]["quote"] = text.replace("\n", " ")
    result = await Engine(store, Model(reply)).recall(SCOPE, "alice 开发的客户端")
    assert result["selected"][0]["evidence"][0]["quote"] == text
    assert json.dumps(text, ensure_ascii=False)[1:-1] in result["injection"]

    # A real phrase attributed to a different/nonexistent source is not repaired.
    reply = answers(mid, "not-a-source")
    reply[0]["decisions"][0]["evidence"][0]["quote"] = text.replace("\n", " ")
    result = await Engine(store, Model(reply)).recall(SCOPE, "alice 开发的客户端")
    assert not result["selected"]


async def test_extraction_stores_restored_source_substring(store):
    text = "我在做客户端\n欢迎试用"
    source = await store.call("capture", SCOPE, "new", "alice", "Alice", text)
    batch = await store.call("source_rows", SCOPE, [source])
    candidate = Candidate(
        summary="alice 开发客户端",
        kind="event",
        subject_id="alice",
        stance="self_report",
        evidence=[{"message_id": source, "quote": "我在做客户端 欢迎试用"}],
    )
    ids = await store.call("save_extraction", batch, [candidate])
    assert len(ids) == 1
    assert (await store.call("detail", ids[0]))["sources"][0]["quote"] == text


def test_long_reason_does_not_discard_valid_decision_but_fact_errors_still_fail():
    raw = {"decisions": [{"id": "m1", "action": "reject", "reason": "解释" * 500}]}
    result, repaired = parse_output(GroundedReview, json.dumps(raw))
    assert result.decisions[0].action == "reject"
    assert len(result.decisions[0].reason) == 300 and repaired == 1
    raw["decisions"][0]["action"] = "maybe"
    with pytest.raises(ValidationError):
        parse_output(GroundedReview, json.dumps(raw))


def test_fences_and_diagnostic_lengths_do_not_invent_required_fields():
    result, _ = parse_output(
        QueryPlan, "```JSON\n" + json.dumps({"queries": [], "reason": "说明" * 400}) + "\n```"
    )
    assert len(result.reason) == 400
    for raw in ["{}", "[]", '{"candidates":[]}', '{"selected":', '说明 {"selected":[]}']:
        with pytest.raises(ValidationError):
            parse_output(CandidateShortlist, raw)


async def test_json_mode_is_request_local_and_preserves_nonthinking_options():
    seen = []

    class Provider:
        provider_config = {
            "type": "openai_chat_completion",
            "custom_extra_body": {"enable_thinking": False, "temperature": 0.8},
        }

        async def text_chat(self, **kwargs):
            seen.append(self.provider_config)
            return "ok"

    provider = Provider()
    original = json.loads(json.dumps(provider.provider_config))
    context = SimpleNamespace(get_provider_by_id=lambda _: provider)
    await generate_once(context, "memory", "system", "prompt", json_mode=True)
    assert provider.provider_config == original
    assert seen[0]["custom_extra_body"] == {
        "enable_thinking": False,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    await generate_once(context, "memory", "system", "prompt")
    assert seen[1] == original
