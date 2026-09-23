from eyewitness.budget import InjectionBudget, fit_memory_block, plan_budget
from eyewitness.context import render_memory
from tests.conftest import SCOPE, seed


def test_headroom_shrinks_as_history_and_tools_grow():
    small = plan_budget(8192, 1000, 500)
    large = plan_budget(8192, 5000, 500)
    assert 0 < large.available < small.available
    assert plan_budget(8192, 7000).available == 0
    assert small.reserved >= 512


async def test_token_budget_discards_neighbors_but_keeps_source(store):
    mid, source = await seed(store)
    for i in range(10):
        await store.call("capture", SCOPE, f"nearby-{i}", "bob", "B", "邻近对话" * 20 + str(i))
    rows = await store.call("source_context", mid, SCOPE)
    item = {
        "id": mid,
        "version": 1,
        "subject_id": "alice",
        "text": "alice 曾计划绘画比赛",
        "context": rows,
        "evidence": [{"message_id": source}],
    }
    whole, _ = render_memory(item, 8000)
    block, ids, reason = fit_memory_block(item, 8000, 1000, "固定前缀\n", len, set())
    assert reason == ""
    assert source in ids and "我计划下个月参加绘画比赛" in block
    assert len(block) < len(whole)
    assert len("固定前缀\n" + block) <= 1000


async def test_tiny_budget_omits_all_instead_of_clipping_source(store):
    mid, source = await seed(store)
    rows = await store.call("source_context", mid, SCOPE)
    item = {
        "id": mid,
        "version": 1,
        "subject_id": "alice",
        "text": "alice 曾计划绘画比赛",
        "context": rows,
        "evidence": [{"message_id": source}],
    }
    block, ids, reason = fit_memory_block(item, 8000, 50, "固定前缀\n", len, set())
    assert not block and not ids and reason == "tokens"


def test_budget_metadata_is_data_only():
    assert InjectionBudget(8192, 1000, 2000, 5192).available == 5192
