import pytest

from eyewitness.admin import AdminAPI
from eyewitness.engine import Engine
from tests.conftest import SCOPE, seed


async def fake(*args):
    return "{}"


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/settings",
        "//evil/settings",
        "../settings",
        "settings#x",
        "settings/../settings",
    ],
)
async def test_bridge_dispatch_rejects_unregistered_paths(store, path):
    api = AdminAPI(store, Engine(store, fake))
    _, status = await api.handle({"path": path, "method": "GET"})
    assert status == 400


async def test_bridge_active_settings_and_query(store):
    api = AdminAPI(store, Engine(store, fake))
    cfg = (await store.call("get_settings")).model_dump()
    cfg["mode"] = "active"
    data, status = await api.handle({"path": "settings", "method": "PUT", "body": cfg})
    assert status == 200 and data["mode"] == "active"
    data, status = await api.handle({"path": "memories?scope=" + SCOPE})
    assert status == 200 and data["items"] == []
    assert (await api.handle({"path": "settings", "method": "DELETE"}))[1] == 400
    assert (await api.handle({"path": "settings", "body": "x" * 66000}))[1] == 400


async def test_memory_detail_includes_context_before_and_after_edit(store):
    mid, _ = await seed(store)
    await store.call("capture", SCOPE, "next", "bob", "Bob", "需要我帮你准备吗")
    api = AdminAPI(store, Engine(store, fake))
    for request in [
        {"path": "memories/" + mid},
        {
            "path": "memories/" + mid,
            "method": "PUT",
            "body": {"version": 1, "summary": "alice 曾计划参赛，进度未知", "status": "active"},
        },
    ]:
        data, status = await api.handle(request)
        assert status == 200 and len(data["context"]) == 2
        assert len([r for r in data["context"] if r["is_source"]]) == 1
        assert all(r["time"].endswith("+08:00") for r in data["context"])


async def test_preview_is_separate_from_real_recall_stats_and_trace_filter(store):
    from tests.test_engine import Model, answers

    target, source = await seed(store)
    engine = Engine(store, Model(answers(target, source)))
    await engine.recall(SCOPE, "绘画比赛", preview=True)
    api = AdminAPI(store, engine)
    overview, status = await api.handle({"path": "overview"})
    assert status == 200
    assert overview["recall_24h"]["real"] == {"calls": 0, "injected": 0}
    assert overview["recall_24h"]["preview"] == {"calls": 1, "injected": 1}
    await engine.recall(SCOPE, "哭哭")
    overview, _ = await api.handle({"path": "overview"})
    assert overview["recall_24h"]["real"] == {"calls": 1, "injected": 0}
    real, _ = await api.handle({"path": "traces?mode=active"})
    preview, _ = await api.handle({"path": "traces?mode=preview"})
    assert len(real) == len(preview) == 1
    assert real[0]["query"] == "哭哭" and preview[0]["mode"] == "preview"
    assert (await api.handle({"path": "traces?mode=invalid"}))[1] == 400
