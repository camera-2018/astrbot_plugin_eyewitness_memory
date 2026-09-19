import pytest

from evidence.admin import AdminAPI
from evidence.engine import Engine
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
