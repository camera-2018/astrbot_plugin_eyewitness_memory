from evidence.engine import Engine
from evidence.panel import create_app, load_token
from tests.conftest import SCOPE, seed

TOKEN = "x" * 48
HEADERS = {"Authorization": "Bearer " + TOKEN}


async def fake(*args):
    return "{}"


async def test_panel_auth_and_static(tmp_path, store, aiohttp_client):
    (tmp_path / "index.html").write_text("hello")
    client = await aiohttp_client(create_app(store, Engine(store, fake), TOKEN, tmp_path))
    assert (await client.get("/")).status == 200
    assert (await client.get("/api/overview")).status == 401
    assert (await client.get("/api/overview?token=" + TOKEN)).status == 401
    r = await client.get("/api/overview", headers=HEADERS)
    assert r.status == 200 and (await r.json())["mode"] == "active"
    assert "Content-Security-Policy" in r.headers
    assert (
        await client.get("/api/overview", headers={**HEADERS, "Sec-Fetch-Site": "cross-site"})
    ).status == 403
    assert (await client.get("/missing.js")).status == 404


async def test_panel_edit_preview_settings(store, tmp_path, aiohttp_client):
    mid, _ = await seed(store)
    client = await aiohttp_client(create_app(store, Engine(store, fake), TOKEN, tmp_path))
    r = await client.put(
        "/api/memories/" + mid,
        headers=HEADERS,
        json={"version": 1, "summary": "alice 曾计划参加绘画比赛", "status": "disabled"},
    )
    assert r.status == 200 and (await r.json())["version"] == 2
    assert (
        await client.put(
            "/api/memories/" + mid,
            headers=HEADERS,
            json={"version": 1, "summary": "冲突的覆盖请求", "status": "active"},
        )
    ).status == 400
    cfg = (await store.call("get_settings")).model_dump()
    cfg["collection"] = "repeat_memory"
    assert (await client.put("/api/settings", headers=HEADERS, json=cfg)).status == 400
    r = await client.post("/api/preview", headers=HEADERS, json={"scope": SCOPE, "query": "哈哈"})
    assert r.status == 200 and not (await r.json())["injection"]
    assert (
        await client.post(
            "/api/erase-subject", headers=HEADERS, json={"scope": SCOPE, "subject_id": "alice"}
        )
    ).status == 400


def test_token_file_permissions_and_persistence(tmp_path):
    path = tmp_path / "panel.token"
    first = load_token(path)
    assert len(first) >= 32 and first == load_token(path)
    assert path.stat().st_mode & 0o777 == 0o600
