from aiohttp import web

from evidence.models import Settings
from evidence.vector import VectorIndex
from tests.conftest import SCOPE


async def test_qdrant_filter_payload_and_namespace(aiohttp_server):
    calls = []
    exists = set()

    async def handler(request):
        body = await request.json() if request.can_read_body else None
        calls.append((request.method, request.path, body))
        path = request.path
        if path == "/collections":
            return web.json_response(
                {
                    "result": {
                        "collections": [{"name": "repeat_memory"}, *[{"name": n} for n in exists]]
                    }
                }
            )
        name = path.split("/")[2]
        if request.method == "GET" and name not in exists:
            raise web.HTTPNotFound()
        if request.method == "PUT" and path == f"/collections/{name}":
            exists.add(name)
        return web.json_response({"result": {"points": []}})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    server = await aiohttp_server(app)
    cfg = Settings(qdrant_url=str(server.make_url("/")).rstrip("/"), embedding_provider_id="bge")

    async def embed(pid, text):
        return [0.1, 0.2, 0.3], "model-v1"

    index = VectorIndex(embed)
    await index.start()
    try:
        memory = {
            "id": "cc5b4d33-c705-40cd-81ec-e1087f2bb916",
            "summary": "原文不应出现在 payload",
            "scope": SCOPE,
            "version": 1,
        }
        await index.upsert(cfg, memory)
        await index.search(cfg, SCOPE, "绘画")
        await index.delete(cfg, memory["id"])
        query = next(b for _, p, b in calls if p.endswith("/points/query"))
        assert query["filter"]["must"][0]["match"]["value"] == SCOPE
        upsert = next(b for _, p, b in calls if p.endswith("/points"))
        assert upsert["points"][0]["payload"] == {"scope": SCOPE, "version": 1}
        assert all("repeat_memory" not in p for _, p, _ in calls)
        old = next(iter(exists))

        async def embed2(pid, text):
            return [0.1, 0.2, 0.3], "model-v2"

        index.embed = embed2
        new, _ = await index.collection(cfg, "foo")
        assert new != old
    finally:
        await index.close()
