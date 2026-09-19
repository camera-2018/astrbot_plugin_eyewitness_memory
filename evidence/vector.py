from __future__ import annotations

import hashlib

import aiohttp

from .models import Settings


class VectorIndex:
    """Optional Qdrant REST adapter; authoritative checks always happen in SQLite."""

    def __init__(self, embed, api_key: str = ""):
        self.embed = embed
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None
        self.ready: set[tuple[str, str]] = set()

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))

    async def close(self):
        if self.session:
            await self.session.close()

    def enabled(self, cfg: Settings):
        return bool(cfg.qdrant_url and cfg.embedding_provider_id)

    async def request(self, cfg: Settings, method: str, path: str, body=None):
        headers = {"api-key": self.api_key} if self.api_key else {}
        async with self.session.request(
            method, cfg.qdrant_url + path, json=body, headers=headers
        ) as r:
            if r.status == 404:
                return None
            r.raise_for_status()
            return await r.json()

    async def collection(self, cfg: Settings, text: str):
        vector, model_signature = await self.embed(cfg.embedding_provider_id, text)
        if not vector or len(vector) > 8192:
            raise ValueError("Invalid embedding dimensions")
        suffix = hashlib.sha256((cfg.embedding_provider_id + model_signature).encode()).hexdigest()[
            :10
        ]
        name = f"{cfg.collection}_{suffix}_{len(vector)}"
        key = (cfg.qdrant_url, name)
        if key not in self.ready:
            data = await self.request(cfg, "GET", f"/collections/{name}")
            if data is None:
                await self.request(
                    cfg,
                    "PUT",
                    f"/collections/{name}",
                    {"vectors": {"size": len(vector), "distance": "Cosine"}},
                )
                await self.request(
                    cfg,
                    "PUT",
                    f"/collections/{name}/index",
                    {"field_name": "scope", "field_schema": "keyword"},
                )
            self.ready.add(key)
        return name, vector

    async def upsert(self, cfg: Settings, memory: dict):
        name, vector = await self.collection(cfg, memory["summary"])
        await self.request(
            cfg,
            "PUT",
            f"/collections/{name}/points?wait=true",
            {
                "points": [
                    {
                        "id": memory["id"],
                        "vector": vector,
                        "payload": {"scope": memory["scope"], "version": memory["version"]},
                    }
                ]
            },
        )

    async def delete(self, cfg: Settings, mid: str):
        # Also removes older model generations in this plugin's collection namespace.
        data = await self.request(cfg, "GET", "/collections")
        for c in (data or {}).get("result", {}).get("collections", []):
            if c["name"].startswith(cfg.collection + "_"):
                await self.request(
                    cfg,
                    "POST",
                    f"/collections/{c['name']}/points/delete?wait=true",
                    {"points": [mid]},
                )

    async def search(self, cfg: Settings, scope: str, query: str):
        name, vector = await self.collection(cfg, query)
        result = await self.request(
            cfg,
            "POST",
            f"/collections/{name}/points/query",
            {
                "query": vector,
                "limit": 12,
                "with_payload": True,
                "filter": {"must": [{"key": "scope", "match": {"value": scope}}]},
            },
        )
        return (result or {}).get("result", {}).get("points", [])
