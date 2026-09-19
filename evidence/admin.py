"""Management operations shared by AstrBot Pages and the isolated dev server."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError

from .models import Settings


class AdminAPI:
    def __init__(self, store, engine, providers=None):
        self.store, self.engine, self.providers = store, engine, providers

    async def handle(self, envelope):
        try:
            if not isinstance(envelope, dict) or len(json.dumps(envelope)) > 65536:
                raise ValueError("Invalid request")
            data = await self.dispatch(envelope)
            return data, 200
        except ValidationError:
            return {"error": "配置格式或取值不合法，请检查字段"}, 400
        except (ValueError, KeyError, TypeError):
            return {"error": "请求无效或数据版本冲突，请检查并刷新"}, 400
        except LookupError:
            return {"error": "记录不存在"}, 404
        except Exception:
            return {"error": "管理服务暂时不可用"}, 503

    async def dispatch(self, e):
        path, method, body = e.get("path", ""), e.get("method", "GET"), e.get("body", {})
        if not isinstance(path, str) or not isinstance(body, dict):
            raise ValueError("Invalid operation")
        u = urlsplit(path)
        if u.scheme or u.netloc or u.fragment or "\\" in path:
            raise ValueError("Relative operation required")
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        route, store, engine = u.path, self.store, self.engine
        if route == "overview" and method == "GET":
            cfg = await store.call("get_settings")
            return {
                **await store.call("stats"),
                "mode": cfg.mode,
                "last_error": engine.last_error,
                "last_cycle": engine.last_cycle,
                "scopes": await store.call("list_scopes"),
                "version": "0.1.3",
            }
        if route == "providers" and method == "GET":
            return await self.providers() if self.providers else {"chat": [], "embedding": []}
        if route == "settings" and method in ("GET", "PUT"):
            if method == "PUT":
                await store.call("save_settings", Settings.model_validate(body))
            return (await store.call("get_settings")).model_dump()
        if route == "memories" and method == "GET":
            return await store.call(
                "list_memories",
                q.get("scope", ""),
                q.get("status", ""),
                q.get("q", ""),
                max(0, min(100000, int(q.get("offset", 0)))),
            )
        if re.fullmatch(r"memories/[a-fA-F0-9-]{36}", route):
            mid = route.split("/")[1]
            if method == "DELETE":
                await store.call("delete", mid)
                return {"ok": True}
            if method == "PUT":
                if type(body.get("version")) is not int or not isinstance(body.get("summary"), str):
                    raise ValueError("Invalid edit")
                result = await store.call(
                    "edit", mid, body["version"], body["summary"], body["status"]
                )
            elif method == "GET":
                result = await store.call("detail", mid)
            else:
                raise ValueError("Unsupported method")
            if not result:
                raise LookupError(mid)
            result["context"] = await store.call("source_context", mid, result["scope"])
            return result
        if route == "traces" and method == "GET":
            return await store.call("traces", q.get("scope", ""))
        if route == "preview" and method == "POST":
            if not body.get("query") or not all(
                isinstance(body.get(k, ""), str) for k in ("scope", "query", "sender_id")
            ):
                raise ValueError("Invalid preview")
            return await engine.recall(
                body["scope"], body["query"], body.get("sender_id", ""), preview=True
            )
        if route == "erase-subject" and method == "POST":
            if (
                body.get("confirm") != "DELETE"
                or not body.get("scope")
                or not body.get("subject_id")
            ):
                raise ValueError("Confirmation required")
            return {
                "deleted_memories": await store.call(
                    "erase_subject", body["scope"], body["subject_id"]
                )
            }
        if route == "reindex" and method == "POST":
            return {"queued": await store.call("reindex")}
        raise ValueError("Unknown operation")
