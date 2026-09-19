from __future__ import annotations

import inspect
import json
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart
from astrbot.core.config import AstrBotConfig

from .evidence.admin import AdminAPI
from .evidence.context import history_markers, platform_time
from .evidence.engine import Engine
from .evidence.store import Store
from .evidence.vector import VectorIndex

PLUGIN = "astrbot_plugin_evidence_memory"


class EvidenceMemoryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.data_dir = Path(StarTools.get_data_dir(PLUGIN))
        self.data_dir.chmod(0o700)  # Protect SQLite and its WAL/SHM files.
        self.store = Store(self.data_dir / "memory.sqlite3")
        self.engine = Engine(
            self.store, self.generate, VectorIndex(self.embed, config.get("qdrant_api_key", ""))
        )
        self.admin = AdminAPI(self.store, self.engine, self.providers)
        self.ready = False
        self.context.register_web_api(
            f"/{PLUGIN}/panel", self.panel_info, ["GET"], "External memory panel"
        )
        self.context.register_web_api(f"/{PLUGIN}/api", self.panel_api, ["POST"], "群聊记忆管理")

    async def initialize(self):
        try:
            await self.store.call("open")
            await self.engine.start()
            self.ready = True
            logger.info("群聊记忆已启动；管理页面复用 AstrBot 登录。")
        except Exception:
            await self.terminate()
            raise

    async def generate(self, provider_id: str, system: str, prompt: str) -> str:
        response = await self.context.llm_generate(
            chat_provider_id=provider_id, prompt=prompt, system_prompt=system, max_tokens=3000
        )
        return response.completion_text or ""

    async def embed(self, provider_id: str, text: str):
        provider = self.context.get_provider_by_id(provider_id)
        if inspect.isawaitable(provider):
            provider = await provider
        if provider is None or not hasattr(provider, "get_embedding"):
            raise ValueError("Embedding provider 不可用")
        config = getattr(provider, "provider_config", {})
        signature = json.dumps(
            {k: v for k, v in config.items() if k in ("model", "embedding_model", "type")},
            sort_keys=True,
        )
        return await provider.get_embedding(text), signature

    async def providers(self):
        return {
            "chat": [p.meta().id for p in self.context.get_all_providers()],
            "embedding": [p.meta().id for p in self.context.get_all_embedding_providers()],
        }

    async def panel_info(self):
        from astrbot.api.web import json_response

        return json_response({"enabled": self.ready, "page": "console"})

    async def panel_api(self):
        from astrbot.api.web import json_response, request

        if not self.ready:
            return json_response({"error": "插件未就绪"}, status_code=503)
        # Authentication and plugin-page authorization are enforced by AstrBot.
        data, status = await self.admin.handle(await request.json())
        return json_response(data, status_code=status)

    @staticmethod
    def reply_id(event):
        for part in event.get_messages():
            if part.__class__.__name__ == "Reply":
                return str(getattr(part, "id", ""))
        return ""

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=100)
    async def capture(self, event: AstrMessageEvent):
        if not self.ready or str(event.get_sender_id()) == str(event.get_self_id()):
            return
        try:
            mid = str(getattr(event.message_obj, "message_id", ""))
            if not mid:
                return  # Never invent an ID that makes retries duplicate captures.
            await self.store.call(
                "capture",
                event.unified_msg_origin,
                mid,
                str(event.get_sender_id()),
                event.get_sender_name(),
                event.get_message_str(),
                self.reply_id(event),
                sent_at=platform_time(getattr(event.message_obj, "raw_message", {}).get("time"))
                if hasattr(getattr(event.message_obj, "raw_message", None), "get")
                else None,
            )
        except Exception as exc:
            logger.warning("Evidence Memory 采集降级：%s", type(exc).__name__)

    @filter.on_llm_request(priority=-100)
    async def recall(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.ready or not event.get_group_id():
            return
        try:
            cfg = await self.store.call("get_settings")
            if cfg.mode == "off" or event.unified_msg_origin not in cfg.allowed_scopes:
                return
            # Never stringify image/base64 payloads or copy whole historical contexts.
            visible = (
                (req.prompt or "")
                + "\n"
                + "\n".join(getattr(p, "text", "") for p in (req.extra_user_content_parts or []))
            )
            for message in (req.contexts or [])[-6:]:
                content = (
                    message.get("content")
                    if isinstance(message, dict)
                    else getattr(message, "content", None)
                )
                if isinstance(content, str):
                    visible += "\n" + content[:2000]
                elif isinstance(content, list):
                    for part in content:
                        text = (
                            part.get("text", "")
                            if isinstance(part, dict)
                            else getattr(part, "text", "")
                        )
                        visible += "\n" + text[:2000]
            args = (
                event.unified_msg_origin,
                event.get_message_str(),
                str(event.get_sender_id()),
                self.reply_id(event),
                history_markers(req.contexts) + "\n" + visible[:20000],
            )
            result = await self.engine.recall(*args)
            if result["mode"] == "active" and result["injection"]:
                req.extra_user_content_parts.append(TextPart(text=result["injection"]))
        except Exception as exc:
            logger.warning("Evidence Memory 召回降级：%s", type(exc).__name__)

    async def terminate(self):
        self.ready = False
        await self.engine.close()
        await self.store.call("close")
