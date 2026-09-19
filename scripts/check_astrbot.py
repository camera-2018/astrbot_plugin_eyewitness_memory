"""Run with an AstrBot environment: PYTHONPATH=<repo> python scripts/check_astrbot.py."""

import asyncio
import copy
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path


async def check():
    # AstrBot imports initialize some runtime assets; isolate those too.
    sandbox = tempfile.TemporaryDirectory(prefix="evidence-sdk-runtime-")
    previous_root = os.environ.get("ASTRBOT_ROOT")
    os.environ["ASTRBOT_ROOT"] = sandbox.name
    from astrbot.api.provider import ProviderRequest
    from astrbot.core.agent.message import Message, dump_messages_with_checkpoints

    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("evidence_plugin_check")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("evidence_plugin_check.main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Context:
        def register_web_api(self, *args):
            assert args[0].endswith(("/panel", "/api"))

        def get_all_providers(self):
            return []

        def get_all_embedding_providers(self):
            return []

    class Event:
        unified_msg_origin = "default:GroupMessage:100"
        message_obj = types.SimpleNamespace(
            message_id="integration-1", raw_message={"time": 1789800000}
        )

        def get_group_id(self):
            return "100"

        def get_sender_id(self):
            return "alice"

        def get_self_id(self):
            return "bot"

        def get_sender_name(self):
            return "Alice"

        def get_message_str(self):
            return "绘画比赛怎么样了"

        def get_messages(self):
            return []

    with tempfile.TemporaryDirectory(prefix="evidence-astrbot-check-") as tmp:
        module.StarTools.get_data_dir = lambda *_: Path(tmp)
        plugin = module.EvidenceMemoryPlugin(
            Context(), {"panel_host": "127.0.0.1", "panel_port": 0}
        )
        await plugin.initialize()
        try:
            assert plugin.ready
            cfg = await plugin.store.call("get_settings")
            cfg.mode = "active"
            cfg.allowed_scopes = [Event.unified_msg_origin]
            await plugin.store.call("save_settings", cfg)
            await plugin.capture(Event())
            captured = (await plugin.store.call("recent", Event.unified_msg_origin))[0]
            assert captured["sent_at"] == Event.message_obj.raw_message["time"]

            async def recall(*args):
                return {"mode": "active", "injection": "一条追加且随历史保留的测试记忆"}

            plugin.engine.recall = recall
            req = ProviderRequest(
                prompt="测试问题",
                system_prompt="固定系统提示词",
                contexts=[{"role": "user", "content": "旧消息"}],
            )
            from astrbot.core.agent.message import TextPart

            req.extra_user_content_parts.append(TextPart(text="已有附加内容"))
            old_contexts = copy.deepcopy(req.contexts)
            await plugin.recall(Event(), req)
            assert req.system_prompt == "固定系统提示词" and req.contexts == old_contexts
            assert req.prompt == "测试问题" and len(req.extra_user_content_parts) == 2
            assert req.extra_user_content_parts[0].text == "已有附加内容"
            assembled = await req.assemble_context()
            assert [p["text"] for p in assembled["content"]] == [
                "测试问题",
                "已有附加内容",
                "一条追加且随历史保留的测试记忆",
            ]
            dumped = dump_messages_with_checkpoints(
                [Message(role="user", content=assembled["content"])]
            )
            assert dumped[0]["content"] == assembled["content"], (
                "Memory snapshot must remain in history"
            )
            previous = copy.deepcopy(dumped)
            req2 = ProviderRequest(
                prompt="第二轮",
                system_prompt=req.system_prompt,
                contexts=dumped + [{"role": "assistant", "content": "上轮回复"}],
            )
            history_before = copy.deepcopy(req2.contexts)
            await plugin.recall(Event(), req2)
            assert req2.contexts == history_before and req2.contexts[: len(previous)] == previous
            assert req2.system_prompt == req.system_prompt
            print(
                "PASS: raw platform timestamp, user-tail append, persistent snapshot, unchanged system/history across two turns"
            )
            cfg.mode = "off"
            await plugin.store.call("save_settings", cfg)
            called = False

            async def unexpected_recall(*args):
                nonlocal called
                called = True
                return {"mode": "active", "injection": "must not be used"}

            plugin.engine.recall = unexpected_recall
            req = ProviderRequest(prompt="测试停用后不召回")
            await asyncio.wait_for(plugin.recall(Event(), req), timeout=1)
            assert not called and not req.extra_user_content_parts
            print("PASS: disabled hook skips recall and injection")
        finally:
            await plugin.terminate()
        assert not plugin.ready and plugin.engine.worker_task.done()
        # Same port/data lifecycle must be safe after hot reload.
        await plugin.initialize()
        await plugin.terminate()
        print("PASS: shutdown and reload")
    if previous_root is None:
        os.environ.pop("ASTRBOT_ROOT", None)
    else:
        os.environ["ASTRBOT_ROOT"] = previous_root
    sandbox.cleanup()


if __name__ == "__main__":
    asyncio.run(check())
