"""Run with an AstrBot environment: PYTHONPATH=<repo> python scripts/check_astrbot.py."""

import asyncio
import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path


async def check():
    # AstrBot imports initialize some runtime assets; isolate those too.
    sandbox = tempfile.TemporaryDirectory(prefix="eyewitness-sdk-runtime-")
    previous_root = os.environ.get("ASTRBOT_ROOT")
    os.environ["ASTRBOT_ROOT"] = sandbox.name
    from astrbot.api.provider import ProviderRequest
    from astrbot.core.agent.message import Message, dump_messages_with_checkpoints
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    from astrbot.core.config import AstrBotConfig

    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("eyewitness_plugin_check")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("eyewitness_plugin_check.main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Context:
        tool_schema_mode = "skills_like"

        def register_web_api(self, *args):
            assert args[0].startswith("/astrbot_plugin_eyewitness_memory/")
            assert args[0].endswith(("/panel", "/api"))

        def get_all_providers(self):
            return []

        def get_all_embedding_providers(self):
            return []

        async def get_using_provider_async(self, umo):
            assert umo == Event.unified_msg_origin
            return types.SimpleNamespace(provider_config={"max_context_tokens": 8192})

        def get_config(self, umo):
            assert umo == Event.unified_msg_origin
            return {
                "agent_runner": {"config": {"misc": {"tool_schema_mode": self.tool_schema_mode}}}
            }

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

    with tempfile.TemporaryDirectory(prefix="eyewitness-astrbot-check-") as tmp:

        def data_dir(name):
            assert name == "astrbot_plugin_eyewitness_memory"
            return Path(tmp)

        module.StarTools.get_data_dir = data_dir
        config = AstrBotConfig(
            config_path=str(Path(tmp) / "native-config.json"),
            schema=json.loads((root / "_conf_schema.json").read_text()),
        )
        plugin = module.EyewitnessMemoryPlugin(Context(), config)
        await plugin.initialize()
        try:
            assert plugin.ready
            cfg = await plugin.store.call("get_settings")
            cfg.mode = "active"
            cfg.allowed_scopes = [Event.unified_msg_origin]
            await plugin.store.call("save_settings", cfg)
            assert config["mode"] == "active" and config["allowed_scopes"] == cfg.allowed_scopes
            assert (
                json.loads(Path(config.config_path).read_text(encoding="utf-8-sig"))["mode"]
                == "active"
            )
            config.save_config({"provider_id": "native-selection"})
            assert (await plugin.store.call("get_settings")).provider_id == "native-selection"
            assert config.schema["embedding_provider_id"]["options"] == [""]
            print("PASS: native AstrBot config and plugin settings share values and persistence")
            await plugin.capture(Event())
            captured = (await plugin.store.call("recent", Event.unified_msg_origin))[0]
            assert captured["sent_at"] == Event.message_obj.raw_message["time"]

            async def recall(*args, **kwargs):
                assert kwargs["budget"].available > 0
                assert kwargs["count_injection_tokens"]("一段记忆") > 0
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
            from eyewitness_plugin_check.eyewitness.budget import request_budget

            crowded = ProviderRequest(
                prompt="本轮消息",
                system_prompt="系统提示词",
                contexts=[{"role": "user", "content": "历史消息" * 3000}],
            )
            limited, _ = await request_budget(plugin.context, Event(), crowded)
            assert limited.available == 0 and "上下文预算" in limited.reason
            assert limited.native_messages > 0
            assert limited.estimated_messages < limited.message_chars
            tools = ToolSet(
                tools=[
                    FunctionTool(
                        name="lookup",
                        description="Lookup an item",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Long parameter guidance " * 100,
                                }
                            },
                            "required": ["query"],
                        },
                    )
                ]
            )
            schema_req = ProviderRequest(prompt="查找", func_tool=tools)
            light, _ = await request_budget(plugin.context, Event(), schema_req)
            plugin.context.tool_schema_mode = "full"
            full, _ = await request_budget(plugin.context, Event(), schema_req)
            assert light.tool_schema_mode == "skills_like"
            assert light.estimated_tools < full.estimated_tools
            assert schema_req.func_tool is tools
            print(
                "PASS: raw timestamp, light tool budget, persistent user-tail append, unchanged system/history"
            )
            cfg.mode = "off"
            await plugin.store.call("save_settings", cfg)
            called = False

            async def unexpected_recall(*args, **kwargs):
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
