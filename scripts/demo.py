"""Isolated local demo. Synthetic data only; never contacts a model or Qdrant."""

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from aiohttp import web

from evidence.engine import Engine
from evidence.models import Candidate, Settings
from evidence.panel import create_app
from evidence.store import Store


async def run(port: int):
    with tempfile.TemporaryDirectory(prefix="evidence-memory-demo-") as tmp:
        store = Store(Path(tmp) / "demo.sqlite3")
        await store.call("open")
        scope = "demo:GroupMessage:10001"
        await store.call(
            "save_settings",
            Settings(
                mode="active", allowed_scopes=[scope], provider_id="demo-only", online_timeout=3.0
            ),
        )
        for i, (name, subject, text, kind) in enumerate(
            [
                ("小林", "demo-alice", "我准备在十月参加学校的绘画比赛，现在还在画草稿。", "goal"),
                ("阿远", "demo-bob", "我喝咖啡喜欢不加糖，平时比较常喝拿铁。", "preference"),
                ("小林", "demo-alice", "我们约好周六下午去图书馆讨论展示方案。", "agreement"),
                ("小禾", "demo-carol", "上周我报名了摄影课，不过这周还没确认上课时间。", "event"),
            ]
        ):
            mid = await store.call("capture", scope, str(i), subject, name, text)
            rows = await store.call("source_rows", scope, [mid])
            await store.call(
                "save_extraction",
                rows,
                [
                    Candidate(
                        summary=f"{name}自述：{text}",
                        kind=kind,
                        subject_id=subject,
                        stance="self_report",
                        evidence=[{"message_id": mid, "quote": text}],
                    )
                ],
            )

        await store.call(
            "capture",
            scope,
            "neighbor-demo",
            "demo-bob",
            "阿远",
            "你说的是十月的那场比赛吗？",
            sent_at=time.time(),
        )

        async def generate(provider, system, prompt):
            data = json.loads(prompt)["data"]
            if "candidates" in data:
                return json.dumps(
                    {
                        "decisions": [
                            {
                                "id": m["id"],
                                "action": "needs_source",
                                "reason": "演示：核对原始发言",
                            }
                            for m in data["candidates"][:1]
                        ]
                    }
                )
            if "summary" in data:
                m = next(m for m in data["messages"] if m["sender_id"] == data["subject_id"])
                return json.dumps(
                    {
                        "supported": True,
                        "text": data["summary"] + "（历史陈述，当前状态未知）",
                        "reason": "演示：原文支持",
                        "evidence": [{"message_id": m["id"], "quote": m["text"]}],
                    }
                )
            return '{"candidates":[]}'

        engine = Engine(store, generate)
        app = create_app(
            store,
            engine,
            "demo-local-only-token-do-not-use-in-production",
            Path(__file__).resolve().parents[1] / "pages" / "console",
        )
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        print(
            f"Synthetic demo http://127.0.0.1:{port}; token: demo-local-only-token-do-not-use-in-production",
            flush=True,
        )
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
            await store.call("close")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19876)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.port))
    except KeyboardInterrupt:
        pass
