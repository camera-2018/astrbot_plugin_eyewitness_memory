import time

import pytest_asyncio

from eyewitness.models import Candidate, Settings
from eyewitness.store import Store

SCOPE = "default:GroupMessage:100"
OTHER = "default:GroupMessage:200"


@pytest_asyncio.fixture
async def store(tmp_path):
    db = Store(tmp_path / "memory.sqlite3")
    await db.call("open")
    await db.call(
        "save_settings",
        Settings(
            mode="active", allowed_scopes=[SCOPE, OTHER], provider_id="test", online_timeout=3.0
        ),
    )
    yield db
    await db.call("close")


async def seed(
    db,
    scope=SCOPE,
    sender="alice",
    text="我计划下个月参加绘画比赛",
    summary="alice 自述计划下个月参加绘画比赛",
):
    mid = await db.call("capture", scope, str(time.time_ns()), sender, "测试用户", text)
    batch = await db.call("source_rows", scope, [mid])
    candidate = Candidate(
        summary=summary,
        kind="goal",
        subject_id=sender,
        stance="self_report",
        evidence=[{"message_id": mid, "quote": text}],
    )
    ids = await db.call("save_extraction", batch, [candidate])
    return ids[0], mid
