import json
import sqlite3
import time
import uuid

import pytest

from eyewitness.models import Candidate, Settings
from eyewitness.store import Store
from tests.conftest import OTHER, SCOPE, seed


async def add_context_memory(store, *, scope=SCOPE, question_id=None, created=None):
    """A short answer is evidence; its separately retained question is not."""
    if question_id is None:
        question_id = await store.call(
            "capture",
            scope,
            uuid.uuid4().hex,
            "bob",
            "提问者",
            "你目前在哪一类企业工作？",
            created=created,
        )
    answer_id = await store.call(
        "capture",
        scope,
        uuid.uuid4().hex,
        "alice",
        "回答者",
        "国企",
        created=created + 60 if created is not None else None,
    )
    batch = await store.call("source_rows", scope, [answer_id])
    memory_id = (
        await store.call(
            "save_extraction",
            batch,
            [
                Candidate(
                    summary=f"alice 在该次问答中自述在国企工作（记录 {answer_id}）",
                    kind="event",
                    subject_id="alice",
                    stance="self_report",
                    evidence=[{"message_id": answer_id, "quote": "国企"}],
                )
            ],
        )
    )[0]
    with store.db:
        store.db.execute(
            "INSERT INTO memory_context(memory_id,message_id) VALUES(?,?)",
            (memory_id, question_id),
        )
    return memory_id, question_id, answer_id


@pytest.mark.parametrize("status", ["active", "disabled"])
async def test_retention_keeps_old_linked_question_without_promoting_it_to_evidence(store, status):
    now = time.time()
    question = await store.call(
        "capture",
        SCOPE,
        "question",
        "bob",
        "提问者",
        "你目前在哪一类企业工作？",
        created=now - 60 * 86400,
    )
    # Keep the question outside the ordinary ten-message neighbor window and
    # omit reply_id: only the explicit memory_context link can recover it.
    unrelated = []
    for i in range(12):
        unrelated.append(
            await store.call(
                "capture",
                SCOPE,
                f"unrelated-{i}",
                "carol",
                "群友",
                f"无关话题第 {i} 条",
                created=now - 45 * 86400 + i,
            )
        )
    mid, _, answer = await add_context_memory(store, question_id=question, created=now - 40 * 86400)
    if status == "disabled":
        memory = await store.call("detail", mid)
        await store.call("edit", mid, memory["version"], memory["summary"], "disabled")
    before = {r["id"]: r for r in await store.call("source_rows", SCOPE, [question, answer])}
    context = {r["id"]: r for r in await store.call("source_context", mid, SCOPE)}
    assert context[question]["is_source"] is False
    assert context[question]["quote"] == ""
    assert context[answer]["is_source"] is True

    await store.call("maintain", Settings(retention_days=30))

    assert {
        r["id"]: r for r in await store.call("source_rows", SCOPE, [question, answer])
    } == before
    assert await store.call("source_rows", SCOPE, unrelated) == []
    context = await store.call("source_context", mid, SCOPE)
    assert [r["id"] for r in context] == [question, answer]
    assert [r["id"] for r in context if r["is_source"]] == [answer]
    assert context[0]["text"] == "你目前在哪一类企业工作？"
    memory = await store.call("detail", mid)
    assert memory["status"] == status
    assert [r["id"] for r in memory["sources"]] == [answer]
    assert await store.call("source_context", mid, OTHER) == []


async def test_erase_question_author_removes_dependent_memories_and_advances_epoch(store):
    first, question, first_answer = await add_context_memory(store)
    second, _, second_answer = await add_context_memory(store, question_id=question)
    independent, independent_source = await seed(store, sender="carol")
    foreign, foreign_question, foreign_answer = await add_context_memory(store, scope=OTHER)
    epoch = await store.call("epoch")

    assert await store.call("erase_subject", SCOPE, "bob") == 2

    assert await store.call("epoch") > epoch
    assert await store.call("source_rows", SCOPE, [question]) == []
    for mid in [first, second]:
        assert await store.call("detail", mid) is None
        assert await store.call("source_context", mid, SCOPE) == []
        for table in ["memory_context", "sources", "versions"]:
            assert (
                store.db.execute(
                    f"SELECT count(*) FROM {table} WHERE memory_id=?", (mid,)
                ).fetchone()[0]
                == 0
            )
        assert (
            store.db.execute("SELECT count(*) FROM memory_fts WHERE id=?", (mid,)).fetchone()[0]
            == 0
        )
        assert store.db.execute("SELECT 1 FROM outbox WHERE memory_id=?", (mid,)).fetchone()
    # Erasing the question author does not erase another person's original
    # answer, unrelated memories, or that author's messages in another group.
    assert {
        r["id"] for r in await store.call("source_rows", SCOPE, [first_answer, second_answer])
    } == {first_answer, second_answer}
    assert (await store.call("detail", independent))["sources"][0]["id"] == independent_source
    assert await store.call("detail", foreign)
    assert {
        r["id"] for r in await store.call("source_rows", OTHER, [foreign_question, foreign_answer])
    } == {foreign_question, foreign_answer}
    assert [tuple(r) for r in store.db.execute("SELECT * FROM memory_context")] == [
        (foreign, foreign_question)
    ]


async def test_delete_memory_unlinks_context_without_deleting_original_messages(store):
    mid, question, answer = await add_context_memory(store)
    await store.call("delete", mid)
    assert store.db.execute("SELECT count(*) FROM memory_context").fetchone()[0] == 0
    assert {r["id"] for r in await store.call("source_rows", SCOPE, [question, answer])} == {
        question,
        answer,
    }


def create_v4_database(path):
    """A v4 database has evidence links and audit history but no context table."""
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE settings (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL);
            CREATE TABLE messages (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, platform_id TEXT NOT NULL,
                sender_id TEXT NOT NULL, sender_name TEXT NOT NULL, text TEXT NOT NULL,
                reply_id TEXT, created REAL NOT NULL, processed INTEGER NOT NULL DEFAULT 0,
                sent_at REAL, extraction_error TEXT NOT NULL DEFAULT '',
                extraction_attempted_at REAL, UNIQUE(scope,platform_id)
            );
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, subject_id TEXT NOT NULL,
                summary TEXT NOT NULL, kind TEXT NOT NULL, stance TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                expires REAL NOT NULL, version INTEGER NOT NULL DEFAULT 1,
                fingerprint TEXT NOT NULL UNIQUE
            );
            CREATE TABLE sources (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                quote TEXT NOT NULL, PRIMARY KEY(memory_id,message_id)
            );
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY, memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                version INTEGER NOT NULL, snapshot TEXT NOT NULL, actor TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE privacy_epoch (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
            PRAGMA user_version=4;
        """)
        db.execute(
            "INSERT INTO settings VALUES(1,?)",
            (json.dumps({"mode": "active", "allowed_scopes": [SCOPE], "provider_id": "test"}),),
        )
        db.execute(
            "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "answer",
                SCOPE,
                "old-answer",
                "alice",
                "回答者",
                "我在国企工作",
                "question",
                1700000060,
                1,
                1700000000,
                "",
                1700000061,
            ),
        )
        memory = (
            "memory",
            SCOPE,
            "alice",
            "alice 自述在国企工作",
            "event",
            "self_report",
            "active",
            1700000062,
            1700000063,
            1900000000,
            2,
            "legacy-fingerprint",
        )
        db.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", memory)
        db.execute("INSERT INTO sources VALUES('memory','answer','我在国企工作')")
        db.executemany(
            "INSERT INTO versions VALUES(?,?,?,?,?,?)",
            [
                (1, "memory", 1, '{"summary":"旧摘要"}', "extractor", 1700000062),
                (2, "memory", 2, '{"summary":"alice 自述在国企工作"}', "panel-admin", 1700000063),
            ],
        )
        db.execute("INSERT INTO privacy_epoch VALUES(1,7)")


def persisted_rows(db):
    return {
        table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1")]
        for table in ["settings", "messages", "memories", "sources", "versions", "privacy_epoch"]
    }


async def test_v4_migration_preserves_data_and_reopening_is_idempotent(tmp_path):
    path = tmp_path / "legacy-v4.sqlite3"
    create_v4_database(path)
    with sqlite3.connect(path) as db:
        before = persisted_rows(db)
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_context'").fetchone()
    store = Store(path)
    try:
        for _ in range(2):
            await store.call("open")
            assert store.db.execute("PRAGMA user_version").fetchone()[0] == 5
            assert store.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert persisted_rows(store.db) == before
            assert store.db.execute("SELECT count(*) FROM memory_context").fetchone()[0] == 0
            memory = await store.call("detail", "memory")
            assert memory["version"] == 2
            assert [r["id"] for r in memory["sources"]] == ["answer"]
            assert len(memory["versions"]) == 2
            context = await store.call("source_context", "memory", SCOPE)
            assert len(context) == 1 and context[0]["is_source"] is True
            with pytest.raises(sqlite3.IntegrityError), store.db:
                store.db.execute("INSERT INTO memory_context VALUES('memory','nonexistent')")
            await store.call("close")
    finally:
        await store.call("close")


@pytest.mark.parametrize("version", [6, 999])
async def test_newer_schema_is_rejected_without_mutation(tmp_path, version):
    path = tmp_path / "newer.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE future_data (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        db.execute("INSERT INTO future_data VALUES(1,'不能被旧插件改写')")
        db.execute(f"PRAGMA user_version={version}")
    before = path.read_bytes()
    store = Store(path)
    with pytest.raises(ValueError, match="数据库版本较新"):
        await store.call("open")
    assert store.db is None
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == version
        assert db.execute("SELECT * FROM future_data").fetchall() == [(1, "不能被旧插件改写")]
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_context'").fetchone()
