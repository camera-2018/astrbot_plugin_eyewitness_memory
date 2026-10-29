import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from eyewitness import maintenance
from eyewitness.store import Store
from tests.conftest import OTHER, SCOPE, seed


def config(*, scopes=(SCOPE,), enabled=True, hours=24):
    return SimpleNamespace(
        mode="active",
        allowed_scopes=list(scopes),
        maintenance_enabled=enabled,
        maintenance_interval_hours=hours,
    )


def row(store, mid):
    return dict(store.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone())


def dump(store):
    return "\n".join(store.db.iterdump())


def expire(store, mid, now):
    with store.db:
        store.db.execute("UPDATE memories SET expires=? WHERE id=?", (now - 1, mid))


async def test_first_call_only_schedules_then_due_run_disables_expired_without_renewal(store):
    mid, source = await seed(store)
    now = time.time()
    expire(store, mid, now)
    old = row(store, mid)
    cfg = config()

    scheduled = maintenance.run(store, cfg, now=now)
    assert scheduled["status"] == "scheduled"
    assert scheduled["next_due"] == now + 86400
    assert row(store, mid) == old
    assert maintenance.run(store, cfg, now=now + 86399)["status"] == "not_due"

    result = maintenance.run(store, cfg, now=now + 86400)
    updated = row(store, mid)
    assert result["counts"]["expired"] == result["counts"]["disabled"] == 1
    assert updated["status"] == "disabled"
    assert updated["expires"] == old["expires"] and updated["version"] == old["version"] + 1
    assert store.db.execute("SELECT text FROM messages WHERE id=?", (source,)).fetchone()
    assert store.db.execute("SELECT quote FROM sources WHERE memory_id=?", (mid,)).fetchone()
    version = store.db.execute(
        "SELECT * FROM versions WHERE memory_id=? ORDER BY version DESC LIMIT 1", (mid,)
    ).fetchone()
    assert json.loads(version["snapshot"]) == updated
    assert version["actor"].endswith(":expired")
    assert result["next_due"] == now + 2 * 86400


async def test_disabled_scheduler_and_dry_run_never_change_data_or_schedule(store):
    mid, _ = await seed(store)
    now = time.time()
    expire(store, mid, now)
    original = dump(store)
    cfg = config(enabled=False)

    assert maintenance.run(store, cfg, now=now)["status"] == "disabled"
    preview = maintenance.run(store, cfg, now=now, dry_run=True)
    assert preview["status"] == "preview" and preview["counts"]["expired"] == 1
    assert preview["next_due"] is None
    assert dump(store) == original

    maintenance.run(store, config(), now=now)
    before = dump(store)
    preview = maintenance.run(store, config(), now=now + 90000, dry_run=True)
    assert preview["next_due"] == now + 86400
    assert dump(store) == before


async def test_only_allowed_scopes_and_active_or_pending_memories_are_touched(store):
    ours, _ = await seed(store)
    other, _ = await seed(store, scope=OTHER)
    disabled, _ = await seed(store, summary="alice 另一条明确计划")
    now = time.time()
    for mid in (ours, other, disabled):
        expire(store, mid, now)
    with store.db:
        store.db.execute("UPDATE memories SET status='disabled' WHERE id=?", (disabled,))
    originals = {mid: row(store, mid) for mid in (other, disabled)}

    result = maintenance.run(store, config(), force=True, now=now)

    assert result["scanned"] == result["counts"]["changed"] == 1
    assert {mid: row(store, mid) for mid in originals} == originals


async def test_invalid_sources_are_disabled_and_author_mismatch_only_returns_to_pending(store):
    ids = []
    for number in range(5):
        ids.append(await seed(store, summary=f"alice 自述计划参加第{number}场绘画比赛"))
    missing, invalid, cross_scope, wrong_author, hearsay = ids
    with store.db:
        store.db.execute("DELETE FROM sources WHERE memory_id=?", (missing[0],))
        store.db.execute(
            "UPDATE sources SET quote='原文没有的内容' WHERE memory_id=?", (invalid[0],)
        )
        store.db.execute("UPDATE messages SET scope=? WHERE id=?", (OTHER, cross_scope[1]))
        store.db.execute("UPDATE messages SET sender_id='bob' WHERE id=?", (wrong_author[1],))
        store.db.execute("UPDATE messages SET sender_id='bob' WHERE id=?", (hearsay[1],))
        store.db.execute(
            "UPDATE memories SET status='pending',stance='hearsay' WHERE id=?", (hearsay[0],)
        )
    messages_before = list(store.db.execute("SELECT * FROM messages ORDER BY id"))
    sources_before = list(store.db.execute("SELECT * FROM sources ORDER BY memory_id,message_id"))

    result = maintenance.run(store, config(), force=True)

    assert result["counts"]["disabled"] == 3
    assert result["counts"]["demoted_to_pending"] == 1
    assert row(store, wrong_author[0])["status"] == "pending"
    assert row(store, hearsay[0])["status"] == "pending"
    assert row(store, hearsay[0])["version"] == 1
    assert list(store.db.execute("SELECT * FROM messages ORDER BY id")) == messages_before
    assert (
        list(store.db.execute("SELECT * FROM sources ORDER BY memory_id,message_id"))
        == sources_before
    )
    again = maintenance.run(store, config(), force=True)
    assert again["counts"]["changed"] == 0


async def test_exact_duplicate_only_same_subject_scope_kind_stance_and_no_unique_fact_lost(store):
    first, _ = await seed(store, text="我喜欢猫", summary="我 喜欢 猫")
    duplicate, _ = await seed(store, text="我喜欢猫", summary="我  喜欢   猫")
    unique, _ = await seed(store, text="我喜欢狗", summary="我 喜欢 狗")
    other_subject, _ = await seed(store, sender="bob", text="我喜欢猫", summary="我 喜欢 猫")
    other_scope, _ = await seed(store, scope=OTHER, text="我喜欢猫", summary="我 喜欢 猫")
    other_kind, _ = await seed(store, text="我喜欢猫", summary=" 我 喜欢 猫 ")
    other_stance, _ = await seed(store, text="我喜欢猫", summary="我\n喜欢 猫")
    now = time.time()
    with store.db:
        store.db.execute("UPDATE memories SET expires=? WHERE id=?", (now + 100, first))
        store.db.execute("UPDATE memories SET expires=? WHERE id=?", (now + 200, duplicate))
        store.db.execute("UPDATE memories SET kind='preference' WHERE id=?", (other_kind,))
        store.db.execute(
            "UPDATE memories SET stance='hearsay',status='pending' WHERE id=?", (other_stance,)
        )
    original_sources = list(store.db.execute("SELECT * FROM sources ORDER BY memory_id,message_id"))

    result = maintenance.run(store, config(scopes=(SCOPE, OTHER)), force=True, now=now)

    assert result["counts"]["exact_duplicate"] == 1
    assert row(store, first)["status"] == "disabled"
    assert result["decisions"][0]["kept_id"] == duplicate
    for mid in (duplicate, unique, other_subject, other_scope, other_kind):
        assert row(store, mid)["status"] == "active"
    assert row(store, other_stance)["status"] == "pending"
    assert (
        list(store.db.execute("SELECT * FROM sources ORDER BY memory_id,message_id"))
        == original_sources
    )
    versions = store.db.execute("SELECT count(*) FROM versions").fetchone()[0]
    assert (
        maintenance.run(store, config(scopes=(SCOPE, OTHER)), force=True, now=now)["counts"][
            "changed"
        ]
        == 0
    )
    assert store.db.execute("SELECT count(*) FROM versions").fetchone()[0] == versions


async def test_duplicate_keeper_must_have_valid_source(store):
    invalid, _ = await seed(store, text="我喜欢猫", summary="我 喜欢 猫")
    valid, _ = await seed(store, text="我喜欢猫", summary="我  喜欢 猫")
    with store.db:
        store.db.execute("DELETE FROM sources WHERE memory_id=?", (invalid,))
    result = maintenance.run(store, config(), force=True)
    assert result["counts"]["missing_source"] == 1
    assert result["counts"].get("exact_duplicate", 0) == 0
    assert row(store, valid)["status"] == "active"


@pytest.mark.parametrize("kind", ["event", "agreement"])
async def test_identical_event_descriptions_on_independent_dates_are_kept(store, kind):
    first, first_source = await seed(store, text="我去参加了比赛", summary="alice 参加比赛")
    second, second_source = await seed(store, text="我去参加了比赛", summary="alice  参加比赛")
    now = time.time()
    with store.db:
        store.db.execute("UPDATE memories SET kind=? WHERE id IN (?,?)", (kind, first, second))
        store.db.execute("UPDATE messages SET sent_at=? WHERE id=?", (now - 86400, first_source))
        store.db.execute("UPDATE messages SET sent_at=? WHERE id=?", (now, second_source))
    originals = {mid: row(store, mid) for mid in (first, second)}

    result = maintenance.run(store, config(), force=True, now=now)

    assert result["counts"]["changed"] == 0
    assert {mid: row(store, mid) for mid in originals} == originals


@pytest.mark.parametrize("kind", ["event", "agreement"])
async def test_same_source_event_duplicates_are_disabled_without_deleting_evidence(store, kind):
    first, source = await seed(store, text="我去参加了比赛", summary="alice 参加比赛")
    duplicate, _ = await seed(store, text="我去参加了比赛", summary="alice  参加比赛")
    now = time.time()
    with store.db:
        store.db.execute("UPDATE memories SET kind=? WHERE id IN (?,?)", (kind, first, duplicate))
        store.db.execute("UPDATE memories SET expires=? WHERE id=?", (now + 200, first))
        store.db.execute("UPDATE memories SET expires=? WHERE id=?", (now + 100, duplicate))
        store.db.execute("UPDATE sources SET message_id=? WHERE memory_id=?", (source, duplicate))
    sources_before = list(store.db.execute("SELECT * FROM sources ORDER BY memory_id,message_id"))

    result = maintenance.run(store, config(), force=True, now=now)

    assert result["counts"]["exact_duplicate"] == 1
    assert row(store, first)["status"] == "active"
    assert row(store, duplicate)["status"] == "disabled"
    assert result["decisions"][0]["kept_id"] == first
    assert (
        list(store.db.execute("SELECT * FROM sources ORDER BY memory_id,message_id"))
        == sources_before
    )


@pytest.mark.parametrize("kind", ["event", "agreement"])
@pytest.mark.parametrize("overlap", ["first", "second", "both"])
async def test_event_duplicates_compare_all_retained_sources_without_merging_events(
    store, kind, overlap
):
    first, first_source = await seed(store, text="我去参加了比赛", summary="alice 参加比赛")
    second, second_source = await seed(store, text="我去参加了比赛", summary="alice  参加比赛")
    duplicate, _ = await seed(store, text="我去参加了比赛", summary="alice   参加比赛")
    now = time.time()
    with store.db:
        for mid, duration in ((first, 300), (second, 200), (duplicate, 100)):
            store.db.execute(
                "UPDATE memories SET kind=?,expires=? WHERE id=?", (kind, now + duration, mid)
            )
        store.db.execute("DELETE FROM sources WHERE memory_id=?", (duplicate,))
        overlapping_sources = {
            "first": [first_source],
            "second": [second_source],
            "both": [first_source, second_source],
        }[overlap]
        store.db.executemany(
            "INSERT INTO sources(memory_id,message_id,quote) VALUES(?,?,?)",
            [(duplicate, sid, "我去参加了比赛") for sid in overlapping_sources],
        )

    result = maintenance.run(store, config(), force=True, now=now)

    assert result["counts"]["exact_duplicate"] == 1
    assert row(store, first)["status"] == row(store, second)["status"] == "active"
    assert row(store, duplicate)["status"] == "disabled"
    assert result["decisions"][0]["kept_id"] == (second if overlap == "second" else first)
    assert maintenance.run(store, config(), force=True, now=now)["counts"]["changed"] == 0


async def test_scheduling_survives_restart_and_interval_change_does_not_run_immediately(store):
    cfg = config()
    first = maintenance.run(store, cfg, now=1000)
    restarted = Store(store.path)
    restarted.db = sqlite3.connect(store.path)
    restarted.db.row_factory = sqlite3.Row
    try:
        later = maintenance.run(restarted, cfg, now=2000)
        assert later["status"] == "not_due" and later["next_due"] == first["next_due"]
        changed = maintenance.run(restarted, config(hours=12), now=3000)
        assert changed["status"] == "rescheduled" and changed["next_due"] == 3000 + 12 * 3600
    finally:
        restarted.db.close()


async def test_audit_failure_rolls_back_all_status_versions_outbox_and_schedule(store):
    mid, _ = await seed(store)
    now = time.time()
    expire(store, mid, now)
    maintenance.run(store, config(), now=now)
    store.db.execute("""CREATE TRIGGER reject_maintenance_audit BEFORE INSERT ON maintenance_runs
                       BEGIN SELECT RAISE(ABORT,'test audit failure'); END""")
    original = dump(store)

    with pytest.raises(sqlite3.IntegrityError, match="test audit failure"):
        maintenance.run(store, config(), now=now + 86400)

    assert not store.db.in_transaction
    assert dump(store) == original


async def test_nested_call_never_commits_callers_transaction(store):
    mid, _ = await seed(store)
    now = time.time()
    expire(store, mid, now)
    original = dump(store)
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("INSERT INTO budgets VALUES('2099-01-01',7)")

    result = maintenance.run(store, config(), force=True, now=now)

    assert result["counts"]["changed"] == 1 and store.db.in_transaction
    with sqlite3.connect(store.path) as observer:
        assert (
            observer.execute("SELECT status FROM memories WHERE id=?", (mid,)).fetchone()[0]
            == "active"
        )
    store.db.rollback()
    assert dump(store) == original


async def test_busy_database_does_not_advance_next_due_or_record_a_run(store):
    now = time.time()
    maintenance.run(store, config(), now=now)
    original = maintenance.status(store, config())
    store.db.execute("PRAGMA busy_timeout=1")
    blocker = sqlite3.connect(store.path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            maintenance.run(store, config(), now=now + 86400)
        assert maintenance.status(store, config()) == original
        assert not store.db.in_transaction
    finally:
        blocker.rollback()
        blocker.close()
    assert maintenance.run(store, config(), now=now + 86400)["status"] == "completed"


async def test_scan_limit_rolls_back_and_does_not_skip_large_unreviewed_population(
    store, monkeypatch
):
    await seed(store)
    await seed(store, summary="另一条尚未审核的独立计划")
    original = dump(store)
    monkeypatch.setattr(maintenance, "MAX_MEMORIES", 1)
    with pytest.raises(ValueError, match="上限"):
        maintenance.run(store, config(), force=True)
    assert dump(store) == original


async def test_failed_nested_run_preserves_callers_uncommitted_changes(store):
    mid, _ = await seed(store)
    now = time.time()
    expire(store, mid, now)
    maintenance.ensure_schema(store.db)
    store.db.execute("""CREATE TRIGGER reject_nested_audit BEFORE INSERT ON maintenance_runs
                       BEGIN SELECT RAISE(ABORT,'nested failure'); END""")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("INSERT INTO budgets VALUES('2099-01-01',7)")
    original = dump(store)
    with pytest.raises(sqlite3.IntegrityError, match="nested failure"):
        maintenance.run(store, config(), force=True, now=now)
    assert store.db.in_transaction
    assert dump(store) == original
    store.db.rollback()


async def test_commit_failure_rolls_back_memory_and_schedule_changes(store):
    mid, _ = await seed(store)
    now = time.time()
    expire(store, mid, now)

    class CommitFailure(sqlite3.Connection):
        def commit(self):
            raise sqlite3.OperationalError("commit blocked")

    connection = sqlite3.connect(":memory:", factory=CommitFailure)
    connection.row_factory = sqlite3.Row
    store.db.backup(connection)
    proxy = SimpleNamespace(db=connection)
    original = dump(proxy)
    try:
        with pytest.raises(sqlite3.OperationalError, match="commit blocked"):
            maintenance.run(proxy, config(), force=True, now=now)
        assert not connection.in_transaction
        assert dump(proxy) == original
    finally:
        connection.close()


async def test_source_scan_limit_rolls_back_without_partial_changes(store, monkeypatch):
    await seed(store)
    original = dump(store)
    monkeypatch.setattr(maintenance, "MAX_SOURCES", 0)
    with pytest.raises(ValueError, match="来源上限"):
        maintenance.run(store, config(), force=True)
    assert dump(store) == original
