"""Conservative, reversible memory maintenance; never calls a model or deletes evidence."""

from __future__ import annotations

import json
import math
import time
import unicodedata
import uuid
from collections import Counter
from contextlib import contextmanager

MAX_MEMORIES = 100_000
MAX_SOURCES = 500_000


@contextmanager
def _transaction(db, *, write=True):
    """Own our transaction, or keep the caller's transaction intact with a savepoint."""
    nested = db.in_transaction
    name = "maintenance_" + uuid.uuid4().hex
    db.execute(f"SAVEPOINT {name}" if nested else "BEGIN IMMEDIATE" if write else "BEGIN")
    try:
        yield
        if nested:
            db.execute(f"RELEASE {name}")
        else:
            db.commit()
    except BaseException:
        if nested and db.in_transaction:
            db.execute(f"ROLLBACK TO {name}")
            db.execute(f"RELEASE {name}")
        elif not nested:
            db.rollback()
        raise


def ensure_schema(db):
    """Create only owned tables, without committing an existing outer transaction."""
    with _transaction(db):
        db.execute("""CREATE TABLE IF NOT EXISTS maintenance_state (
            id INTEGER PRIMARY KEY CHECK(id=1), next_due REAL NOT NULL,
            interval_seconds INTEGER NOT NULL, last_success REAL,
            last_run_id TEXT
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS maintenance_runs (
            id TEXT PRIMARY KEY, started REAL NOT NULL, finished REAL NOT NULL,
            forced INTEGER NOT NULL, scanned INTEGER NOT NULL,
            counts_json TEXT NOT NULL, decisions_json TEXT NOT NULL
        )""")


def _state(db):
    if not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='maintenance_state'"
    ).fetchone():
        return None
    row = db.execute("SELECT * FROM maintenance_state WHERE id=1").fetchone()
    return dict(row) if row else None


def status(store, cfg):
    """Small read-only panel payload, including the latest successful run summary."""
    state = _state(store.db)
    result = {
        "enabled": cfg.maintenance_enabled,
        "interval_hours": cfg.maintenance_interval_hours,
        "next_due": state["next_due"] if state else None,
        "last_success": state["last_success"] if state else None,
        "last_run": None,
    }
    if state and state["last_run_id"]:
        row = store.db.execute(
            "SELECT id,started,finished,forced,scanned,counts_json "
            "FROM maintenance_runs WHERE id=?",
            (state["last_run_id"],),
        ).fetchone()
        if row:
            result["last_run"] = {
                **{k: row[k] for k in ("id", "started", "finished", "forced", "scanned")},
                "counts": json.loads(row["counts_json"]),
            }
    return result


def _normal_summary(summary):
    # Do not remove punctuation/case or perform semantic similarity matching.
    return " ".join(unicodedata.normalize("NFC", summary).split())


def _scan(db, cfg, now):
    scopes = list(dict.fromkeys(cfg.allowed_scopes))
    if not scopes:
        return [], [], Counter()
    marks = ",".join("?" for _ in scopes)
    memories = [
        dict(r)
        for r in db.execute(
            f"SELECT * FROM memories WHERE scope IN ({marks}) "
            "AND status IN ('active','pending') ORDER BY id LIMIT ?",
            [*scopes, MAX_MEMORIES + 1],
        )
    ]
    if len(memories) > MAX_MEMORIES:
        raise ValueError("维护扫描超过10万条上限，请缩小作用域")
    sources = {}
    source_count = 0
    ids = [m["id"] for m in memories if m["expires"] > now]
    for offset in range(0, len(ids), 400):
        chunk = ids[offset : offset + 400]
        marks = ",".join("?" for _ in chunk)
        for row in db.execute(
            "SELECT s.memory_id,s.message_id,s.quote,x.scope AS message_scope,"
            "x.sender_id,x.text FROM sources s LEFT JOIN messages x ON x.id=s.message_id "
            f"WHERE s.memory_id IN ({marks}) LIMIT ?",
            [*chunk, MAX_SOURCES - source_count + 1],
        ):
            source_count += 1
            if source_count > MAX_SOURCES:
                raise ValueError("维护扫描超过50万条来源上限，请缩小作用域")
            sources.setdefault(row["memory_id"], []).append(dict(row))
    decisions, groups, counters = [], {}, Counter()

    def decide(memory, target, code, reason, **extra):
        counters[code] += 1
        if memory["status"] != target:
            decisions.append(
                {
                    "id": memory["id"],
                    "scope": memory["scope"],
                    "version": memory["version"],
                    "from_status": memory["status"],
                    "to_status": target,
                    "code": code,
                    "reason": reason,
                    **extra,
                }
            )

    for memory in memories:
        evidence = sources.get(memory["id"], [])
        if memory["expires"] <= now:
            decide(memory, "disabled", "expired", "已超过原有有效期；停用且不延长有效期")
        elif not evidence:
            decide(memory, "disabled", "missing_source", "没有可核对的原文来源")
        elif any(row["message_scope"] is None for row in evidence):
            decide(memory, "disabled", "missing_source", "引用的原文来源不存在")
        elif any(row["message_scope"] != memory["scope"] for row in evidence):
            decide(memory, "disabled", "cross_scope_source", "来源属于其他会话，不能用于本会话")
        elif any(not row["quote"] or row["quote"] not in row["text"] for row in evidence):
            decide(memory, "disabled", "invalid_quote", "引用无法与原文逐字匹配")
        elif memory["stance"] == "self_report" and any(
            row["sender_id"] != memory["subject_id"] for row in evidence
        ):
            decide(memory, "pending", "author_mismatch", "自述的证据包含其他作者，退回核验")
        else:
            key = (
                memory["scope"],
                memory["subject_id"],
                memory["kind"],
                memory["stance"],
                _normal_summary(memory["summary"]),
            )
            groups.setdefault(key, []).append(memory)
    for group in groups.values():
        # Prefer an already approved, still-supported memory, then the longest
        # existing lifetime. Never renew a record or promote an unreviewed one.
        group.sort(key=lambda m: (m["status"] != "active", -m["expires"], m["created"], m["id"]))
        keepers = []
        for duplicate in group:
            source_ids = {s["message_id"] for s in sources[duplicate["id"]]}
            event_like = duplicate["kind"] in {"event", "agreement"}
            keeper = next(
                (
                    retained
                    for retained, retained_sources in keepers
                    if not event_like or source_ids & retained_sources
                ),
                None,
            )
            if keeper is None:
                keepers.append((duplicate, source_ids))
                continue
            # Identical descriptions can represent separate events on different
            # dates. Only shared source messages prove an event duplicate; do
            # not merge source sets transitively through discarded records.
            decide(
                duplicate,
                "disabled",
                "exact_duplicate",
                "同会话、主体、类型及立场的摘要归一化后完全相同，"
                + ("且引用同一来源消息；" if event_like else "")
                + "保留有有效来源的一条",
                kept_id=keeper["id"],
            )
    return memories, decisions, counters


def run(store, cfg, *, force=False, now=None, dry_run=False):
    """Synchronously maintain one SQLite connection inside Store.call's serialized worker.

    Automatic calls are opt-in. First call (or an interval change) only schedules;
    force runs immediately, and dry_run previews even while scheduling is disabled.
    Every status change is reversible and leaves messages, sources and expires intact.
    """
    started = time.time() if now is None else now
    if type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
        raise ValueError("无效维护时间")
    hours = cfg.maintenance_interval_hours
    if type(hours) is not int or not 1 <= hours <= 720:
        raise ValueError("维护间隔必须为1到720小时")
    interval = hours * 3600
    db = store.db
    automatic_off = not cfg.maintenance_enabled or cfg.mode == "off"
    if automatic_off and not force and not dry_run:
        return {
            **status(store, cfg),
            "status": "disabled",
            "scanned": 0,
            "counts": {},
            "decisions": [],
        }
    with _transaction(db, write=not dry_run):
        if not dry_run:
            ensure_schema(db)
        state = _state(db)
        if not force and not dry_run:
            if state is None or state["interval_seconds"] != interval:
                db.execute(
                    "INSERT INTO maintenance_state(id,next_due,interval_seconds) VALUES(1,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET next_due=excluded.next_due,"
                    "interval_seconds=excluded.interval_seconds",
                    (started + interval, interval),
                )
                return {
                    **status(store, cfg),
                    "status": "scheduled" if state is None else "rescheduled",
                    "scanned": 0,
                    "counts": {},
                    "decisions": [],
                }
            if state["next_due"] > started:
                return {
                    **status(store, cfg),
                    "status": "not_due",
                    "scanned": 0,
                    "counts": {},
                    "decisions": [],
                }
        memories, decisions, counters = _scan(db, cfg, started)
        counts = {
            **dict(counters),
            "changed": len(decisions),
            "disabled": sum(d["to_status"] == "disabled" for d in decisions),
            "demoted_to_pending": sum(d["to_status"] == "pending" for d in decisions),
            "unchanged": len(memories) - len(decisions),
        }
        if dry_run:
            return {
                **status(store, cfg),
                "status": "preview",
                "scanned": len(memories),
                "counts": counts,
                "decisions": decisions,
            }
        run_id = str(uuid.uuid4())
        for decision in decisions:
            changed = db.execute(
                "UPDATE memories SET status=?,version=version+1,updated=? WHERE id=? "
                "AND version=? AND status=?",
                (
                    decision["to_status"],
                    started,
                    decision["id"],
                    decision["version"],
                    decision["from_status"],
                ),
            )
            if changed.rowcount != 1:
                raise ValueError("维护期间记忆版本发生变化，已撤销本次整理")
            memory = dict(
                db.execute("SELECT * FROM memories WHERE id=?", (decision["id"],)).fetchone()
            )
            db.execute(
                "INSERT INTO versions(memory_id,version,snapshot,actor,created) VALUES(?,?,?,?,?)",
                (
                    memory["id"],
                    memory["version"],
                    json.dumps(memory, ensure_ascii=False),
                    "maintenance:" + run_id + ":" + decision["code"],
                    started,
                ),
            )
            db.execute("INSERT OR REPLACE INTO outbox(memory_id) VALUES(?)", (memory["id"],))
        finished = time.time() if now is None else started
        db.execute(
            "INSERT INTO maintenance_runs VALUES(?,?,?,?,?,?,?)",
            (
                run_id,
                started,
                finished,
                int(force),
                len(memories),
                json.dumps(counts, ensure_ascii=False),
                json.dumps(decisions, ensure_ascii=False),
            ),
        )
        db.execute(
            "INSERT INTO maintenance_state VALUES(1,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "next_due=excluded.next_due,interval_seconds=excluded.interval_seconds,"
            "last_success=excluded.last_success,last_run_id=excluded.last_run_id",
            (finished + interval, interval, finished, run_id),
        )
        return {
            **status(store, cfg),
            "status": "completed",
            "run_id": run_id,
            "scanned": len(memories),
            "counts": counts,
            "decisions": decisions,
        }
