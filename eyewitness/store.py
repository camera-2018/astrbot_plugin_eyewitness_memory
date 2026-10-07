from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .config import ConfigSettings, settings_revision
from .context import bounded_context, context_message, message_order, message_time, platform_time
from .errors import AuxiliaryModelFailure, RetryConflict
from .models import Candidate, Settings, extraction_noise, safe_text, terms


class Store:
    """Serialized SQLite access off the event loop; each write is one transaction."""

    # messages.processed: 0 pending, 1 complete, 2 noise, 3 failed, 4 in flight.

    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()
        self.db: sqlite3.Connection | None = None
        self.config_settings: ConfigSettings | None = None
        self.last_batch_scope = ""

    async def call(self, name: str, *args, **kwargs):
        async with self.lock:
            task = asyncio.create_task(asyncio.to_thread(getattr(self, name), *args, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A timeout cannot stop a running SQLite thread. Keep the lock until it
                # finishes so another request cannot interleave on the same connection.
                await task
                raise

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
        self.db.row_factory = sqlite3.Row
        if self.db.execute("PRAGMA user_version").fetchone()[0] > 4:
            self.db.close()
            self.db = None
            raise ValueError("数据库版本较新，不能用旧版插件打开")
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, platform_id TEXT NOT NULL,
                sender_id TEXT NOT NULL, sender_name TEXT NOT NULL, text TEXT NOT NULL,
                reply_id TEXT, created REAL NOT NULL, processed INTEGER NOT NULL DEFAULT 0,
                UNIQUE(scope, platform_id)
            );
            CREATE INDEX IF NOT EXISTS msg_scope_time ON messages(scope, created);
            CREATE INDEX IF NOT EXISTS msg_pending ON messages(processed, scope, created);
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, subject_id TEXT NOT NULL,
                summary TEXT NOT NULL, kind TEXT NOT NULL, stance TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                expires REAL NOT NULL, version INTEGER NOT NULL DEFAULT 1,
                fingerprint TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS mem_scope_status ON memories(scope, status, updated);
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, tokens);
            CREATE TABLE IF NOT EXISTS sources (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                quote TEXT NOT NULL, PRIMARY KEY(memory_id, message_id)
            );
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY, memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                version INTEGER NOT NULL, snapshot TEXT NOT NULL, actor TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outbox (
                memory_id TEXT PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0,
                retry_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, query TEXT NOT NULL,
                mode TEXT NOT NULL, reason TEXT NOT NULL, data TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL, created REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS trace_scope_time ON traces(scope, created);
            CREATE TABLE IF NOT EXISTS budgets (day TEXT PRIMARY KEY, calls INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS model_calls (
                id TEXT PRIMARY KEY, stage TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, elapsed_ms INTEGER, prompt_chars INTEGER NOT NULL,
                input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER,
                error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS model_calls_time ON model_calls(created);
            CREATE TABLE IF NOT EXISTS privacy_epoch (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
            INSERT OR IGNORE INTO privacy_epoch VALUES(1,0);
        """)
        # Retired preview-only installs must not start injecting after an upgrade.
        with self.db:
            columns = {r[1] for r in self.db.execute("PRAGMA table_info(messages)")}
            if "sent_at" not in columns:
                self.db.execute("ALTER TABLE messages ADD COLUMN sent_at REAL")
            if "extraction_error" not in columns:
                self.db.execute(
                    "ALTER TABLE messages ADD COLUMN extraction_error TEXT NOT NULL DEFAULT ''"
                )
            if "extraction_attempted_at" not in columns:
                self.db.execute("ALTER TABLE messages ADD COLUMN extraction_attempted_at REAL")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS msg_event_time ON messages(scope,COALESCE(sent_at,created),created,id)"
            )
            # An interrupted request may already have been billed. Never resubmit it
            # automatically after a process crash or plugin reload.
            self.db.execute(
                "UPDATE messages SET processed=3,extraction_error=? WHERE processed=4",
                ("上次提取被中断，未自动重试",),
            )
            self.db.execute(
                "UPDATE model_calls SET status='interrupted',error='请求被中断，用量未知' "
                "WHERE status='running'"
            )
            self.db.execute("PRAGMA user_version=4")
            self.db.execute("""
                UPDATE settings SET data=json_set(data, '$.mode', 'off')
                WHERE json_extract(data, '$.mode')='shadow'
            """)
        self.path.chmod(0o600)

    def close(self):
        if self.db:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.close()
            self.db = None

    def get_settings(self) -> Settings:
        if self.config_settings is not None:
            return self.config_settings.read()
        return self._cached_settings() or Settings()

    def _cached_settings(self) -> Settings | None:
        row = self.db.execute("SELECT data FROM settings WHERE id=1").fetchone()
        if not row:
            return None
        values = json.loads(row[0])
        # Older snapshots can contain settings removed from the native schema.
        return Settings.model_validate({k: values[k] for k in Settings.model_fields if k in values})

    def bind_config(self, config):
        previous = self._cached_settings()
        backend = ConfigSettings(config)
        current = backend.migrate(previous)
        self._cache_settings(previous or Settings(), current)
        self.config_settings = backend

    def settings_document(self):
        settings = self.get_settings()
        return {**settings.model_dump(), "revision": settings_revision(settings)}

    def save_settings(
        self, settings: Settings, qdrant_api_key: str | None = None, revision: str | None = None
    ):
        old = self.get_settings()
        if revision is not None and revision != settings_revision(old):
            raise ValueError("Settings changed; reload before saving")
        if self.config_settings is not None:
            self.config_settings.write(settings, qdrant_api_key)
        elif qdrant_api_key is not None:
            raise ValueError("Secrets require AstrBot configuration")
        self._cache_settings(old, settings)

    def _cache_settings(self, old: Settings, settings: Settings):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES(1,?)", (settings.model_dump_json(),)
            )
            if (old.embedding_provider_id, old.collection, old.qdrant_url) != (
                settings.embedding_provider_id,
                settings.collection,
                settings.qdrant_url,
            ):
                self.db.execute("INSERT OR REPLACE INTO outbox(memory_id) SELECT id FROM memories")

    def size_bytes(self):
        return sum(
            p.stat().st_size for p in [self.path, Path(str(self.path) + "-wal")] if p.exists()
        )

    def skip_existing_noise(self) -> int:
        rows = self.db.execute("SELECT id,text,reply_id FROM messages WHERE processed=0").fetchall()
        skipped = [
            (row["id"],) for row in rows if extraction_noise(row["text"], bool(row["reply_id"]))
        ]
        with self.db:
            self.db.executemany("UPDATE messages SET processed=2 WHERE id=?", skipped)
        return len(skipped)

    def capture(
        self,
        scope: str,
        platform_id: str,
        sender_id: str,
        sender_name: str,
        text: str,
        reply_id: str | None = None,
        created: float | None = None,
        sent_at: float | None = None,
    ) -> str | None:
        cfg = self.get_settings()
        if cfg.mode == "off" or scope not in cfg.allowed_scopes or sender_id in cfg.bot_ids:
            return None
        if self.size_bytes() > cfg.max_db_mb * 1024**2:
            return None
        text = safe_text(text).strip()
        if not text or text == "[已脱敏]":
            return None
        mid = str(uuid.uuid5(uuid.NAMESPACE_URL, scope + ":" + platform_id))
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO messages(id,scope,platform_id,sender_id,sender_name,text,reply_id,created,sent_at,processed) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    scope,
                    platform_id,
                    sender_id,
                    safe_text(sender_name, 80),
                    text,
                    reply_id,
                    created or time.time(),
                    platform_time(sent_at),
                    2 if extraction_noise(text, bool(reply_id)) else 0,
                ),
            )
        return mid

    def list_scopes(self):
        result = {
            s: {"scope": s, "messages": 0, "pending": 0} for s in self.get_settings().allowed_scopes
        }
        for row in self.db.execute(
            "SELECT scope,count(*) messages,sum(processed=0) pending FROM messages GROUP BY scope"
        ):
            result[row["scope"]] = dict(row)
        return list(result.values())

    def recent(self, scope: str, limit: int = 12):
        return [
            dict(r)
            for r in reversed(
                self.db.execute(
                    "SELECT * FROM messages WHERE scope=? ORDER BY created DESC LIMIT ?",
                    (scope, limit),
                ).fetchall()
            )
        ]

    def referenced_message(self, scope: str, platform_id: str):
        row = self.db.execute(
            "SELECT * FROM messages WHERE scope=? AND platform_id=?", (scope, platform_id)
        ).fetchone()
        return dict(row) if row else None

    def pending_batch(self, cfg: Settings, scan_limit: int | None = None):
        now = time.time()
        scopes = list(cfg.allowed_scopes)
        if self.last_batch_scope in scopes:
            start = scopes.index(self.last_batch_scope) + 1
            scopes = scopes[start:] + scopes[:start]
        for scope in scopes:
            row = self.db.execute(
                "SELECT count(*),min(created) FROM messages WHERE scope=? AND processed=0", (scope,)
            ).fetchone()
            if row[0] and (row[0] >= cfg.batch_size or now - row[1] >= cfg.batch_age_seconds):
                self.last_batch_scope = scope
                return [
                    dict(r)
                    for r in self.db.execute(
                        "SELECT * FROM messages WHERE scope=? AND processed=0 ORDER BY created LIMIT ?",
                        (scope, min(scan_limit or cfg.batch_size, 300)),
                    )
                ]
        return []

    def extraction_context(self, batch: list[dict]):
        """Bounded preceding/replied-to messages, including already processed rows."""
        if not batch:
            return []
        scope = batch[0]["scope"]
        rows = [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM messages WHERE scope=? AND created<? AND created>=? "
                "ORDER BY created DESC LIMIT 5",
                (scope, batch[0]["created"], batch[0]["created"] - 900),
            )
        ]
        for item in batch:
            if item.get("reply_id"):
                row = self.referenced_message(scope, item["reply_id"])
                if row:
                    rows.append(row)
        batch_ids = {r["id"] for r in batch}
        return list({r["id"]: r for r in rows if r["id"] not in batch_ids}.values())[:20]

    def claim_extraction(self, batch: list[dict]) -> bool:
        """Persist the attempt before calling a billable provider, all-or-nothing."""
        if not batch:
            return False
        scope = batch[0]["scope"]
        ids = list(dict.fromkeys(r["id"] for r in batch))
        if any(r["scope"] != scope for r in batch):
            raise ValueError("提取批次不能跨群")
        marks = ",".join("?" for _ in ids)
        with self.db:
            cursor = self.db.execute(
                f"UPDATE messages SET processed=4,extraction_error='',extraction_attempted_at=? "
                f"WHERE scope=? AND id IN ({marks}) AND processed=0 "
                f"AND (SELECT count(*) FROM messages WHERE scope=? "
                f"AND id IN ({marks}) AND processed=0)=?",
                [time.time(), scope, *ids, scope, *ids, len(ids)],
            )
        return cursor.rowcount == len(ids)

    def fail_extraction(self, batch: list[dict], reason: str):
        # Only update claimed rows; never recreate privacy-deleted messages or
        # overwrite already completed work. Diagnostics contain no response body.
        with self.db:
            self.db.executemany(
                "UPDATE messages SET processed=3,extraction_error=? "
                "WHERE scope=? AND id=? AND processed=4",
                [(safe_text(reason, 500), r["scope"], r["id"]) for r in batch],
            )

    def failed_batches(self, scope: str = "", offset: int = 0):
        cfg = self.get_settings()
        where = "processed=3 AND (?='' OR scope=?)"
        grouped = " FROM messages WHERE " + where + " GROUP BY scope,extraction_attempted_at"
        total = self.db.execute(
            "SELECT count(*) FROM (SELECT 1" + grouped + ")", (scope, scope)
        ).fetchone()[0]
        rows = self.db.execute(
            "SELECT min(id) id,scope,extraction_attempted_at attempted_at,count(*) messages,"
            "min(created) first_message,max(created) last_message,max(extraction_error) reason"
            + grouped
            + " ORDER BY extraction_attempted_at DESC LIMIT 30 OFFSET ?",
            (scope, scope, max(0, offset)),
        )
        return {
            "total": total,
            "items": [
                {
                    **dict(row),
                    "can_retry": bool(
                        cfg.mode == "active"
                        and cfg.provider_id
                        and row["scope"] in cfg.allowed_scopes
                        and row["attempted_at"] is not None
                    ),
                }
                for row in rows
            ],
        }

    def retry_failed_batch(self, anchor_id: str, attempted_at: float):
        if (
            type(attempted_at) not in (float, int)
            or not math.isfinite(attempted_at)
            or attempted_at <= 0
        ):
            raise ValueError("Invalid attempt timestamp")
        cfg = self.get_settings()
        with self.db:
            row = self.db.execute(
                "SELECT scope,processed,extraction_attempted_at FROM messages WHERE id=?",
                (anchor_id,),
            ).fetchone()
            if not row or row["processed"] != 3 or row["extraction_attempted_at"] != attempted_at:
                raise RetryConflict("该失败批次已被重提、删除或状态已变化，请刷新")
            if (
                cfg.mode != "active"
                or not cfg.provider_id
                or row["scope"] not in cfg.allowed_scopes
            ):
                raise RetryConflict("请先启用该群记录并配置辅助模型")
            cursor = self.db.execute(
                "UPDATE messages SET processed=0,extraction_error='',extraction_attempted_at=NULL "
                "WHERE processed=3 AND scope=? AND extraction_attempted_at=?",
                (row["scope"], attempted_at),
            )
        return {"queued": cursor.rowcount, "scope": row["scope"]}

    def source_rows(self, scope: str, ids: list[str]):
        if not ids:
            return []
        marks = ",".join("?" for _ in ids[:500])
        return [
            dict(r)
            for r in self.db.execute(
                f"SELECT * FROM messages WHERE scope=? AND id IN ({marks}) ORDER BY created",
                [scope, *ids[:500]],
            )
        ]

    def save_extraction(
        self,
        batch: list[dict],
        candidates: list[Candidate],
        context: list[dict] = (),
        expected_epoch: int | None = None,
    ):
        if not batch:
            return []
        if expected_epoch is not None and self.epoch() != expected_epoch:
            raise AuxiliaryModelFailure("提取期间发生隐私删除，候选已撤销")
        scope = batch[0]["scope"]
        # Reload source rows: concurrent privacy deletion must invalidate in-flight extraction.
        rows = {r["id"]: r for r in self.source_rows(scope, [r["id"] for r in [*batch, *context]])}
        batch_ids = {r["id"] for r in batch}
        ids = []
        with self.db:
            for c in candidates:
                if c.stance in ("joke", "uncertain") or not c.subject_id:
                    continue
                if not any(e.message_id in batch_ids for e in c.evidence):
                    continue  # Context alone must not repeatedly create old memories.
                if any(
                    e.message_id not in rows or e.quote not in rows[e.message_id]["text"]
                    for e in c.evidence
                ):
                    continue
                if safe_text(c.summary) != c.summary:
                    continue
                status = (
                    "active"
                    if c.stance == "self_report"
                    and all(rows[e.message_id]["sender_id"] == c.subject_id for e in c.evidence)
                    else "pending"
                )
                fingerprint = hashlib.sha256(
                    (scope + c.subject_id + c.summary).encode()
                ).hexdigest()
                if self.db.execute(
                    "SELECT 1 FROM memories WHERE fingerprint=?", (fingerprint,)
                ).fetchone():
                    continue
                mid, now = str(uuid.uuid4()), time.time()
                self.db.execute(
                    "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,1,?)",
                    (
                        mid,
                        scope,
                        c.subject_id,
                        c.summary,
                        c.kind,
                        c.stance,
                        status,
                        now,
                        now,
                        now + c.expires_in_days * 86400,
                        fingerprint,
                    ),
                )
                # Column count is asserted by unit tests against a fresh database.
                for e in c.evidence:
                    self.db.execute(
                        "INSERT OR IGNORE INTO sources VALUES(?,?,?)", (mid, e.message_id, e.quote)
                    )
                self.db.execute(
                    "INSERT INTO memory_fts VALUES(?,?)", (mid, " ".join(terms(c.summary)))
                )
                self._version(mid, "extractor")
                self.db.execute("INSERT OR REPLACE INTO outbox(memory_id) VALUES(?)", (mid,))
                ids.append(mid)
            self.db.executemany(
                "UPDATE messages SET processed=1 WHERE id=?", [(r["id"],) for r in batch]
            )
        return ids

    def _version(self, mid: str, actor: str):
        row = dict(self.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone())
        self.db.execute(
            "INSERT INTO versions(memory_id,version,snapshot,actor,created) VALUES(?,?,?,?,?)",
            (mid, row["version"], json.dumps(row, ensure_ascii=False), actor, time.time()),
        )

    def detail(self, mid: str, scope: str | None = None):
        row = self.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        if not row or (scope is not None and row["scope"] != scope):
            return None
        result = dict(row)
        result["sources"] = [
            dict(r)
            for r in self.db.execute(
                # Start with the handful of sources, not all messages in a busy
                # group merely to satisfy ORDER BY through msg_scope_time.
                "SELECT m.*,s.quote FROM sources s CROSS JOIN messages m ON m.id=s.message_id WHERE s.memory_id=? AND m.scope=? ORDER BY m.created",
                (mid, row["scope"]),
            )
        ]
        result["versions"] = [
            dict(r)
            for r in self.db.execute(
                "SELECT version,snapshot,actor,created FROM versions WHERE memory_id=? ORDER BY id DESC LIMIT 20",
                (mid,),
            )
        ]
        return result

    def source_context(self, mid: str, scope: str):
        memory = self.detail(mid, scope)
        if not memory or not memory["sources"]:
            return []
        found = {r["id"]: r for r in memory["sources"]}
        for r in memory["sources"]:
            for op, direction in (("<", "DESC"), (">", "ASC")):
                for n in self.db.execute(
                    f"SELECT * FROM messages WHERE scope=? AND (COALESCE(sent_at,created),created,id) {op} (?,?,?) ORDER BY COALESCE(sent_at,created) {direction},created {direction},id {direction} LIMIT 10",
                    (scope, message_time(r), r["created"], r["id"]),
                ):
                    found[n["id"]] = dict(n)
            if r["reply_id"]:
                n = self.db.execute(
                    "SELECT * FROM messages WHERE scope=? AND platform_id=?", (scope, r["reply_id"])
                ).fetchone()
                if n:
                    found[n["id"]] = dict(n)
        quotes = {r["id"]: r["quote"] for r in memory["sources"]}
        rows = [
            context_message(r, source=r["id"] in quotes, quote=quotes.get(r["id"], ""))
            for r in found.values()
        ]
        rows.sort(key=message_order)
        return bounded_context(rows, set(quotes))

    def list_memories(self, scope: str = "", status: str = "", query: str = "", offset: int = 0):
        where, args = ["1=1"], []
        for column, value in [("scope", scope), ("status", status)]:
            if value:
                where.append(column + "=?")
                args.append(value)
        if query:
            where.append("(summary LIKE ? OR subject_id=?)")
            args.extend(["%" + query[:200] + "%", query[:200]])
        clause = " AND ".join(where)
        count = self.db.execute("SELECT count(*) FROM memories WHERE " + clause, args).fetchone()[0]
        rows = self.db.execute(
            "SELECT * FROM memories WHERE " + clause + " ORDER BY updated DESC LIMIT 30 OFFSET ?",
            [*args, max(0, offset)],
        )
        return {"items": [dict(r) for r in rows], "total": count}

    def search(self, scope: str, query: str):
        ts = terms(query)
        if not ts:
            return []
        match = " OR ".join('"' + t + '"' for t in ts)
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT m.*,bm25(memory_fts) AS lexical_score FROM memory_fts f JOIN memories m ON m.id=f.id WHERE memory_fts MATCH ? AND m.scope=? AND m.status='active' AND m.expires>? ORDER BY bm25(memory_fts) LIMIT 36",
                (match, scope, time.time()),
            )
        ]

    def named_subjects(self, scope: str, query: str) -> list[str]:
        # Only names actually attached to source messages in this scope. Short
        # names and collisions are deliberately not resolved to a single person.
        rows = self.db.execute(
            "SELECT DISTINCT x.sender_id,x.sender_name FROM sources s "
            "CROSS JOIN messages x ON x.id=s.message_id WHERE x.scope=?",
            (scope,),
        )
        return list(
            dict.fromkeys(
                r[0]
                for r in rows
                if (len(r[0]) >= 3 and r[0] in query)
                or (len(r[1]) >= 2 and r[1].lower() in query.lower())
            )
        )[:20]

    def subject_memories(self, scope: str, subjects: list[str]):
        if not subjects:
            return []
        marks = ",".join("?" for _ in subjects[:20])
        return [
            dict(r)
            for r in self.db.execute(
                f"SELECT * FROM memories WHERE scope=? AND subject_id IN ({marks}) "
                "AND status='active' AND expires>? ORDER BY updated DESC LIMIT 24",
                [scope, *subjects[:20], time.time()],
            )
        ]

    def edit(self, mid: str, version: int, summary: str, status: str):
        if status not in {"active", "pending", "disabled"} or not 4 <= len(summary) <= 600:
            raise ValueError("无效状态或摘要长度")
        if safe_text(summary) != summary:
            raise ValueError("摘要疑似包含凭据")
        with self.db:
            row = self.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
            if not row:
                raise LookupError("记忆不存在")
            if row["version"] != version:
                raise ValueError("记忆已被更新，请刷新后重试")
            if (
                status == "active"
                and not self.db.execute(
                    "SELECT 1 FROM sources WHERE memory_id=?", (mid,)
                ).fetchone()
            ):
                raise ValueError("没有来源，不能启用")
            self.db.execute(
                "UPDATE memories SET summary=?,status=?,version=version+1,updated=? WHERE id=?",
                (summary, status, time.time(), mid),
            )
            self.db.execute("DELETE FROM memory_fts WHERE id=?", (mid,))
            self.db.execute("INSERT INTO memory_fts VALUES(?,?)", (mid, " ".join(terms(summary))))
            self._version(mid, "panel-admin")
            self.db.execute("INSERT OR REPLACE INTO outbox(memory_id) VALUES(?)", (mid,))
        return self.detail(mid)

    def delete(self, mid: str):
        with self.db:
            self.db.execute("UPDATE privacy_epoch SET value=value+1 WHERE id=1")
            self.db.execute("DELETE FROM memory_fts WHERE id=?", (mid,))
            self.db.execute("DELETE FROM memories WHERE id=?", (mid,))
            self.db.execute("INSERT OR REPLACE INTO outbox(memory_id) VALUES(?)", (mid,))
            # Traces may contain old summaries. Clear rather than retain deleted personal data.
            self.db.execute("DELETE FROM traces")

    def erase_subject(self, scope: str, subject: str):
        ids = [
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT m.id FROM memories m LEFT JOIN sources s ON s.memory_id=m.id LEFT JOIN messages x ON x.id=s.message_id WHERE m.scope=? AND (m.subject_id=? OR x.sender_id=?)",
                (scope, subject, subject),
            )
        ]
        for mid in ids:
            self.delete(mid)
        with self.db:
            self.db.execute("UPDATE privacy_epoch SET value=value+1 WHERE id=1")
            self.db.execute("DELETE FROM messages WHERE scope=? AND sender_id=?", (scope, subject))
            self.db.execute("DELETE FROM traces WHERE scope=?", (scope,))
        return len(ids)

    def record_call(self, stage: str = "", prompt_chars: int = 0):
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO budgets VALUES(?,0)", (day,))
            self.db.execute("UPDATE budgets SET calls=calls+1 WHERE day=?", (day,))
            if stage:
                cid = str(uuid.uuid4())
                self.db.execute(
                    "INSERT INTO model_calls(id,stage,status,created,prompt_chars) VALUES(?,?,'running',?,?)",
                    (cid, stage, time.time(), prompt_chars),
                )
                return cid

    def finish_call(self, cid, status, elapsed_ms, usage, error=""):
        with self.db:
            self.db.execute(
                "UPDATE model_calls SET status=?,elapsed_ms=?,input_tokens=?,output_tokens=?,"
                "cached_tokens=?,error=? WHERE id=?",
                (status, elapsed_ms, *usage, safe_text(error, 500), cid),
            )

    def usage_stats(self):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT stage,count(*) AS calls,sum(status='success') AS succeeded,"
                "sum(status NOT IN ('success','running')) AS failed,"
                "sum(status='running') AS running,count(input_tokens) AS usage_known,"
                "sum(input_tokens) AS input_tokens,avg(input_tokens) AS avg_input_tokens,"
                "sum(output_tokens) AS output_tokens,avg(elapsed_ms) AS avg_elapsed_ms "
                "FROM model_calls WHERE created>=? GROUP BY stage",
                (time.time() - 86400,),
            )
        ]

    def epoch(self):
        return self.db.execute("SELECT value FROM privacy_epoch WHERE id=1").fetchone()[0]

    def trace(
        self,
        scope: str,
        query: str,
        mode: str,
        reason: str,
        data: Any,
        elapsed_ms: int,
        expected_epoch: int | None = None,
    ):
        if expected_epoch is not None and self.epoch() != expected_epoch:
            return None
        tid = str(uuid.uuid4())
        with self.db:
            self.db.execute(
                "INSERT INTO traces VALUES(?,?,?,?,?,?,?,?)",
                (
                    tid,
                    scope,
                    safe_text(query, 500),
                    mode,
                    reason,
                    json.dumps(data, ensure_ascii=False),
                    elapsed_ms,
                    time.time(),
                ),
            )
            self.db.execute(
                "DELETE FROM traces WHERE id NOT IN (SELECT id FROM traces ORDER BY created DESC LIMIT 2000)"
            )
        return tid

    def traces(self, scope: str = "", mode: str = ""):
        if mode not in ("", "active", "preview"):
            raise ValueError("Invalid trace mode")
        rows = self.db.execute(
            "SELECT * FROM traces WHERE (?='' OR scope=?) AND (?='' OR mode=?) ORDER BY created DESC LIMIT 100",
            (scope, scope, mode, mode),
        )
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]

    def recall_stats(self, scope: str = ""):
        result = {
            "window_hours": 24,
            "real": {"calls": 0, "injected": 0},
            "preview": {"calls": 0, "injected": 0},
        }
        for row in self.db.execute(
            "SELECT mode,count(*) calls,sum(length(json_extract(data,'$.injection'))>0) injected "
            "FROM traces WHERE created>=? AND (?='' OR scope=?) AND mode IN ('active','preview') GROUP BY mode",
            (time.time() - 86400, scope, scope),
        ):
            result["real" if row["mode"] == "active" else "preview"] = {
                "calls": row["calls"],
                "injected": row["injected"] or 0,
            }
        return result

    def index_jobs(self):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM outbox WHERE retry_at<=? LIMIT 10", (time.time(),)
            )
        ]

    def reindex(self):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO outbox(memory_id) SELECT id FROM memories")
        return self.db.execute("SELECT count(*) FROM outbox").fetchone()[0]

    def index_done(self, mid: str, version: int | None):
        row = self.db.execute("SELECT version FROM memories WHERE id=?", (mid,)).fetchone()
        if (row[0] if row else None) == version:
            with self.db:
                self.db.execute("DELETE FROM outbox WHERE memory_id=?", (mid,))

    def index_failed(self, mid: str, attempts: int):
        with self.db:
            self.db.execute(
                "UPDATE outbox SET attempts=attempts+1,retry_at=? WHERE memory_id=?",
                (time.time() + min(3600, 30 * 2 ** min(attempts, 7)), mid),
            )

    def maintain(self, cfg: Settings):
        with self.db:
            self.db.execute(
                "DELETE FROM messages WHERE created<? AND id NOT IN (SELECT message_id FROM sources)",
                (time.time() - cfg.retention_days * 86400,),
            )
            self.db.execute(
                "DELETE FROM traces WHERE created<?", (time.time() - cfg.trace_days * 86400,)
            )
            self.db.execute("DELETE FROM budgets WHERE day<date('now','-30 day')")
            self.db.execute("DELETE FROM model_calls WHERE created<?", (time.time() - 30 * 86400,))
            self.db.execute(
                "DELETE FROM model_calls WHERE id IN "
                "(SELECT id FROM model_calls ORDER BY created DESC LIMIT -1 OFFSET 10000)"
            )
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def stats(self):
        day = time.strftime("%Y-%m-%d", time.gmtime())
        row = self.db.execute("SELECT calls FROM budgets WHERE day=?", (day,)).fetchone()
        failure = self.db.execute(
            "SELECT scope,extraction_error AS reason,extraction_attempted_at AS created "
            "FROM messages WHERE processed=3 ORDER BY extraction_attempted_at DESC LIMIT 1"
        ).fetchone()
        return {
            "messages": self.db.execute("SELECT count(*) FROM messages").fetchone()[0],
            "memories": self.db.execute("SELECT count(*) FROM memories").fetchone()[0],
            "active": self.db.execute(
                "SELECT count(*) FROM memories WHERE status='active' AND expires>?", (time.time(),)
            ).fetchone()[0],
            "pending": self.db.execute(
                "SELECT count(*) FROM memories WHERE status='pending'"
            ).fetchone()[0],
            "index_queue": self.db.execute("SELECT count(*) FROM outbox").fetchone()[0],
            "calls_today": row[0] if row else 0,
            "llm_usage_24h": self.usage_stats(),
            "recall_24h": self.recall_stats(),
            "extraction_failed_messages": self.db.execute(
                "SELECT count(*) FROM messages WHERE processed=3"
            ).fetchone()[0],
            "last_extraction_failure": dict(failure) if failure else None,
            "disk_bytes": self.size_bytes(),
        }
