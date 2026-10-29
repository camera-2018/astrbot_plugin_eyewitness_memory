import asyncio
import json
import sqlite3
import time
import uuid

import pytest

from eyewitness.models import Candidate, MaintenanceReview
from eyewitness.semantic_maintenance import apply, prepare
from eyewitness.store import Store
from tests.conftest import OTHER, SCOPE


async def enable(store):
    cfg = (await store.call("get_settings")).model_copy(
        update={"maintenance_enabled": True, "maintenance_semantic_enabled": True}
    )
    await store.call("save_settings", cfg)
    return cfg


async def add_memory(
    store,
    text="我喜欢红色键盘",
    summary="alice 喜欢红色键盘",
    *,
    scope=SCOPE,
    subject="alice",
    kind="preference",
    stance="self_report",
    sent_at=None,
    source_id=None,
):
    if source_id is None:
        source_id = await store.call(
            "capture", scope, uuid.uuid4().hex, subject, subject, text, sent_at=sent_at
        )
    batch = await store.call("source_rows", scope, [source_id])
    candidate = Candidate(
        summary=summary,
        subject_id=subject,
        kind=kind,
        stance=stance,
        evidence=[{"message_id": source_id, "quote": text}],
    )
    mid = (await store.call("save_extraction", batch, [candidate]))[0]
    if stance != "self_report":
        with store.db:
            store.db.execute("UPDATE memories SET status='active' WHERE id=?", (mid,))
    return mid, source_id


async def add_context(store, mid, *, scope=SCOPE, author="bob", text="你还喜欢红色键盘吗？"):
    context_id = await store.call(
        "capture", scope, uuid.uuid4().hex, author, author, text, sent_at=time.time() - 60
    )
    with store.db:
        store.db.execute("INSERT INTO memory_context VALUES(?,?)", (mid, context_id))
    return context_id


def decision(mid, source, quote="我喜欢红色键盘", **updates):
    return {
        "id": mid,
        "action": "forget",
        "category": "low_value",
        "reason": "这是一条临时状态",
        "evidence": [{"message_id": source, "quote": quote}],
        **updates,
    }


def review(*decisions):
    return MaintenanceReview.model_validate({"decisions": list(decisions)})


def stored(store, mid):
    return dict(store.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone())


def original_data(store, mid, source):
    return {
        "memory": stored(store, mid),
        "source": dict(store.db.execute("SELECT * FROM messages WHERE id=?", (source,)).fetchone()),
        "links": [
            tuple(r) for r in store.db.execute("SELECT * FROM sources WHERE memory_id=?", (mid,))
        ],
        "versions": [
            tuple(r) for r in store.db.execute("SELECT * FROM versions WHERE memory_id=?", (mid,))
        ],
    }


async def test_default_off_and_stale_enabled_config_never_prepare_work(store):
    await add_memory(store)
    off = await store.call("get_settings")
    assert prepare(store, off) is None
    enabled = off.model_copy(
        update={"maintenance_enabled": True, "maintenance_semantic_enabled": True}
    )
    assert prepare(store, enabled) is None  # A caller cannot bypass the current setting.
    assert not store.db.execute(
        "SELECT 1 FROM sqlite_master WHERE name='memory_semantic_checks'"
    ).fetchone()


async def test_prepare_scope_batch_limit_and_exclusions(store):
    cfg = await enable(store)
    mids = []
    for i in range(10):
        mid, _ = await add_memory(store, text=f"我喜欢颜色{i}", summary=f"alice 喜欢颜色{i}")
        mids.append(mid)
    foreign, _ = await add_memory(store, scope=OTHER, subject="eve", summary="eve 喜欢红色键盘")
    snap = prepare(store, cfg, limit=100, excluded=[mids[0]])
    assert snap["scope"] == SCOPE
    assert len(snap["targets"]) == 8 and mids[0] not in snap["targets"]
    assert foreign not in snap["records"]
    assert len(snap["records"]) <= 16
    assert all(r["memory"]["scope"] == SCOPE for r in snap["records"].values())
    assert snap["epoch"] == store.epoch()


async def test_prepare_presents_retained_context_separately_from_self_report_evidence(store):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    context_id = await add_context(store, mid)

    snap = prepare(store, cfg)

    payload = next(m for m in snap["payload"]["memories"] if m["id"] == mid)
    assert payload["subject_id"] == "alice"
    assert [s["message_id"] for s in payload["sources"]] == [source]
    assert payload["sources"][0]["author"] == "alice"
    assert payload["context"] == [
        {
            "message_id": context_id,
            "author": "bob",
            "time": snap["records"][mid]["context"][0]["sent_at"],
            "text": "你还喜欢红色键盘吗？",
            "role": "解释题干，不是本人的自述证据",
        }
    ]
    assert snap["records"][mid]["context"][0]["scope"] == SCOPE


@pytest.mark.parametrize("include_source", [False, True])
async def test_context_quote_cannot_be_used_as_forget_evidence(store, include_source):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    context_id = await add_context(store, mid)
    snap = prepare(store, cfg)
    before = original_data(store, mid, source)
    evidence = [{"message_id": context_id, "quote": "你还喜欢红色键盘吗？"}]
    if include_source:
        evidence.append({"message_id": source, "quote": "我喜欢红色键盘"})

    result = apply(store, cfg, snap, review(decision(mid, source, evidence=evidence)))

    assert result["forgotten"] == [] and result["uncertain"] == [mid]
    assert original_data(store, mid, source) == before
    assert store.db.execute(
        "SELECT 1 FROM memory_context WHERE memory_id=? AND message_id=?", (mid, context_id)
    ).fetchone()


@pytest.mark.parametrize("category", ["superseded", "duplicate"])
async def test_related_context_is_not_a_replacement_for_related_source_evidence(store, category):
    cfg = await enable(store)
    mid, source = await add_memory(store, sent_at=time.time() - 100)
    other, _ = await add_memory(
        store,
        text="我现在不喜欢红色键盘了",
        summary="alice 不再喜欢红色键盘",
        sent_at=time.time() - 50,
    )
    context_id = await add_context(store, other, text="是不是现在不喜欢红色键盘了？")
    snap = prepare(store, cfg, limit=1)
    before = stored(store, mid)

    result = apply(
        store,
        cfg,
        snap,
        review(
            decision(
                mid,
                source,
                category=category,
                related_id=other,
                evidence=[
                    {"message_id": source, "quote": "我喜欢红色键盘"},
                    {"message_id": context_id, "quote": "是不是现在不喜欢红色键盘了？"},
                ],
            )
        ),
    )

    assert result["forgotten"] == [] and result["uncertain"] == [mid]
    assert stored(store, mid) == before


@pytest.mark.parametrize("role", ["target", "reference"])
async def test_prepare_never_sends_cross_scope_retained_context(store, role):
    cfg = await enable(store)
    target, _ = await add_memory(store)
    if role == "reference":
        bad, _ = await add_memory(store, text="我喜欢绿色键盘", summary="alice 喜欢绿色键盘")
    else:
        bad = target
    await add_context(store, bad, scope=OTHER)

    snap = prepare(store, cfg, limit=1)

    if role == "target":
        assert snap is None
    else:
        assert snap["targets"] == [target]
        assert bad not in snap["records"]
        assert [m["id"] for m in snap["payload"]["memories"]] == [target]


@pytest.mark.parametrize(
    "change", ["text", "author", "scope", "sent_at", "created", "remove_link", "remove_message"]
)
async def test_changed_retained_context_cancels_the_entire_review(store, change):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    context_id = await add_context(store, mid)
    snap = prepare(store, cfg)
    before = original_data(store, mid, source)
    with store.db:
        if change == "text":
            store.db.execute("UPDATE messages SET text='改成了其他问题' WHERE id=?", (context_id,))
        elif change == "author":
            store.db.execute("UPDATE messages SET sender_id='eve' WHERE id=?", (context_id,))
        elif change == "scope":
            store.db.execute("UPDATE messages SET scope=? WHERE id=?", (OTHER, context_id))
        elif change in {"sent_at", "created"}:
            store.db.execute(f"UPDATE messages SET {change}={change}+10 WHERE id=?", (context_id,))
        elif change == "remove_link":
            store.db.execute("DELETE FROM memory_context WHERE memory_id=?", (mid,))
        else:
            store.db.execute("DELETE FROM messages WHERE id=?", (context_id,))

    with pytest.raises(ValueError, match="记忆或原文已变化"):
        apply(store, cfg, snap, review(decision(mid, source)))

    assert original_data(store, mid, source) == before
    assert store.db.execute("SELECT count(*) FROM memory_semantic_checks").fetchone()[0] == 0


async def test_context_added_after_snapshot_cancels_the_review(store):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    snap = prepare(store, cfg)
    before = original_data(store, mid, source)
    await add_context(store, mid)

    with pytest.raises(ValueError, match="记忆或原文已变化"):
        apply(store, cfg, snap, review(decision(mid, source)))

    assert original_data(store, mid, source) == before
    assert store.db.execute("SELECT count(*) FROM memory_semantic_checks").fetchone()[0] == 0


async def test_forget_is_soft_and_keeps_original_versions_sources_and_expiry(store):
    cfg = await enable(store)
    mid, source = await add_memory(store, stance="hearsay")
    snap = prepare(store, cfg)
    before = original_data(store, mid, source)
    result = apply(store, cfg, snap, review(decision(mid, source)))
    after = original_data(store, mid, source)
    assert result == {"reviewed": 1, "forgotten": [mid], "uncertain": []}
    assert after["memory"]["status"] == "disabled"
    assert after["memory"]["version"] == before["memory"]["version"] + 1
    for key in (
        "scope",
        "subject_id",
        "summary",
        "kind",
        "stance",
        "expires",
        "created",
        "fingerprint",
    ):
        assert after["memory"][key] == before["memory"][key]
    assert after["source"] == before["source"] and after["links"] == before["links"]
    assert after["versions"][:-1] == before["versions"]
    assert json.loads(after["versions"][-1][3]) == after["memory"]
    assert after["versions"][-1][4].startswith("scheduled-semantic:low_value:")
    assert store.db.execute("SELECT 1 FROM outbox WHERE memory_id=?", (mid,)).fetchone()
    assert store.db.execute("SELECT count(*) FROM model_calls").fetchone()[0] == 0


@pytest.mark.parametrize(
    "bad_kind", ["unknown_id", "duplicate_id", "reference_only", "credential_reason"]
)
async def test_invalid_review_rejects_entire_batch_without_partial_forgetting(store, bad_kind):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    ref, ref_source = await add_memory(store, text="我喜欢绿色键盘", summary="alice 喜欢绿色键盘")
    snap = prepare(store, cfg, limit=1)
    good = decision(mid, source)
    if bad_kind == "unknown_id":
        bad = decision(str(uuid.uuid4()), source)
    elif bad_kind == "duplicate_id":
        bad = decision(mid, source)
    elif bad_kind == "reference_only":
        assert ref in snap["records"] and ref not in snap["targets"]
        bad = decision(ref, ref_source, "我喜欢绿色键盘")
    else:
        bad = decision(ref, ref_source, "我喜欢绿色键盘", reason="password=not-a-real-test-secret")
        snap = prepare(store, cfg)
    before = original_data(store, mid, source)
    with pytest.raises(ValueError):
        apply(store, cfg, snap, review(good, bad))
    assert original_data(store, mid, source) == before
    assert store.db.execute("SELECT count(*) FROM memory_semantic_checks").fetchone()[0] == 0


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [{"message_id": "outside", "quote": "我喜欢红色键盘"}],
        [{"message_id": None, "quote": "不存在的原话"}],
    ],
)
async def test_no_verbatim_target_source_cannot_forget(store, evidence):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    evidence = [{**e, "message_id": e["message_id"] or source} for e in evidence]
    snap = prepare(store, cfg)
    before = stored(store, mid)
    result = apply(store, cfg, snap, review(decision(mid, source, evidence=evidence)))
    assert result["forgotten"] == [] and result["uncertain"] == [mid]
    assert stored(store, mid) == before


@pytest.mark.parametrize(
    "change",
    [
        "delete",
        "edit",
        "source_text",
        "source_author",
        "source_quote",
        "epoch",
        "config",
        "disabled",
    ],
)
async def test_changed_data_or_actual_configuration_cancels_snapshot(store, change):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    snap = prepare(store, cfg)
    if change == "delete":
        await store.call("delete", mid)
    elif change == "edit":
        await store.call("edit", mid, 1, "alice 后来换了另一个键盘", "active")
    elif change == "config":
        await store.call(
            "save_settings", cfg.model_copy(update={"maintenance_semantic_enabled": False})
        )
    else:
        with store.db:
            if change == "source_text":
                store.db.execute("UPDATE messages SET text='已改变的原文' WHERE id=?", (source,))
            elif change == "source_author":
                store.db.execute(
                    "UPDATE messages SET sender_id='someone_else' WHERE id=?", (source,)
                )
            elif change == "source_quote":
                store.db.execute("UPDATE sources SET quote='红色' WHERE memory_id=?", (mid,))
            elif change == "epoch":
                store.db.execute("UPDATE privacy_epoch SET value=value+1")
            elif change == "disabled":
                store.db.execute("UPDATE memories SET status='disabled' WHERE id=?", (mid,))
    with pytest.raises(ValueError):
        apply(store, cfg, snap, review(decision(mid, source)))
    assert store.db.execute("SELECT count(*) FROM memory_semantic_checks").fetchone()[0] == 0


@pytest.mark.parametrize(
    "bad_source", ["cross_group", "wrong_self_report_author", "bot", "missing", "quote_not_present"]
)
async def test_prepare_does_not_send_invalid_or_cross_group_sources(store, bad_source):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    with store.db:
        if bad_source == "cross_group":
            store.db.execute("UPDATE messages SET scope=? WHERE id=?", (OTHER, source))
        elif bad_source == "wrong_self_report_author":
            store.db.execute("UPDATE messages SET sender_id='someone_else' WHERE id=?", (source,))
        elif bad_source == "missing":
            store.db.execute("DELETE FROM sources WHERE memory_id=?", (mid,))
        elif bad_source == "quote_not_present":
            store.db.execute("UPDATE sources SET quote='没有依据的片段' WHERE memory_id=?", (mid,))
    if bad_source == "bot":
        cfg = cfg.model_copy(update={"bot_ids": ["alice"]})
        await store.call("save_settings", cfg)
    assert prepare(store, cfg) is None


async def test_invalid_first_group_does_not_starve_another_allowed_group(store):
    cfg = await enable(store)
    bad, _ = await add_memory(store)
    with store.db:
        store.db.execute("DELETE FROM sources WHERE memory_id=?", (bad,))
    good, _ = await add_memory(store, scope=OTHER, subject="eve", summary="eve 喜欢红色键盘")
    snap = prepare(store, cfg)
    assert snap["scope"] == OTHER and snap["targets"] == [good]


@pytest.mark.parametrize("mismatch", ["scope", "subject", "kind", "stance"])
async def test_cross_scope_person_kind_or_stance_is_not_a_replacement(store, mismatch):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    options = (
        {"scope": OTHER}
        if mismatch == "scope"
        else {"subject": "eve"}
        if mismatch == "subject"
        else {"kind": "goal"}
        if mismatch == "kind"
        else {"stance": "hearsay"}
    )
    other, other_source = await add_memory(
        store, text="我现在喜欢绿色键盘", summary="后来喜欢绿色键盘", **options
    )
    snap = prepare(store, cfg, limit=1)
    result = apply(
        store,
        cfg,
        snap,
        review(
            decision(
                mid,
                source,
                category="superseded",
                related_id=other,
                evidence=[
                    {"message_id": source, "quote": "我喜欢红色键盘"},
                    {"message_id": other_source, "quote": "我现在喜欢绿色键盘"},
                ],
            )
        ),
    )
    assert result["forgotten"] == [] and result["uncertain"] == [mid]


@pytest.mark.parametrize("newer", [True, False])
async def test_superseded_requires_later_original_source_time(store, newer):
    cfg = await enable(store)
    now = time.time()
    mid, source = await add_memory(store, sent_at=now - 100)
    other, other_source = await add_memory(
        store,
        text="我现在不喜欢红色键盘了",
        summary="alice 不再喜欢红色键盘",
        sent_at=now - (50 if newer else 200),
    )
    snap = prepare(store, cfg, limit=1)
    result = apply(
        store,
        cfg,
        snap,
        review(
            decision(
                mid,
                source,
                category="superseded",
                related_id=other,
                evidence=[
                    {"message_id": source, "quote": "我喜欢红色键盘"},
                    {"message_id": other_source, "quote": "我现在不喜欢红色键盘了"},
                ],
            )
        ),
    )
    assert result["forgotten"] == ([mid] if newer else [])
    assert stored(store, other)["status"] == "active"


async def test_replacement_proposed_for_forgetting_cannot_delete_both_memories(store):
    cfg = await enable(store)
    mid, source = await add_memory(store, stance="hearsay", sent_at=time.time() - 100)
    other, other_source = await add_memory(
        store,
        text="我现在不喜欢红色键盘了",
        summary="alice 不再喜欢红色键盘",
        sent_at=time.time() - 50,
        stance="hearsay",
    )
    snap = prepare(store, cfg)
    result = apply(
        store,
        cfg,
        snap,
        review(
            decision(
                mid,
                source,
                category="superseded",
                related_id=other,
                evidence=[
                    {"message_id": source, "quote": "我喜欢红色键盘"},
                    {"message_id": other_source, "quote": "我现在不喜欢红色键盘了"},
                ],
            ),
            decision(other, other_source, "我现在不喜欢红色键盘了"),
        ),
    )
    assert mid not in result["forgotten"] and mid in result["uncertain"]
    assert other in result["forgotten"]


@pytest.mark.parametrize("shared_source", [True, False])
async def test_independent_events_are_not_deduplicated_just_for_similar_summary(
    store, shared_source
):
    cfg = await enable(store)
    text = "我今天参加了绘画比赛"
    mid, source = await add_memory(store, text=text, summary="alice 参加绘画比赛", kind="event")
    other, other_source = await add_memory(
        store,
        text=text,
        summary="alice 有一次绘画比赛经历",
        kind="event",
        source_id=source if shared_source else None,
    )
    snap = prepare(store, cfg, limit=1)
    evidence = [{"message_id": source, "quote": text}]
    if not shared_source:
        evidence.append({"message_id": other_source, "quote": text})
    result = apply(
        store,
        cfg,
        snap,
        review(decision(mid, source, category="duplicate", related_id=other, evidence=evidence)),
    )
    assert result["forgotten"] == ([mid] if shared_source else [])


async def test_record_expiring_during_model_wait_is_not_modified_or_renewed(store, monkeypatch):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    snap = prepare(store, cfg)
    before = stored(store, mid)
    monkeypatch.setattr("eyewitness.semantic_maintenance.time.time", lambda: before["expires"] + 1)
    result = apply(store, cfg, snap, review(decision(mid, source)))
    assert result["forgotten"] == [] and stored(store, mid) == before


async def test_keep_and_missing_decision_change_only_review_checkpoint(store):
    cfg = await enable(store)
    mid, source = await add_memory(store)
    other, other_source = await add_memory(
        store, text="我喜欢绿色键盘", summary="alice 喜欢绿色键盘"
    )
    snap = prepare(store, cfg)
    before = {m: original_data(store, m, s) for m, s in [(mid, source), (other, other_source)]}
    result = apply(
        store,
        cfg,
        snap,
        review(decision(mid, source, action="keep", category="useful", evidence=[])),
    )
    assert result["forgotten"] == [] and result["uncertain"] == [other]
    assert before == {
        m: original_data(store, m, s) for m, s in [(mid, source), (other, other_source)]
    }
    assert prepare(store, cfg) is None


async def test_version_write_failure_rolls_back_disable_and_audit(store, monkeypatch):
    cfg = await enable(store)
    mid, source = await add_memory(store, stance="hearsay")
    snap = prepare(store, cfg)
    before = original_data(store, mid, source)

    def fail(*args):
        raise sqlite3.OperationalError("injected write failure")

    monkeypatch.setattr(store, "_version", fail)
    with pytest.raises(sqlite3.OperationalError):
        apply(store, cfg, snap, review(decision(mid, source)))
    assert original_data(store, mid, source) == before
    assert store.db.execute("SELECT count(*) FROM memory_semantic_checks").fetchone()[0] == 0


async def test_configuration_change_at_commit_boundary_rolls_back_everything(store, monkeypatch):
    cfg = await enable(store)
    mid, source = await add_memory(store, stance="hearsay")
    snap = prepare(store, cfg)
    before = original_data(store, mid, source)
    calls = 0

    def current_settings():
        nonlocal calls
        calls += 1
        return cfg if calls == 1 else cfg.model_copy(update={"maintenance_semantic_enabled": False})

    monkeypatch.setattr(store, "get_settings", current_settings)
    with pytest.raises(ValueError):
        apply(store, cfg, snap, review(decision(mid, source)))
    assert original_data(store, mid, source) == before


async def test_concurrent_replay_only_one_transaction_can_disable(store):
    cfg = await enable(store)
    mid, source = await add_memory(store, stance="hearsay")
    snap = prepare(store, cfg)
    second = Store(store.path)
    second.db = sqlite3.connect(store.path, timeout=5, check_same_thread=False)
    second.db.row_factory = sqlite3.Row
    try:
        outcomes = await asyncio.gather(
            asyncio.to_thread(apply, store, cfg, snap, review(decision(mid, source))),
            asyncio.to_thread(apply, second, cfg, snap, review(decision(mid, source))),
            return_exceptions=True,
        )
        assert sum(isinstance(x, dict) and x["forgotten"] == [mid] for x in outcomes) == 1
        assert sum(isinstance(x, ValueError) for x in outcomes) == 1
        assert stored(store, mid)["version"] == 2
        assert (
            store.db.execute("SELECT count(*) FROM versions WHERE memory_id=?", (mid,)).fetchone()[
                0
            ]
            == 2
        )
    finally:
        second.db.close()


@pytest.mark.parametrize("kind", ["event", "preference", "goal", "agreement"])
async def test_explicit_self_report_cannot_be_forgotten_as_merely_low_value(store, kind):
    cfg = await enable(store)
    text = "我前两天刚续了90天麦金卡"
    mid, source = await add_memory(store, text=text, summary="两天前续费90天餐饮会员", kind=kind)
    snap = prepare(store, cfg)
    before = original_data(store, mid, source)
    result = apply(
        store, cfg, snap, review(decision(mid, source, text, reason="琐碎日常消费，价值不高"))
    )
    assert result["forgotten"] == [] and result["uncertain"] == [mid]
    assert original_data(store, mid, source) == before
    row = store.db.execute(
        "SELECT action,reason FROM memory_semantic_checks WHERE memory_id=?", (mid,)
    ).fetchone()
    assert row[0] == "uncertain" and "不能仅凭价值评价" in row[1]
