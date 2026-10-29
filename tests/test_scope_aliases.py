import json
import time
import uuid

import pytest

from eyewitness.context import context_message
from eyewitness.engine import Engine
from eyewitness.extraction import compact_message
from eyewitness.models import Settings
from eyewitness.scope_aliases import (
    MESSAGE_KEY_PREFIX,
    canonical_scope,
    decode_message_key,
    normalize_scope_aliases,
    parse_scope_aliases,
    platform_message_key,
)
from tests.conftest import OTHER, SCOPE, seed
from tests.test_engine import Model, answers, planning_passthrough

OFFICIAL = "qq_official:GroupMessage:GROUP_OPENID"
SECOND = "qq_second:GroupMessage:OTHER_GROUP_OPENID"
RULES = [f"{OFFICIAL}={SCOPE}", f"{SECOND}={SCOPE}"]


@pytest.mark.parametrize(
    "entries",
    [
        [""],
        [SCOPE],
        [f"{OFFICIAL}={SCOPE}=extra"],
        [f"{OFFICIAL}={OFFICIAL}"],
        [f"{OFFICIAL}=default:FriendMessage:123"],
        [f"qq:FriendMessage:abc={SCOPE}"],
        [f"{OFFICIAL}={SCOPE}", f"{OFFICIAL}={OTHER}"],
        [f"{OFFICIAL}={SCOPE}", f"{SCOPE}={OTHER}"],
        [f"{OFFICIAL}={SCOPE}", f"{SCOPE}={OFFICIAL}"],
        [f"qq:GroupMessage:bad id={SCOPE}"],
        [None],
    ],
)
def test_invalid_scope_aliases_fail_closed(entries):
    with pytest.raises(ValueError):
        parse_scope_aliases(entries)
    with pytest.raises(ValueError):
        Settings(scope_aliases=entries)


def test_direct_many_to_one_aliases_are_explicit_and_normalized():
    entries = [f" {OFFICIAL} = {SCOPE} ", f"{OFFICIAL}={SCOPE}", RULES[1]]
    assert normalize_scope_aliases(entries) == RULES
    assert canonical_scope(OFFICIAL, entries) == SCOPE
    assert canonical_scope(SECOND, entries) == SCOPE
    assert canonical_scope(SCOPE, entries) == SCOPE
    assert canonical_scope(OTHER, entries) == OTHER
    assert canonical_scope("qq:FriendMessage:123", entries) == "qq:FriendMessage:123"


def test_transport_keys_round_trip_and_escape_reserved_prefix():
    assert platform_message_key(SCOPE, SCOPE, "42") == "42"
    alias = platform_message_key(OFFICIAL, SCOPE, "42")
    assert alias != "42"
    assert decode_message_key(alias, SCOPE) == (OFFICIAL, "42")
    # A genuine canonical ID spelling an encoded alias cannot overwrite that alias.
    literal = platform_message_key(SCOPE, SCOPE, alias)
    assert literal != alias
    assert decode_message_key(literal, SCOPE) == (SCOPE, alias)
    assert platform_message_key(SECOND, SCOPE, "42") != alias
    assert decode_message_key(MESSAGE_KEY_PREFIX + "invalid!", SCOPE) == (
        SCOPE,
        MESSAGE_KEY_PREFIX + "invalid!",
    )
    assert decode_message_key(None, SCOPE) == (SCOPE, None)
    assert decode_message_key("", SCOPE) == (SCOPE, "")


async def configure(store, aliases=RULES, allowed=None):
    cfg = await store.call("get_settings")
    await store.call(
        "save_settings",
        cfg.model_copy(update={"scope_aliases": aliases, "allowed_scopes": allowed or [SCOPE]}),
    )


async def test_capture_and_replies_share_group_but_never_collide(store):
    await configure(store)
    ids = []
    for scope, sender, name in [(SCOPE, "123456", "OneBot"), (OFFICIAL, "USER_OPENID", "Official")]:
        ids.append(await store.call("capture", scope, "42", sender, name, f"{name} 原消息"))
        await store.call("capture", scope, "43", sender, name, f"{name} 回复", reply_id="42")
    assert len(set(ids)) == 2
    assert ids[0] == str(uuid.uuid5(uuid.NAMESPACE_URL, SCOPE + ":42"))
    assert ids[1] == str(uuid.uuid5(uuid.NAMESPACE_URL, OFFICIAL + ":42"))
    canonical = await store.call("referenced_message", SCOPE, "42")
    official = await store.call("referenced_message", OFFICIAL, "42")
    assert canonical["id"] == ids[0] and official["id"] == ids[1]
    assert canonical["scope"] == official["scope"] == SCOPE
    assert canonical["sender_id"] == "123456"
    assert official["sender_id"] == "USER_OPENID"  # No nickname-based identity merge.
    assert await store.call("referenced_message", SECOND, "42") is None
    for scope, expected in [(SCOPE, ids[0]), (OFFICIAL, ids[1])]:
        reply = await store.call("referenced_message", scope, "43")
        referenced = await store.call(
            "referenced_message", SCOPE, reply["reply_id"], stored_key=True
        )
        assert referenced["id"] == expected
    display = context_message(official)
    assert display["origin_scope"] == OFFICIAL
    assert display["platform_message_id"] == "42"
    display_reply = context_message(await store.call("referenced_message", OFFICIAL, "43"))
    assert display_reply["reply_id"] == "42"
    assert MESSAGE_KEY_PREFIX not in json.dumps(display_reply)
    assert compact_message(official)["origin_scope"] == OFFICIAL


async def test_legacy_literal_reserved_message_id_and_alias_do_not_collide(store):
    await configure(store)
    encoded_alias = platform_message_key(OFFICIAL, SCOPE, "42")
    first = await store.call("capture", OFFICIAL, "42", "openid", "A", "别名消息")
    second = await store.call("capture", SCOPE, encoded_alias, "123", "B", "真实ID恰好像内部键")
    assert first != second
    assert (await store.call("referenced_message", OFFICIAL, "42"))["id"] == first
    assert (await store.call("referenced_message", SCOPE, encoded_alias))["id"] == second


async def test_extraction_reply_expansion_preserves_origin(store):
    await configure(store)
    now = time.time()
    wrong = await store.call(
        "capture", SCOPE, "42", "123", "Old", "另一平台的同ID消息", created=now - 5000
    )
    correct = await store.call(
        "capture", OFFICIAL, "42", "openid", "New", "被引用的原始消息", created=now - 5000
    )
    reply = await store.call(
        "capture", OFFICIAL, "43", "openid", "New", "我回复旧消息", reply_id="42", created=now
    )
    batch = await store.call("source_rows", SCOPE, [reply])
    context = await store.call("extraction_context", batch)
    assert {r["id"] for r in context} == {correct}
    assert wrong not in {r["id"] for r in context}


async def test_alias_requires_whitelisted_canonical_and_never_enables_private(store):
    await configure(store, allowed=[OTHER])
    assert await store.call("capture", OFFICIAL, "1", "openid", "A", "不应采集") is None
    assert await store.call("capture", "qq:FriendMessage:abc", "1", "openid", "A", "私聊") is None
    model = Model([])
    result = await Engine(store, model).recall(OFFICIAL, "绘画比赛")
    assert result["reason"] == "作用域未启用"
    assert not result["injection"] and not model.calls


async def test_official_recall_finds_existing_onebot_memory(store):
    await configure(store)
    target, source = await seed(store)
    result = await Engine(store, Model(answers(target, source))).recall(
        OFFICIAL, "绘画比赛", "USER_OPENID"
    )
    assert result["selected"][0]["id"] == target
    assert result["selected"][0]["subject_id"] == "alice"
    assert result["injection"]


async def test_official_recall_resolves_only_official_quote(store):
    await configure(store)
    target, source = await seed(store)
    await store.call("capture", SCOPE, "42", "123", "Old", "同号的 OneBot 错误引用")
    await store.call("capture", OFFICIAL, "42", "openid", "New", "官方引用绘画比赛")
    prompts = []

    @planning_passthrough
    async def generate(provider, system, prompt):
        prompts.append(json.loads(prompt)["data"])
        return json.dumps(answers(target, source)[0], ensure_ascii=False)

    result = await Engine(store, generate).recall(OFFICIAL, "后来呢", "openid", reply_id="42")
    assert result["injection"]
    assert prompts and all(p["quoted_message"]["text"] == "官方引用绘画比赛" for p in prompts)
    assert all(p["reply_id"] == "42" for p in prompts)


async def test_mapping_change_while_model_waits_revokes_old_group_result(store):
    await configure(store)
    target, source = await seed(store)

    @planning_passthrough
    async def generate(provider, system, prompt):
        cfg = await store.call("get_settings")
        await store.call(
            "save_settings",
            cfg.model_copy(
                update={"scope_aliases": [f"{OFFICIAL}={OTHER}"], "allowed_scopes": [SCOPE, OTHER]}
            ),
        )
        return json.dumps(answers(target, source)[0], ensure_ascii=False)

    result = await Engine(store, generate).recall(OFFICIAL, "绘画比赛", "openid")
    assert not result["injection"] and not result["selected"]
    assert result["reason"] == "运行期间作用域或群映射已改变，结果已撤销"


async def test_admin_queries_and_privacy_erase_resolve_same_group(store):
    await configure(store, allowed=[SCOPE, OTHER])
    target, source = await seed(store)
    foreign, _ = await seed(store, scope=OTHER)
    assert (await store.call("detail", target, OFFICIAL))["id"] == target
    assert (await store.call("list_memories", OFFICIAL))["total"] == 1
    assert await store.call("source_context", target, OFFICIAL)
    assert [m["id"] for m in await store.call("search", OFFICIAL, "绘画比赛")] == [target]
    assert await store.call("erase_subject", OFFICIAL, "alice") == 1
    assert await store.call("detail", target) is None
    assert await store.call("detail", foreign) is not None
    assert await store.call("source_rows", SCOPE, [source]) == []
