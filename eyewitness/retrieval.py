"""Bounded conversation hints and balanced lexical/semantic candidate ranking."""

from __future__ import annotations

import re
from copy import deepcopy

from .models import low_signal, safe_text, terms

# These words indicate a conversational request, not a concrete remembered topic.
GENERIC_TERMS = set(
    terms(
        "之前 上次 以前 后来 现在 今天 明天 昨天 最近 时候 时间 什么 怎么 为什么 怎么样 "
        "一下 一个 这个 那个 这些 那些 这里 那里 可以 还是 不是 已经 然后 就是 因为 所以 "
        "没有 有没有 觉得 知道 记得 记忆 说过 提过 表示 自述 计划 喜欢 打算 准备 "
        "帮我 看看 告诉 说说 你们 我们 大家 群友 自己 目前 还有 事情"
    )
)

HISTORY_REFERENCE = re.compile(
    r"之前|以前|上次|记得|记忆|历史|习惯|偏好|原来|那[个次台份]|照旧|平时|曾经|当时|后来|上回"
)
FOLLOWUP = re.compile(
    r"^(?:教我|教教我|教一下|怎么弄|怎么做|怎么学|咋弄|咋办|那怎么办|"
    r"细说|讲讲|说说看|详细说说|展开说说|继续|然后呢|我也想学)$"
)


def needs_topic_context(query: str) -> bool:
    plain = re.sub(r"[\s，,。.!！?？~～…、]+", "", query)
    return bool(FOLLOWUP.fullmatch(plain))


def followup_topic(conversation: list[dict]) -> str:
    """Use the latest conversation topic, never an unrelated group interjection."""
    for row in reversed(conversation):
        text = row["text"][:400]
        if needs_topic_context(text) or not (set(terms(text)) - GENERIC_TERMS):
            continue
        if not low_signal(text):
            return text
    return ""


DIRECT_OPERATION = re.compile(
    r"(?:^|[，,。\s]|帮我|给我)(?:请)?(?:打开|关闭|开启|关掉|暂停|播放|重启|停止|执行|运行|"
    r"安装|卸载|删除|翻译|朗读|设闹钟|设置闹钟|调到|调成)"
    r"|把.{1,40}(?:打开|关闭|关掉|关了|开到|调到|调成|删除|重启)"
    r"|(?:看看|检查).{0,12}(?:电视|空调|灯|插座|风扇).{0,12}(?:开了|关了|状态)"
)


def direct_operation(query: str) -> bool:
    """Complete operational commands need live tools, not incidental old facts.

    Historical references deliberately escape this shortcut ("照上次的设置打开").
    This does not gate person questions, recommendations or implicit topic recall.
    """
    return not HISTORY_REFERENCE.search(query) and bool(DIRECT_OPERATION.search(query))


# These are deliberately narrow shapes, not a blanket "no history keyword"
# filter: recommendations, person questions and concrete project discussions
# remain eligible even when the user did not explicitly ask to remember.
FEEDBACK = re.compile(
    r"^(?:有道理|没道理|不是|确实如此|降智了|何意味|你闹麻了|你个[人入]机|"
    r"说话|说点话|多说几句|别催我|别说了|少说点话)(?:[\W_]|啊|呀|呢|吧)*$",
    re.IGNORECASE,
)
TEMPORARY_STATE = re.compile(
    r"^我?(?:(?:好|很|太|有点)?(?:饿|热|冷|困|累|无聊)(?:了|啊|呀|呢|死了)?"
    r"(?:也|又|还|而且|并且)?){1,5}$"
)
PRESENTATION = re.compile(
    r"^(?:必须|全部|都|请|要|用)?(?:发|说|讲|回)?(?:语音|声音)(?:回复|回答|解释)?$"
    r"|^你(?:可以|能|能不能|可不可以)发出.{0,30}(?:声|声音)(?:吗|么)?$"
)
REFERENTIAL = re.compile(r"他|她|它|[这那](?:台|件|份|次|个(?!群|词|字|人类))")
BACKGROUND_REQUEST = re.compile(
    r"推荐|适合我|我该买|我应该买|生日|项目|插件|仓库|代码|设备|服务器|进展|进度"
)
PERSONAL_REFERENCE = re.compile(r"我(?:的|喜欢|想|要|打算|准备|住|用|学|工作)")
GENERAL_QUESTION = re.compile(
    r"(?:是什么|是啥|什么意思)|^(?:什么|啥)(?:是|叫)"
    r"|^(?:你)?(?:知道|了解|听说过).{0,60}(?:吗|么)(?:我问问)?$"
    r"|^(?:人类|人们|大家)为什么"
    r"|^(?:永远|总是|老是|一直)?(?:睡不醒|睡不着|犯困|无聊)(?:要|该)?(?:怎么办|咋办)$"
)


def ordinary_chat_reason(query: str, *, has_subject: bool = False, has_quote: bool = False) -> str:
    """Skip obvious transient chatter/factual questions before paid retrieval.

    A source-backed person name or explicit history reference always escapes.
    General questions also escape when quoted or referring to a concrete prior
    object, or asking about the speaker's own background. A quoted format
    request may also refer to that topic. Pure feedback does not need a new
    long-memory review, even when replying to a message.
    """
    if has_subject or HISTORY_REFERENCE.search(query):
        return ""
    plain = re.sub(r"[\s，,。.!！?？~～…、【】]+", "", query)
    if FEEDBACK.fullmatch(plain):
        return "feedback"
    if TEMPORARY_STATE.fullmatch(plain):
        return "temporary_state"
    if not has_quote and PRESENTATION.search(plain):
        return "presentation"
    if (
        not has_quote
        and not REFERENTIAL.search(plain)
        and not BACKGROUND_REQUEST.search(plain)
        and not PERSONAL_REFERENCE.search(plain)
    ):
        if GENERAL_QUESTION.search(plain):
            return "general_question"
    return ""


def compact_review(data):
    """Request-local source aliases avoid repeatedly tokenizing UUIDs.

    Real IDs remain in storage and injected snapshots; aliases never escape this
    one verification request. Unknown returned IDs still fail source validation.
    """
    payload = deepcopy(data)
    source_ids = {f"s{i}": row["id"] for i, row in enumerate(payload["messages"], 1)}
    memory_ids = {f"m{i}": row["id"] for i, row in enumerate(payload["candidates"], 1)}
    sources = {value: key for key, value in source_ids.items()}
    memories = {value: key for key, value in memory_ids.items()}
    for message in payload["messages"]:
        message["id"] = sources[message["id"]]
    for memory in payload["candidates"]:
        memory["id"] = memories[memory["id"]]
        for field in ("source_ids", "context_ids"):
            memory[field] = [sources[mid] for mid in memory[field]]
    return payload, memory_ids, source_ids


def rank_candidates(candidates, query: str, subjects: list[str], semantic_min: float):
    """Cheap eligibility/ranking only. Scores never authorize injection."""
    wanted = set(terms(query)) - GENERIC_TERMS
    ranked, skipped = [], []
    for rank, memory in enumerate(candidates):
        overlap = wanted.intersection(terms(memory["summary"]))
        subject_match = memory["subject_id"] in subjects
        score = memory.get("semantic_score")
        semantic_match = isinstance(score, (float, int)) and score >= semantic_min
        signals = {
            "topic_terms": sorted(overlap)[:12],
            "subject_match": subject_match,
            "semantic_score": score,
            "lexical_score": memory.get("lexical_score"),
        }
        if not (overlap or subject_match or semantic_match):
            skipped.append({"id": memory["id"], "reason": "仅泛词或弱语义匹配", **signals})
            continue
        priority = (
            1.2 * subject_match
            + min(len(overlap), 6) * 0.18
            + (score if isinstance(score, (int, float)) else 0)
            + 0.15 / (rank + 1)
        )
        ranked.append((priority, {**memory, "signals": signals}))
    ranked.sort(key=lambda item: -item[0])
    return [memory for _, memory in ranked], skipped


def conversation_hint(messages, limit: int = 3600) -> list[dict[str, str]]:
    """Copy recent user/assistant text, never media, tools or earlier injected memories."""
    result = []
    remaining = limit
    for message in reversed(list(messages or [])[-6:]):

        def field(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        role = field(message, "role")
        if role not in ("user", "assistant"):
            continue
        content = field(message, "content")
        if isinstance(content, list):
            content = "\n".join(
                field(part, "text")
                for part in content
                if field(part, "type") == "text" and isinstance(field(part, "text"), str)
            )
        if not isinstance(content, str):
            continue
        # Memory is appended at the user tail. Do not use it to establish relevance again.
        content = re.split(r"\[历史记忆参考|\[群聊记忆:", content, maxsplit=1)[0]
        text = safe_text(content.strip(), min(1200, remaining))
        if text:
            result.append({"role": role, "text": text})
            remaining -= len(text)
        if remaining <= 0:
            break
    return list(reversed(result))


def merge_rankings(*rankings: list[dict]) -> list[dict]:
    """Reciprocal rank fusion: neither retrieval path consumes all slots first."""
    records, scores = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            mid = item["id"]
            if mid in seen:
                continue
            seen.add(mid)
            records[mid] = {**records.get(mid, {}), **item}
            scores[mid] = scores.get(mid, 0) + 1 / (60 + rank)
    return [records[mid] for mid in sorted(records, key=lambda mid: -scores[mid])]
