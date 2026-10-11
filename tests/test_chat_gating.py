import pytest

from eyewitness.engine import Engine
from eyewitness.retrieval import ordinary_chat_reason
from tests.conftest import SCOPE, seed
from tests.test_engine import Model, answers


@pytest.mark.parametrize(
    "query,reason",
    [
        ("有道理", "feedback"),
        ("不是", "feedback"),
        ("说话🗣️", "feedback"),
        ("降智了", "feedback"),
        ("你闹麻了", "feedback"),
        ("我好饿", "temporary_state"),
        ("我好热也好饿", "temporary_state"),
        ("我好热，也好饿又无聊", "temporary_state"),
        ("必须发语音", "presentation"),
        ("你可以发出【睦头叹气声】吗", "presentation"),
        ("你知道啤酒烧烤是什么吗", "general_question"),
        ("人类为什么喜欢看百合小说用语音解释", "general_question"),
        ("永远睡不醒怎么办", "general_question"),
    ],
)
async def test_casual_shapes_use_no_embedding_jev_or_gemini(store, query, reason):
    await seed(store)

    class Vector:
        def enabled(self, cfg):
            return True

        async def search(self, *args):
            pytest.fail("Skipped chatter must not call embeddings")

    async def jev(*args):
        pytest.fail("Skipped chatter must not call Jev")

    model = Model([])
    result = await Engine(store, model, Vector(), systemone=jev).recall(SCOPE, query, "bob")
    assert not result["injection"] and model.calls == 0
    assert result["retrieval"]["intent"] == reason
    assert (await store.call("traces", SCOPE))[0]["data"]["retrieval"]["intent"] == reason


@pytest.mark.parametrize(
    "query",
    [
        "推荐点7个以上结局多选择的gal",
        "推荐点国模",
        "omp好在哪",
        "怎么评价谷歌si模型",
        "之前绘画比赛的计划是什么",
        "我平时喝什么咖啡",
        "他后来怎么样了",
        "那台服务器是什么配置",
        "我睡不着怎么办",
        "这个项目是什么",
        "绘画比赛",
        "教我",
        "我应该买什么显卡",
        "我的喜好是什么",
        "我用的推理框架是什么",
    ],
)
def test_history_implicit_background_recommendations_and_personal_questions_are_not_blanket_gated(
    query,
):
    assert ordinary_chat_reason(query) == ""


def test_quote_and_explicit_subject_protect_general_questions():
    assert ordinary_chat_reason("喜欢的饮料是什么", has_subject=True) == ""
    assert ordinary_chat_reason("啤酒烧烤是什么", has_quote=True) == ""
    assert ordinary_chat_reason("之前我好饿是因为什么") == ""
    # Pure acknowledgement does not need a new long-memory review of its quote.
    assert ordinary_chat_reason("有道理", has_quote=True) == "feedback"
    assert ordinary_chat_reason("发语音", has_quote=True) == ""


async def test_quoted_presentation_keeps_historical_topic_recall(store):
    mid, source = await seed(store)
    await store.call("capture", SCOPE, "quote", "alice", "Alice", "我的绘画比赛计划")
    model = Model(answers(mid, source))
    result = await Engine(store, model).recall(SCOPE, "发语音", "bob", reply_id="quote")
    assert result["injection"] and model.calls == 1


async def test_named_person_definition_can_still_recall_own_history(store):
    mid, source = await seed(store)
    model = Model(answers(mid, source))
    result = await Engine(store, model).recall(SCOPE, "测试用户的比赛是什么", "bob")
    assert result["selected"][0]["id"] == mid and model.calls == 1


async def test_recommendation_keeps_personal_preference_recall(store):
    mid, source = await seed(store)
    result = await Engine(store, Model(answers(mid, source))).recall(
        SCOPE, "给我推荐绘画比赛的准备方法", "alice"
    )
    assert result["selected"][0]["id"] == mid


async def test_quote_scoped_to_current_group_still_receives_review(store):
    mid, source = await seed(store)
    await store.call("capture", SCOPE, "quote", "alice", "Alice", "我计划下个月参加绘画比赛")
    model = Model(answers(mid, source))
    result = await Engine(store, model).recall(SCOPE, "比赛是什么", "bob", reply_id="quote")
    assert result["injection"] and model.calls == 1


async def test_current_transient_chatter_is_not_expanded_by_old_conversation(store):
    await seed(store)
    model = Model([])
    result = await Engine(store, model).recall(
        SCOPE, "我好饿", "bob", conversation=[{"role": "user", "content": "alice 的绘画比赛"}]
    )
    assert result["retrieval"]["intent"] == "temporary_state" and model.calls == 0
