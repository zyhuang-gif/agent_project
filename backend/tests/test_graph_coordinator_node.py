import pytest

from app.agent.graph.nodes.coordinator import (
    CoordinatorPlan,
    coordinator_node,
    _plan_to_dict,
)


def test_plan_to_dict_accepts_structured_output():
    plan = _plan_to_dict(CoordinatorPlan(
        task_type="document_compare",
        need_retrieval=True,
        reason="对比新旧制度",
    ))
    assert plan["task_type"] == "document_compare"
    assert plan["need_retrieval"] is True
    assert plan["reason"] == "对比新旧制度"


def test_plan_to_dict_unknown_task_type_is_supported():
    plan = _plan_to_dict(CoordinatorPlan(
        task_type="unknown",
        need_retrieval=False,
        reason="x",
    ))
    assert plan["task_type"] == "unknown"
    assert plan["need_retrieval"] is False


class _FakeMsg:
    usage_metadata = {"total_tokens": 123}


class _FakeStructuredModel:
    def __init__(self, result, captured):
        self.result = result
        self.captured = captured

    async def ainvoke(self, messages):
        self.captured["messages"] = messages
        return {"raw": _FakeMsg(), "parsed": self.result, "parsing_error": None}


class _FakeChatModel:
    def __init__(self, result, captured):
        self.result = result
        self.captured = captured

    def with_structured_output(self, schema, include_raw=False):
        self.captured["schema"] = schema
        self.captured["include_raw"] = include_raw
        return _FakeStructuredModel(self.result, self.captured)


@pytest.mark.asyncio
async def test_coordinator_node_writes_plan(monkeypatch):
    captured = {}
    result = CoordinatorPlan(
        task_type="report_generation",
        need_retrieval=True,
        reason="要生成报告",
    )

    import app.agent.graph.nodes.coordinator as co
    monkeypatch.setattr(co, "chat_model", _FakeChatModel(result, captured))

    update = await coordinator_node({"query": "根据售后政策生成报告"})
    assert captured["schema"] is CoordinatorPlan
    assert captured["include_raw"] is True
    assert update["plan"]["task_type"] == "report_generation"
    assert update["plan"]["need_retrieval"] is True
    assert update["token_usage"] == 123
    assert update["trace"][0]["agent"] == "coordinator"


@pytest.mark.asyncio
async def test_coordinator_uses_recent_history(monkeypatch):
    captured = {}
    result = CoordinatorPlan(
        task_type="knowledge_qa",
        need_retrieval=True,
        reason="x",
    )

    import app.agent.graph.nodes.coordinator as co
    monkeypatch.setattr(co, "chat_model", _FakeChatModel(result, captured))

    state = {
        "query": "那它呢",
        "history": [("差旅报销上限", "上限500元")],
    }
    await coordinator_node(state)
    user_content = captured["messages"][-1]["content"]
    assert "差旅报销上限" in user_content and "那它呢" in user_content


def test_plan_to_dict_normalizes_send_fields():
    plan = _plan_to_dict(CoordinatorPlan(
        task_type="report_generation",
        need_retrieval=True,
        reason="生成并发送报告",
        send_intent=True,
        recipients=["张三", "运营组"],
        channels=["gmail", "feishu"],
    ))

    assert plan["send_intent"] is True
    assert plan["recipients"] == ["张三", "运营组"]
    assert plan["channels"] == ["gmail", "feishu"]


def test_plan_to_dict_defaults_send_fields():
    plan = _plan_to_dict(CoordinatorPlan(
        task_type="knowledge_qa",
        need_retrieval=True,
        reason="普通问答",
    ))

    assert plan["send_intent"] is False
    assert plan["recipients"] == []
    assert plan["channels"] == []
