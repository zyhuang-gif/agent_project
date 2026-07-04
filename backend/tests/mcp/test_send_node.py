import pytest

from app.mcp.payload import SendPayload
from app.mcp.preflight import PreflightResult, ResolvedRecipient


@pytest.mark.asyncio
async def test_send_node_skips_without_intent():
    from app.agent.graph.nodes.send import send_node

    assert await send_node({"plan": {}}) == {}


@pytest.mark.asyncio
async def test_send_node_returns_preflight_message(monkeypatch):
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(False, user_message="需要确认联系人", reason="ambiguous")

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)

    update = await snd.send_node(
        {"plan": {"send_intent": True, "recipients": ["张"], "channels": []}}
    )

    assert update["send_report"] == "需要确认联系人"
    assert update["trace"][0]["status"] == "skipped"


class _CountingTool:
    """记录 ainvoke 调用次数与参数，模拟 langchain BaseTool。"""

    def __init__(self, return_value):
        self.return_value = return_value
        self.calls: list[dict] = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.return_value


@pytest.mark.asyncio
async def test_send_node_invokes_gmail_tool_exactly_once(monkeypatch):
    """send_node 必须直接调 gmail_send_email 一次，不能走 create_react_agent 让 LLM 自由决定。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            recipients=[ResolvedRecipient(name="张三", email="a@example.com")],
            gmail_to=["a@example.com"],
        )

    async def _payload(**kwargs):
        return SendPayload(subject="S", body_text="B", source_answer_preview="A", attachments=[])

    tool = _CountingTool({"message_id": "msg-1"})

    class _Manager:
        def get_tools(self, names=None):
            assert names == ["gmail_send_email"]
            return [tool]

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _payload)
    monkeypatch.setattr(snd, "mcp_manager", _Manager())

    update = await snd.send_node({
        "query": "生成报告并发给张三",
        "final_answer": "answer",
        "session_id": "sid",
        "identity": type("Identity", (), {"user_id": "uid"})(),
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert len(tool.calls) == 1
    assert tool.calls[0] == {
        "to": ["a@example.com"],
        "subject": "S",
        "body": "B",
    }
    assert "msg-1" in update["send_report"]
    assert "张三" in update["send_report"]
    assert update["send_payload"].subject == "S"
    assert update["trace"][0]["agent"] == "send"
    assert update["trace"][0]["status"] == "done"


@pytest.mark.asyncio
async def test_send_node_reports_failure_when_tool_returns_error(monkeypatch):
    """tool 返回 error payload 时，send_node 应报失败并且 trace 为 failed。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            recipients=[ResolvedRecipient(name="张三", email="a@example.com")],
            gmail_to=["a@example.com"],
        )

    async def _payload(**kwargs):
        return SendPayload(subject="S", body_text="B", source_answer_preview="A", attachments=[])

    tool = _CountingTool({"error": "AUTH_FAILED", "channel": "gmail", "message": "token missing"})

    class _Manager:
        def get_tools(self, names=None):
            return [tool]

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _payload)
    monkeypatch.setattr(snd, "mcp_manager", _Manager())

    update = await snd.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert len(tool.calls) == 1
    assert "发送失败" in update["send_report"]
    assert update["trace"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_send_node_reports_failure_when_tool_raises(monkeypatch):
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            recipients=[ResolvedRecipient(name="张三", email="a@example.com")],
            gmail_to=["a@example.com"],
        )

    async def _payload(**kwargs):
        return SendPayload(subject="S", body_text="B", source_answer_preview="A", attachments=[])

    class _RaisingTool:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, args):
            self.calls += 1
            raise RuntimeError("gmail down")

    tool = _RaisingTool()

    class _Manager:
        def get_tools(self, names=None):
            return [tool]

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _payload)
    monkeypatch.setattr(snd, "mcp_manager", _Manager())

    update = await snd.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert tool.calls == 1
    assert "发送失败" in update["send_report"]
    assert update["trace"][0]["status"] == "failed"
