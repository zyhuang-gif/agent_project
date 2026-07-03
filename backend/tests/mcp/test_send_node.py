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


@pytest.mark.asyncio
async def test_send_node_invokes_only_gmail_tool(monkeypatch):
    import app.agent.graph.nodes.send as snd

    captured = {}

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            recipients=[ResolvedRecipient(name="张三", email="a@example.com")],
            gmail_to=["a@example.com"],
        )

    async def _payload(**kwargs):
        return SendPayload(subject="S", body_text="B", source_answer_preview="A", attachments=[])

    class _Manager:
        def get_tools(self, names=None):
            captured["tool_names"] = names
            return ["gmail-tool"]

    class _Agent:
        async def ainvoke(self, data):
            captured["task"] = data["messages"][0]["content"]
            return {"messages": [type("Msg", (), {"content": "已发送 message_id=msg-1"})()]}

    def _create_react_agent(model, tools):
        captured["tools"] = tools
        return _Agent()

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _payload)
    monkeypatch.setattr(snd, "mcp_manager", _Manager())
    monkeypatch.setattr(snd, "create_react_agent", _create_react_agent)
    monkeypatch.setattr(snd, "get_chat_model", lambda role: "model")

    update = await snd.send_node({
        "query": "生成报告并发给张三",
        "final_answer": "answer",
        "session_id": "sid",
        "identity": type("Identity", (), {"user_id": "uid"})(),
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert captured["tool_names"] == ["gmail_send_email"]
    assert captured["tools"] == ["gmail-tool"]
    assert "不要添加附件" in captured["task"]
    assert "a@example.com" in captured["task"]
    assert update["send_report"] == "已发送 message_id=msg-1"
    assert update["send_payload"].subject == "S"
    assert update["trace"][0]["agent"] == "send"
