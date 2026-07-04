import pytest

from app.mcp.payload import SendPayload
from app.mcp.preflight import PreflightResult, ResolvedRecipient


@pytest.fixture(autouse=True)
def _isolate_artifact_root(monkeypatch, tmp_path):
    """让 send_node 的 artifact 输出落在测试用 tmp_path 里，不污染仓库。"""
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    yield


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
    called = tool.calls[0]
    assert called["to"] == ["a@example.com"]
    assert called["subject"] == "S"
    assert called["body"] == "B"
    assert called.get("attachments"), "P1 必须携带 attachments"
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


# ── 真实 langchain-mcp-adapters 返回结构的回归测试 ────────────────────────────
# StructuredTool(response_format="content_and_artifact").ainvoke(...) 实际返回
# list of content blocks，形如：
#   [{"type": "text", "text": "{\"message_id\": \"...\"}"}]
# 或错误形态：
#   [{"type": "text", "text": "{\"error\": \"...\", ...}"}]
# 之前 send_node 只识别 dict / JSON string，把 list content blocks 当成"无 error 且
# 无 message_id" → 报告 done 但 message_id=None，是 P0 blocker。


def _content_blocks(payload_dict):
    """构造一份跟真实适配器返回等价的 list content blocks。"""
    import json as _json
    return [{"type": "text", "text": _json.dumps(payload_dict, ensure_ascii=False)}]


async def _mock_preflight_ok(**kwargs):
    return PreflightResult(
        True,
        recipients=[ResolvedRecipient(name="张三", email="a@example.com")],
        gmail_to=["a@example.com"],
    )


async def _mock_payload(**kwargs):
    return SendPayload(subject="S", body_text="B", source_answer_preview="A", attachments=[])


def _patch_common(monkeypatch, tool):
    import app.agent.graph.nodes.send as snd
    monkeypatch.setattr(snd, "resolve_send_targets", _mock_preflight_ok)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload)

    class _Manager:
        def get_tools(self, names=None):
            return [tool]

    monkeypatch.setattr(snd, "mcp_manager", _Manager())
    return snd


@pytest.mark.asyncio
async def test_send_node_parses_success_from_list_content_blocks(monkeypatch):
    tool = _CountingTool(_content_blocks({"message_id": "18f2a3b"}))
    snd = _patch_common(monkeypatch, tool)

    update = await snd.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert len(tool.calls) == 1
    assert update["trace"][0]["status"] == "done"
    assert "18f2a3b" in update["send_report"]
    # 绝不允许 message_id=None 混进成功文案
    assert "message_id=None" not in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_parses_error_from_list_content_blocks(monkeypatch):
    """真实 MCP RECIPIENT_NOT_ALLOWED 场景：错误也是 list content blocks 而不是纯 dict。"""
    tool = _CountingTool(_content_blocks({
        "error": "RECIPIENT_NOT_ALLOWED",
        "channel": "gmail",
        "recipient": "unlisted@example.com",
    }))
    snd = _patch_common(monkeypatch, tool)

    update = await snd.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert len(tool.calls) == 1
    assert update["trace"][0]["status"] == "failed"
    assert "发送失败" in update["send_report"]
    assert "RECIPIENT_NOT_ALLOWED" in update["trace"][0]["output"]


@pytest.mark.asyncio
async def test_send_node_treats_content_blocks_without_message_id_as_failure(monkeypatch):
    """就算 list 里没有 error 字段，只要拿不到 message_id 就必须失败，避免静默假成功。"""
    tool = _CountingTool([{"type": "text", "text": "ok, no id"}])
    snd = _patch_common(monkeypatch, tool)

    update = await snd.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert update["trace"][0]["status"] == "failed"
    assert "发送失败" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_parses_tuple_content_artifact(monkeypatch):
    """StructuredTool 有时以 (content, artifact) tuple 形式返回；两个位置任一含 message_id 即成功。"""
    tool = _CountingTool(
        (_content_blocks({"message_id": "abc"}), {"message_id": "abc"})
    )
    snd = _patch_common(monkeypatch, tool)

    update = await snd.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert update["trace"][0]["status"] == "done"
    assert "abc" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_end_to_end_with_real_structured_tool(monkeypatch):
    """接近真实 adapter 的回归：用真的 langchain StructuredTool + response_format=content_and_artifact
    包一层假 Gmail 逻辑，验证 send_node 能拿到 message_id 并标记 done。"""
    from langchain_core.tools import StructuredTool
    import app.agent.graph.nodes.send as snd

    call_counter = {"n": 0}

    def _fake_send(to: list[str], subject: str, body: str):
        call_counter["n"] += 1
        # StructuredTool 会把返回值当 (content, artifact)——这里模拟 MCP tool 的常见结构。
        content = _content_blocks({"message_id": "real-id"})
        artifact = {"message_id": "real-id"}
        return content, artifact

    tool = StructuredTool.from_function(
        func=_fake_send,
        name="gmail_send_email",
        description="fake send",
        response_format="content_and_artifact",
    )

    class _Manager:
        def get_tools(self, names=None):
            return [tool]

    monkeypatch.setattr(snd, "resolve_send_targets", _mock_preflight_ok)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload)
    monkeypatch.setattr(snd, "mcp_manager", _Manager())

    update = await snd.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert call_counter["n"] == 1
    assert update["trace"][0]["status"] == "done"
    assert "real-id" in update["send_report"]
    assert "message_id=None" not in update["send_report"]


# ── P1 附件：send_node 生成 Markdown artifact 并透传 ────────────────────────


@pytest.mark.asyncio
async def test_send_node_generates_artifact_and_passes_attachments(monkeypatch):
    """preflight 成功 → 生成 .md artifact → tool_args.attachments 非空 → send_report 含附件名。"""
    import app.agent.graph.nodes.send as snd
    from pathlib import Path

    tool = _CountingTool({"message_id": "msg-attach"})
    snd_mod = _patch_common(monkeypatch, tool)

    update = await snd_mod.send_node({
        "query": "把报告发给张三",
        "final_answer": "# 报告\n\n内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert len(tool.calls) == 1
    args = tool.calls[0]
    assert args["to"] == ["a@example.com"]
    assert args["attachments"], "必须透传 attachments"
    attach_path = args["attachments"][0]
    assert Path(attach_path).exists()
    assert Path(attach_path).read_text(encoding="utf-8") == "# 报告\n\n内容"
    assert "msg-attach" in update["send_report"]
    assert Path(attach_path).name in update["send_report"]
    assert update["artifact_path"] == attach_path
    assert update["artifact_name"] == Path(attach_path).name
    assert update["artifact_mime"] == "text/markdown"


@pytest.mark.asyncio
async def test_send_node_short_circuits_when_artifact_fails(monkeypatch):
    """artifact 生成失败 → 不调 gmail_send_email，返回失败 send_report。"""
    import app.agent.graph.nodes.send as snd
    from app.mcp.artifacts import ArtifactError

    tool = _CountingTool({"message_id": "should-not-fire"})
    snd_mod = _patch_common(monkeypatch, tool)

    def _boom(**kwargs):
        raise ArtifactError("simulated failure")

    monkeypatch.setattr(snd_mod, "write_markdown_artifact", _boom)

    update = await snd_mod.send_node({
        "query": "q",
        "final_answer": "a",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })

    assert tool.calls == []
    assert "发送失败" in update["send_report"]
    assert update["trace"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_send_node_still_parses_list_content_blocks_with_attachments(monkeypatch):
    """回归：附件模式下依旧要解析 langchain-mcp-adapters 的 list content blocks。"""
    tool = _CountingTool(_content_blocks({"message_id": "list-attach"}))
    snd = _patch_common(monkeypatch, tool)

    update = await snd.send_node({
        "query": "q",
        "final_answer": "answer",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": []},
    })
    assert len(tool.calls) == 1
    assert tool.calls[0].get("attachments"), "attachments 必须传给 tool"
    assert update["trace"][0]["status"] == "done"
    assert "list-attach" in update["send_report"]
