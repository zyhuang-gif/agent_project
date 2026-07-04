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


# ── P2：飞书 / 混合通道 ─────────────────────────────────────────────────


class _MultiTool:
    """按 tool name 分发返回值 / 抛错的 fake tool；同时记录 args。"""

    def __init__(self, name, return_value=None, raise_error=None):
        self.name = name
        self.return_value = return_value
        self.raise_error = raise_error
        self.calls: list[dict] = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if self.raise_error is not None:
            raise self.raise_error
        return self.return_value


class _NamedManager:
    """按名字提供不同工具的 fake mcp_manager。缺失名字返回 []。"""

    def __init__(self, tools_by_name):
        self.tools_by_name = tools_by_name

    def get_tools(self, names=None):
        if names is None:
            return list(self.tools_by_name.values())
        return [self.tools_by_name[n] for n in names if n in self.tools_by_name]


async def _mock_payload_body(**kwargs):
    return SendPayload(
        subject="S", body_text="正文", source_answer_preview="A", attachments=[]
    )


@pytest.mark.asyncio
async def test_send_node_feishu_user_success(monkeypatch):
    """channels=['feishu'] + 一个 person：一次 text + 一次 file。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["feishu"],
            recipients=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
            feishu_users=[
                ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")
            ],
        )

    text_tool = _MultiTool("feishu_send_message_to_user", {"message_id": "om_msg"})
    file_tool = _MultiTool(
        "feishu_send_file_to_user",
        {"message_id": "om_file", "file_key": "file_v3_x"},
    )
    manager = _NamedManager({
        "feishu_send_message_to_user": text_tool,
        "feishu_send_file_to_user": file_tool,
    })

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "把报告发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": ["feishu"]},
    })

    assert len(text_tool.calls) == 1
    assert text_tool.calls[0]["open_id"] == "ou_zs"
    assert text_tool.calls[0]["content"] == "正文"
    assert text_tool.calls[0]["msg_type"] == "text"

    assert len(file_tool.calls) == 1
    assert file_tool.calls[0]["open_id"] == "ou_zs"
    assert file_tool.calls[0]["file_path"]
    assert file_tool.calls[0]["file_name"]

    assert update["trace"][0]["status"] == "done"
    assert "om_msg" in update["send_report"]
    assert "om_file" in update["send_report"]
    assert "file_v3_x" in update["send_report"]
    assert "张三" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_feishu_group_success(monkeypatch):
    """channels=['feishu'] + group：send_message_to_group + send_file_to_group。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["feishu"],
            recipients=[ResolvedRecipient(name="运营组", kind="group", chat_id="oc_op", channel="feishu")],
            feishu_groups=[
                ResolvedRecipient(name="运营组", kind="group", chat_id="oc_op", channel="feishu")
            ],
        )

    text_tool = _MultiTool("feishu_send_message_to_group", {"message_id": "om_g_msg"})
    file_tool = _MultiTool(
        "feishu_send_file_to_group",
        {"message_id": "om_g_file", "file_key": "file_g_x"},
    )
    manager = _NamedManager({
        "feishu_send_message_to_group": text_tool,
        "feishu_send_file_to_group": file_tool,
    })

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发到运营组",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["运营组"], "channels": ["feishu"]},
    })

    assert text_tool.calls[0]["chat_id"] == "oc_op"
    assert file_tool.calls[0]["chat_id"] == "oc_op"
    assert update["trace"][0]["status"] == "done"
    assert "om_g_msg" in update["send_report"]
    assert "file_g_x" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_gmail_and_feishu_serial_success(monkeypatch):
    """混合通道：gmail + feishu 各自成功。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        recipient = ResolvedRecipient(
            name="张三", email="a@example.com", feishu_open_id="ou_zs"
        )
        feishu_user = ResolvedRecipient(
            name="张三", feishu_open_id="ou_zs", channel="feishu"
        )
        return PreflightResult(
            True,
            channels=["gmail", "feishu"],
            recipients=[recipient],
            gmail_to=["a@example.com"],
            feishu_users=[feishu_user],
        )

    gmail_tool = _MultiTool("gmail_send_email", {"message_id": "gmail-1"})
    feishu_msg_tool = _MultiTool("feishu_send_message_to_user", {"message_id": "om_msg"})
    feishu_file_tool = _MultiTool(
        "feishu_send_file_to_user",
        {"message_id": "om_file", "file_key": "file_v3_x"},
    )
    manager = _NamedManager({
        "gmail_send_email": gmail_tool,
        "feishu_send_message_to_user": feishu_msg_tool,
        "feishu_send_file_to_user": feishu_file_tool,
    })

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {
            "send_intent": True,
            "recipients": ["张三"],
            "channels": ["gmail", "feishu"],
        },
    })

    assert len(gmail_tool.calls) == 1
    assert len(feishu_msg_tool.calls) == 1
    assert len(feishu_file_tool.calls) == 1
    assert update["trace"][0]["status"] == "done"
    for tag in ("gmail-1", "om_msg", "om_file", "file_v3_x", "张三"):
        assert tag in update["send_report"], f"缺 {tag}"


@pytest.mark.asyncio
async def test_send_node_gmail_success_feishu_fail_reports_failed(monkeypatch):
    """混合通道：Gmail 成功但 feishu 失败 → 整体 trace failed，报告分别列出。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["gmail", "feishu"],
            recipients=[
                ResolvedRecipient(name="张三", email="a@example.com", feishu_open_id="ou_zs")
            ],
            gmail_to=["a@example.com"],
            feishu_users=[
                ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")
            ],
        )

    gmail_tool = _MultiTool("gmail_send_email", {"message_id": "gmail-ok"})
    feishu_msg_tool = _MultiTool(
        "feishu_send_message_to_user",
        {"error": "LARK_API_ERROR", "channel": "feishu", "message": "token missing"},
    )
    feishu_file_tool = _MultiTool(
        "feishu_send_file_to_user",
        {"error": "LARK_API_ERROR", "channel": "feishu", "message": "token missing"},
    )
    manager = _NamedManager({
        "gmail_send_email": gmail_tool,
        "feishu_send_message_to_user": feishu_msg_tool,
        "feishu_send_file_to_user": feishu_file_tool,
    })

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {
            "send_intent": True,
            "recipients": ["张三"],
            "channels": ["gmail", "feishu"],
        },
    })

    assert len(gmail_tool.calls) == 1
    assert len(feishu_msg_tool.calls) == 1
    # 首个失败即停：feishu text 失败后 file 工具绝不能再被调用
    assert feishu_file_tool.calls == []
    assert update["trace"][0]["status"] == "failed"
    assert "gmail-ok" in update["send_report"]
    assert "发送失败" in update["send_report"]
    assert "token missing" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_feishu_missing_tool_reports_failed(monkeypatch):
    """preflight 通过但 feishu MCP 工具未加载 → 明确失败，不假成功。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["feishu"],
            recipients=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
            feishu_users=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
        )

    manager = _NamedManager({})  # 完全没工具

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": ["feishu"]},
    })

    assert update["trace"][0]["status"] == "failed"
    assert "发送失败" in update["send_report"]
    assert "未加载" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_feishu_success_list_content_blocks(monkeypatch):
    """feishu tool 返回 list content blocks 时也要解析出 message_id / file_key。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["feishu"],
            recipients=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
            feishu_users=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
        )

    text_tool = _MultiTool(
        "feishu_send_message_to_user",
        _content_blocks({"message_id": "om_txt"}),
    )
    file_tool = _MultiTool(
        "feishu_send_file_to_user",
        _content_blocks({"message_id": "om_fkey", "file_key": "file_v3_z"}),
    )
    manager = _NamedManager({
        "feishu_send_message_to_user": text_tool,
        "feishu_send_file_to_user": file_tool,
    })
    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": ["feishu"]},
    })

    assert update["trace"][0]["status"] == "done"
    assert "om_txt" in update["send_report"]
    assert "om_fkey" in update["send_report"]
    assert "file_v3_z" in update["send_report"]
    assert "message_id=None" not in update["send_report"]


# ── Fail-closed 预检（回归 P2 部分发送阻塞问题） ───────────────────────


@pytest.mark.asyncio
async def test_send_node_mixed_channel_feishu_missing_precheck_blocks_gmail(monkeypatch):
    """混合通道：Gmail 工具存在但 Feishu 工具缺失时，预检必须阻止 Gmail 发送。

    这是 P2 fail-closed 的最重要回归：如果预检不做，Gmail 会先真实发送，
    然后整体才 failed → 违反 spec §6 "不部分发送"。
    """
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["gmail", "feishu"],
            recipients=[
                ResolvedRecipient(name="张三", email="a@example.com", feishu_open_id="ou_zs")
            ],
            gmail_to=["a@example.com"],
            feishu_users=[
                ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")
            ],
        )

    gmail_tool = _MultiTool("gmail_send_email", {"message_id": "should-not-fire"})
    # Feishu 侧完全不存在：manager 只知道 gmail_send_email
    manager = _NamedManager({"gmail_send_email": gmail_tool})

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {
            "send_intent": True,
            "recipients": ["张三"],
            "channels": ["gmail", "feishu"],
        },
    })

    # 核心断言：Gmail 也 **绝不能** 被调用
    assert gmail_tool.calls == [], "预检发现 Feishu 缺失时 Gmail 不允许发送"
    assert update["trace"][0]["status"] == "failed"
    assert "未加载" in update["send_report"]
    # 报告里应列出具体缺失的工具名，方便排查
    assert "feishu_send_message_to_user" in update["send_report"]
    assert "feishu_send_file_to_user" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_feishu_only_file_missing_precheck_blocks_text(monkeypatch):
    """Feishu-only：text 工具存在但 file 工具缺失 → text 也不能发。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["feishu"],
            recipients=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
            feishu_users=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
        )

    text_tool = _MultiTool("feishu_send_message_to_user", {"message_id": "should-not-fire"})
    # file 工具没配
    manager = _NamedManager({"feishu_send_message_to_user": text_tool})

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": ["feishu"]},
    })

    assert text_tool.calls == [], "file 工具缺失时 text 也不允许发送"
    assert update["trace"][0]["status"] == "failed"
    assert "未加载" in update["send_report"]
    assert "feishu_send_file_to_user" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_stops_after_first_feishu_text_failure(monkeypatch):
    """运行期首个失败即停：feishu text 返回 error 后，file 工具不能再被调用。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["feishu"],
            recipients=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
            feishu_users=[ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")],
        )

    text_tool = _MultiTool(
        "feishu_send_message_to_user",
        {"error": "LARK_API_ERROR", "channel": "feishu", "message": "boom"},
    )
    file_tool = _MultiTool("feishu_send_file_to_user", {"message_id": "should-not-fire"})
    manager = _NamedManager({
        "feishu_send_message_to_user": text_tool,
        "feishu_send_file_to_user": file_tool,
    })

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": ["feishu"]},
    })

    assert len(text_tool.calls) == 1
    assert file_tool.calls == [], "text 失败后不能再调 file"
    assert update["trace"][0]["status"] == "failed"
    assert "发送失败" in update["send_report"]
    assert "boom" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_stops_when_gmail_fails_before_feishu(monkeypatch):
    """混合通道下 Gmail 先执行失败 → 后续 Feishu items 一律不再调用。"""
    import app.agent.graph.nodes.send as snd

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["gmail", "feishu"],
            recipients=[
                ResolvedRecipient(name="张三", email="a@example.com", feishu_open_id="ou_zs")
            ],
            gmail_to=["a@example.com"],
            feishu_users=[
                ResolvedRecipient(name="张三", feishu_open_id="ou_zs", channel="feishu")
            ],
        )

    gmail_tool = _MultiTool(
        "gmail_send_email",
        {"error": "GMAIL_API_ERROR", "channel": "gmail", "message": "quota"},
    )
    text_tool = _MultiTool("feishu_send_message_to_user", {"message_id": "should-not-fire"})
    file_tool = _MultiTool("feishu_send_file_to_user", {"message_id": "should-not-fire"})
    manager = _NamedManager({
        "gmail_send_email": gmail_tool,
        "feishu_send_message_to_user": text_tool,
        "feishu_send_file_to_user": file_tool,
    })

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {
            "send_intent": True,
            "recipients": ["张三"],
            "channels": ["gmail", "feishu"],
        },
    })

    assert len(gmail_tool.calls) == 1
    assert text_tool.calls == []
    assert file_tool.calls == []
    assert update["trace"][0]["status"] == "failed"
    assert "发送失败" in update["send_report"]
    assert "quota" in update["send_report"]


@pytest.mark.asyncio
async def test_send_node_precheck_missing_attachment_blocks_all_tools(monkeypatch, tmp_path):
    """本地文件预检：附件路径不存在时，任何真实工具都不允许被调。

    走一条 Gmail-only 通道，但用 monkeypatch 让 write_markdown_artifact 返回一个
    指向不存在文件的 Artifact——模拟"artifact 被外部删除 / ARTIFACT_ROOT 配错"。
    """
    import app.agent.graph.nodes.send as snd
    from app.mcp.artifacts import Artifact

    async def _preflight(**kwargs):
        return PreflightResult(
            True,
            channels=["gmail"],
            recipients=[ResolvedRecipient(name="张三", email="a@example.com")],
            gmail_to=["a@example.com"],
        )

    fake_path = str(tmp_path / "does-not-exist.md")

    def _fake_artifact(**kwargs):
        return Artifact(path=fake_path, name="does-not-exist.md", mime="text/markdown", size=0)

    gmail_tool = _MultiTool("gmail_send_email", {"message_id": "should-not-fire"})
    manager = _NamedManager({"gmail_send_email": gmail_tool})

    monkeypatch.setattr(snd, "resolve_send_targets", _preflight)
    monkeypatch.setattr(snd, "build_send_payload", _mock_payload_body)
    monkeypatch.setattr(snd, "write_markdown_artifact", _fake_artifact)
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "sid",
        "identity": None,
        "plan": {"send_intent": True, "recipients": ["张三"], "channels": ["gmail"]},
    })

    assert gmail_tool.calls == []
    assert update["trace"][0]["status"] == "failed"
    assert "不存在" in update["send_report"]
