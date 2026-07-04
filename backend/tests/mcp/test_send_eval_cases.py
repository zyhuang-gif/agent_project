"""发送类 eval / 回归集（P3a）。

目标：把 P0/P1/P2 已锁死的安全边界与常见发送话术，用少量固定 fixture 的
参数化测试再走一遍。测试用 monkeypatch 隔离 LLM 与外部 API，只断言：

  - preflight：给定 (recipients, channels) 是否返回 can_send / user_message；
  - send_node：真实工具调用次数 & send_report 关键字 & trace status。

所有 case 都不真发外网。任何"部分发送"、"模糊直发"、"缺工具静默成功" 的
regression 都会在此挂掉。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from app.mcp.payload import SendPayload
from app.mcp.preflight import resolve_send_targets
from app.services.contact_service import ContactMatch, ContactResolveResult


# ── Fake 联系人簿：张三 / 张四 / 运营群 ───────────────────────────────────


class _FakeContactService:
    """dev-friendly 联系人 stub。张三/张四模糊冲突；运营群是 group。"""

    async def resolve(self, name):
        n = (name or "").strip()
        if n == "张三":
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person",
                        name="张三",
                        match_type="exact",
                        email="zhangsan@example.com",
                        feishu_open_id="ou_zs",
                    )
                ]
            )
        if n == "张四":
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person",
                        name="张四",
                        match_type="exact",
                        email="zhangsi@example.com",
                        feishu_open_id="ou_zs4",
                    )
                ]
            )
        if n == "运营群":
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="group",
                        name="运营群",
                        match_type="exact",
                        chat_id="oc_op",
                    )
                ]
            )
        if n == "张":
            return ContactResolveResult(
                matches=[
                    ContactMatch(
                        kind="person",
                        name="张三",
                        match_type="fuzzy",
                        email="zhangsan@example.com",
                        feishu_open_id="ou_zs",
                    ),
                    ContactMatch(
                        kind="person",
                        name="张四",
                        match_type="fuzzy",
                        email="zhangsi@example.com",
                        feishu_open_id="ou_zs4",
                    ),
                ],
                ambiguous=True,
            )
        # "王五不存在" / 任意未知名 → 无匹配
        return ContactResolveResult(matches=[])


# ── Fake MCP tool / manager ───────────────────────────────────────────


class _RecordingTool:
    """记录 ainvoke 调用参数，无外部 API 副作用。"""

    def __init__(self, name: str, return_value):
        self.name = name
        self.return_value = return_value
        self.calls: list[dict] = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.return_value


class _NamedManager:
    def __init__(self, tools_by_name: dict):
        self.tools_by_name = tools_by_name

    def get_tools(self, names=None):
        if names is None:
            return list(self.tools_by_name.values())
        return [self.tools_by_name[n] for n in names if n in self.tools_by_name]


# ── 用来在 send_node 里 stub artifact / payload / audit / contact_service ───


@pytest.fixture(autouse=True)
def _isolate_artifact_root(monkeypatch, tmp_path):
    """让 send_node 的 markdown artifact 落在 tmp_path，避免污染仓库。"""
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    yield


async def _fake_payload_builder(**kwargs):
    return SendPayload(subject="报告", body_text="正文", source_answer_preview="A", attachments=[])


def _all_tools() -> _NamedManager:
    """全通道工具都齐活的 fake manager。"""
    return _NamedManager({
        "gmail_send_email": _RecordingTool("gmail_send_email", {"message_id": "gmail-1"}),
        "feishu_send_message_to_user": _RecordingTool(
            "feishu_send_message_to_user", {"message_id": "om_msg"}
        ),
        "feishu_send_file_to_user": _RecordingTool(
            "feishu_send_file_to_user", {"message_id": "om_file", "file_key": "file_v3_x"}
        ),
        "feishu_send_message_to_group": _RecordingTool(
            "feishu_send_message_to_group", {"message_id": "om_g_msg"}
        ),
        "feishu_send_file_to_group": _RecordingTool(
            "feishu_send_file_to_group", {"message_id": "om_g_file", "file_key": "file_g_x"}
        ),
    })


@pytest.fixture
def send_env(monkeypatch):
    """一次性把 contact_service / mcp_manager / audit 都 stub 掉。"""
    import app.agent.graph.nodes.send as snd
    import app.mcp.preflight as pf
    import app.services.send_audit_service as svc_mod

    fake_contact = _FakeContactService()
    monkeypatch.setattr(pf, "contact_service", fake_contact)
    monkeypatch.setattr(snd, "build_send_payload", _fake_payload_builder)

    # 让 audit DB 不真的连 MySQL；直接 no-op
    class _NoopSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def add(self, *_): pass
        async def commit(self): return None

    monkeypatch.setattr(svc_mod, "AsyncSessionLocal", lambda: _NoopSession())
    # 让 audit JSONL 也走 tmp（通过 setenv 之类不方便；直接 monkeypatch）
    monkeypatch.setattr(svc_mod, "_audit_path", lambda: Path.cwd() / ".pytest-tmp" / "sends.log")

    manager = _all_tools()
    monkeypatch.setattr(snd, "mcp_manager", manager)
    return manager


# ── Eval 用例 ────────────────────────────────────────────────────────


@dataclass
class SendEvalCase:
    id: str
    utterance: str            # 用户话术（仅作文档用途）
    recipients: list[str]
    channels: list[str]
    expected_status: str      # "done" | "failed" | "skipped"
    expected_tool_calls: dict[str, int]
    report_contains: Optional[str] = None
    reason: Optional[str] = None


CASES: list[SendEvalCase] = [
    SendEvalCase(
        id="1_default_gmail",
        utterance="把刚才的报告发给张三",
        recipients=["张三"],
        channels=[],
        expected_status="done",
        expected_tool_calls={"gmail_send_email": 1},
        report_contains="gmail-1",
    ),
    SendEvalCase(
        id="2_explicit_gmail",
        utterance="邮件发给张三",
        recipients=["张三"],
        channels=["gmail"],
        expected_status="done",
        expected_tool_calls={"gmail_send_email": 1},
        report_contains="gmail-1",
    ),
    SendEvalCase(
        id="3_group_plus_gmail_fail_closed",
        utterance="发给张三，同时发到运营群（邮件+飞书）",
        recipients=["张三", "运营群"],
        channels=["gmail", "feishu"],
        expected_status="skipped",
        expected_tool_calls={},
        reason="gmail_group_unsupported",
    ),
    SendEvalCase(
        id="4_ambiguous_no_send",
        utterance="发给张",
        recipients=["张"],
        channels=[],
        expected_status="skipped",
        expected_tool_calls={},
        reason="ambiguous_recipient",
    ),
    SendEvalCase(
        id="5_unknown_no_partial_send",
        utterance="发给张三和不存在的人",
        recipients=["张三", "王五不存在"],
        channels=[],
        expected_status="skipped",
        expected_tool_calls={},
        reason="recipient_not_found",
    ),
    SendEvalCase(
        id="6_group_feishu_only",
        utterance="发到运营群",
        recipients=["运营群"],
        channels=["feishu"],
        expected_status="done",
        expected_tool_calls={
            "feishu_send_message_to_group": 1,
            "feishu_send_file_to_group": 1,
        },
        report_contains="om_g_msg",
    ),
    SendEvalCase(
        id="7_explicit_feishu_user",
        utterance="用飞书发给张三",
        recipients=["张三"],
        channels=["feishu"],
        expected_status="done",
        expected_tool_calls={
            "feishu_send_message_to_user": 1,
            "feishu_send_file_to_user": 1,
        },
        report_contains="om_msg",
    ),
    SendEvalCase(
        id="8_mixed_group_and_gmail_fail_closed",
        utterance="发给张三和运营群，邮件和飞书都发",
        recipients=["张三", "运营群"],
        channels=["gmail", "feishu"],
        expected_status="skipped",
        expected_tool_calls={},
        reason="gmail_group_unsupported",
    ),
    SendEvalCase(
        id="9_group_without_feishu_channel",
        utterance="发到运营群（未指定 feishu）",
        recipients=["运营群"],
        channels=[],   # 默认 gmail
        expected_status="skipped",
        expected_tool_calls={},
        reason="gmail_group_unsupported",
    ),
    SendEvalCase(
        id="10_mixed_gmail_feishu_person_success",
        utterance="邮件和飞书都发给张三",
        recipients=["张三"],
        channels=["gmail", "feishu"],
        expected_status="done",
        expected_tool_calls={
            "gmail_send_email": 1,
            "feishu_send_message_to_user": 1,
            "feishu_send_file_to_user": 1,
        },
        report_contains="om_file",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_send_eval_case(monkeypatch, send_env, case: SendEvalCase):
    """把 (recipients, channels) 直接喂 send_node，断言 tool 调用次数 & trace status。"""
    import app.agent.graph.nodes.send as snd

    manager: _NamedManager = send_env

    update = await snd.send_node({
        "query": case.utterance,
        "final_answer": "# 内容",
        "session_id": "eval-sess",
        "identity": None,
        "plan": {
            "send_intent": True,
            "recipients": case.recipients,
            "channels": case.channels,
        },
    })

    # 1) trace status
    trace = update.get("trace") or []
    assert trace and trace[0].get("agent") == "send", f"{case.id}: 缺 send trace"
    assert trace[0].get("status") == case.expected_status, (
        f"{case.id}: 预期 status={case.expected_status}，实际={trace[0].get('status')}"
    )

    # 2) 工具调用次数——包括应为 0 的工具必须都是 0
    for tool_name, tool in manager.tools_by_name.items():
        expected = case.expected_tool_calls.get(tool_name, 0)
        assert len(tool.calls) == expected, (
            f"{case.id}: 工具 {tool_name} 调用次数应为 {expected}，实际 {len(tool.calls)}"
        )

    # 3) 报告关键字（成功场景）
    if case.report_contains:
        assert case.report_contains in (update.get("send_report") or ""), (
            f"{case.id}: 报告中应含 '{case.report_contains}'"
        )

    # 4) preflight 原因（skipped 场景）
    if case.reason:
        assert case.reason in (trace[0].get("output") or ""), (
            f"{case.id}: 预期 preflight reason 含 '{case.reason}'，"
            f"实际='{trace[0].get('output')}'"
        )


@pytest.mark.asyncio
async def test_send_eval_no_intent_returns_empty(send_env):
    """P0 保护：send_intent=False 时 send_node 直接 no-op，绝不触工具。"""
    import app.agent.graph.nodes.send as snd

    manager: _NamedManager = send_env
    update = await snd.send_node({
        "query": "只是问个问题",
        "final_answer": "# 答案",
        "session_id": "eval-sess",
        "identity": None,
        "plan": {"send_intent": False, "recipients": ["张三"], "channels": []},
    })

    assert update == {}
    for tool in manager.tools_by_name.values():
        assert tool.calls == []


@pytest.mark.asyncio
async def test_send_eval_missing_feishu_tool_blocks_gmail(monkeypatch):
    """P2 fail-closed 回归：混合通道时 Feishu 工具缺失 → Gmail 也不允许发。"""
    import app.agent.graph.nodes.send as snd
    import app.mcp.preflight as pf
    import app.services.send_audit_service as svc_mod

    monkeypatch.setattr(pf, "contact_service", _FakeContactService())
    monkeypatch.setattr(snd, "build_send_payload", _fake_payload_builder)

    class _NoopSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def add(self, *_): pass
        async def commit(self): return None

    monkeypatch.setattr(svc_mod, "AsyncSessionLocal", lambda: _NoopSession())

    gmail_tool = _RecordingTool("gmail_send_email", {"message_id": "should-not-fire"})
    manager = _NamedManager({"gmail_send_email": gmail_tool})
    monkeypatch.setattr(snd, "mcp_manager", manager)

    update = await snd.send_node({
        "query": "发给张三",
        "final_answer": "# 内容",
        "session_id": "eval-sess",
        "identity": None,
        "plan": {
            "send_intent": True,
            "recipients": ["张三"],
            "channels": ["gmail", "feishu"],
        },
    })

    assert gmail_tool.calls == [], "缺 feishu 工具时 Gmail 也不允许发"
    trace = update.get("trace") or []
    assert trace and trace[0].get("status") == "failed"
    assert "未加载" in (update.get("send_report") or "")


# ── preflight 单独跑一遍，把 case 的原因短语固化 ───────────────────────


PREFLIGHT_CASES = [c for c in CASES if c.reason]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", PREFLIGHT_CASES, ids=[c.id for c in PREFLIGHT_CASES])
async def test_preflight_reason_locked(monkeypatch, case: SendEvalCase):
    """preflight 层直接暴露 reason，方便产品排查话术为何未发。"""
    import app.mcp.preflight as pf
    monkeypatch.setattr(pf, "contact_service", _FakeContactService())

    result = await resolve_send_targets(case.recipients, case.channels)
    assert result.can_send is False
    assert result.reason == case.reason, (
        f"{case.id}: preflight reason 应为 {case.reason}，实际 {result.reason}"
    )
    assert result.user_message, "preflight 失败时必须给出中文 user_message"
