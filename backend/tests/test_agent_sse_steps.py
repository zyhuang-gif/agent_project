import json
import importlib

import pytest

from app.agent import agent
from app.utils.auth_utils import RequestIdentity

db_session_manager = importlib.import_module("app.services.database_session_manager")


class _FakeSessionManager:
    def __init__(self):
        self.last_call = None

    async def get_history(self, session_id, user_id):
        return []

    async def add_message(self, session_id, user_id, query, response, citations=None, steps=None, send_report=None):
        self.last_call = {
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "response": response,
            "citations": citations,
            "steps": steps,
            "send_report": send_report,
        }
        return None


async def _fake_agent_stream(query, history, identity=None):
    yield {
        "type": "agent_plan",
        "data": [
            {
                "id": "task_understood",
                "title": "理解用户问题",
                "status": "done",
                "level": "success",
            },
            {
                "id": "tool_rag_summary_tools",
                "title": "检索相关知识库",
                "status": "todo",
                "level": "muted",
            },
            {
                "id": "answer_generated",
                "title": "生成最终回答",
                "status": "todo",
                "level": "muted",
            },
        ],
    }
    yield {
        "type": "agent_step_update",
        "data": {
            "id": "tool_rag_summary_tools",
            "status": "running",
            "level": "info",
            "detail": "正在检索相关知识库",
        },
    }
    yield {
        "type": "step",
        "data": {
            "tool": "rag_summary_tools",
            "tool_input": '{"query": "LangChain"}',
            "tool_output": "检索到 2 个文档",
        },
    }
    yield {"type": "token", "data": "答案"}
    yield {
        "type": "done",
        "steps": [],
        "tokens": 12,
        "citations": [{"filename": "doc.md", "score": 0.9}],
    }


def _decode_sse_events(chunks):
    events = []
    for chunk in chunks:
        for block in chunk.strip().split("\n\n"):
            if not block.startswith("data: "):
                continue
            events.append(json.loads(block.removeprefix("data: ")))
    return events


@pytest.mark.asyncio
async def test_agent_stream_response_sends_tool_steps_to_frontend(monkeypatch):
    monkeypatch.setenv("AGENT_ENGINE", "loop")
    monkeypatch.setattr(
        db_session_manager,
        "database_session_manager",
        _FakeSessionManager(),
    )
    monkeypatch.setattr(agent.agent_loop, "stream", _fake_agent_stream)

    chunks = [
        chunk
        async for chunk in agent.get_agent_stream_response(
            "什么是 LangChain",
            "session-1",
            RequestIdentity(user_id="u1"),
        )
    ]

    events = _decode_sse_events(chunks)

    assert {
        "type": "agent_step",
        "data": {
            "id": "tool_rag_summary_tools",
            "title": "已检索知识库",
            "status": "done",
            "level": "success",
            "detail": "检索到 2 个文档",
        },
    } in events


@pytest.mark.asyncio
async def test_agent_stream_response_sends_plan_and_status_updates(monkeypatch):
    monkeypatch.setenv("AGENT_ENGINE", "loop")
    monkeypatch.setattr(
        db_session_manager,
        "database_session_manager",
        _FakeSessionManager(),
    )
    monkeypatch.setattr(agent.agent_loop, "stream", _fake_agent_stream)

    chunks = [
        chunk
        async for chunk in agent.get_agent_stream_response(
            "什么是 LangChain",
            "session-1",
            RequestIdentity(user_id="u1"),
        )
    ]

    events = _decode_sse_events(chunks)

    assert {
        "type": "agent_plan",
        "data": [
            {
                "id": "task_understood",
                "title": "理解用户问题",
                "status": "done",
                "level": "success",
            },
            {
                "id": "tool_rag_summary_tools",
                "title": "检索相关知识库",
                "status": "todo",
                "level": "muted",
            },
            {
                "id": "answer_generated",
                "title": "生成最终回答",
                "status": "todo",
                "level": "muted",
            },
        ],
    } in events
    assert {
        "type": "agent_step_update",
        "data": {
            "id": "tool_rag_summary_tools",
            "status": "running",
            "level": "info",
            "detail": "正在检索相关知识库",
        },
    } in events


@pytest.mark.asyncio
async def test_agent_stream_response_persists_steps_and_citations(monkeypatch):
    """流结束时,累积后的 step 最终态(按 id 合并)与 citations 应一并传给 add_message。"""
    monkeypatch.setenv("AGENT_ENGINE", "loop")
    fake_sm = _FakeSessionManager()
    monkeypatch.setattr(db_session_manager, "database_session_manager", fake_sm)
    monkeypatch.setattr(agent.agent_loop, "stream", _fake_agent_stream)

    chunks = [
        chunk
        async for chunk in agent.get_agent_stream_response(
            "什么是 LangChain",
            "session-1",
            RequestIdentity(user_id="u1"),
        )
    ]
    assert chunks  # 消费完流

    assert fake_sm.last_call is not None
    assert fake_sm.last_call["citations"] == [{"filename": "doc.md", "score": 0.9}]
    # 非 send 路径不应写入 send_report
    assert fake_sm.last_call["send_report"] is None

    persisted_steps = fake_sm.last_call["steps"]
    assert isinstance(persisted_steps, list)
    by_id = {s["id"]: s for s in persisted_steps if isinstance(s, dict) and s.get("id")}

    # plan 里的三步都在,update 之后 tool_rag_summary_tools 的 status 被 step 事件最终覆盖为 done
    assert set(by_id.keys()) == {
        "task_understood",
        "tool_rag_summary_tools",
        "answer_generated",
    }
    assert by_id["tool_rag_summary_tools"]["status"] == "done"
    assert by_id["tool_rag_summary_tools"]["level"] == "success"
    assert by_id["tool_rag_summary_tools"]["detail"] == "检索到 2 个文档"
