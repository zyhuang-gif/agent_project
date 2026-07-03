import pytest

import app.agent.graph.nodes.coordinator as co
import app.agent.graph.nodes.finalize as fz
import app.agent.graph.nodes.send as snd
from app.agent.graph.nodes.coordinator import CoordinatorPlan
from app.agent.graph.runner import GraphRunner


class _Raw:
    usage_metadata = {"total_tokens": 10}


class _StructFake:
    async def ainvoke(self, messages):
        return {
            "raw": _Raw(),
            "parsed": CoordinatorPlan(
                task_type="knowledge_qa",
                need_retrieval=False,
                reason="直接发送已有上下文",
                send_intent=True,
                recipients=["张三"],
                channels=[],
            ),
            "parsing_error": None,
        }


class _CoordModel:
    def with_structured_output(self, schema, include_raw=False):
        return _StructFake()


class _FinalMsg:
    content = "这是要发送的报告。"


class _FinalModel:
    async def ainvoke(self, messages):
        return _FinalMsg()


@pytest.mark.asyncio
async def test_graph_send_e2e(monkeypatch):
    monkeypatch.setattr(co, "chat_model", _CoordModel())
    monkeypatch.setattr(fz, "chat_model", _FinalModel())

    async def _fake_send_node(state):
        assert state["final_answer"] == "这是要发送的报告。"
        return {
            "send_report": "已通过 Gmail 发送给张三。",
            "trace": [{"agent": "send", "status": "done"}],
        }

    monkeypatch.setattr(snd, "send_node", _fake_send_node)
    # build 里 send 节点是通过引用 snd.send_node 注入的，所以还要 patch build 模块引用。
    import app.agent.graph.build as build_mod
    monkeypatch.setattr(build_mod, "send_node", _fake_send_node)

    runner = GraphRunner()
    events = [
        e
        async for e in runner.stream(
            "把刚才报告发给张三", history=[], identity=None, session_id="sid"
        )
    ]

    tokens = "".join(e["data"] for e in events if e["type"] == "token")
    done = events[-1]

    assert "这是要发送的报告。" in tokens
    assert "发送结果：已通过 Gmail 发送给张三。" in tokens
    assert done["send_report"] == "已通过 Gmail 发送给张三。"
