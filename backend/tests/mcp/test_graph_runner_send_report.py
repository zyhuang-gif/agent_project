import pytest

from app.agent.graph.runner import GraphRunner


@pytest.mark.asyncio
async def test_graph_runner_appends_visible_send_report(monkeypatch):
    runner = GraphRunner()

    class _Chunk:
        def __init__(self, c):
            self.content = c

    async def _astream(state, stream_mode=None):
        yield ("messages", (_Chunk("最终回答"), {"langgraph_node": "finalize"}))
        yield (
            "values",
            {"final_answer": "最终回答", "send_report": "已通过 Gmail 发送给张三。"},
        )

    class _FakeGraph:
        def astream(self, state, stream_mode=None):
            return _astream(state, stream_mode=stream_mode)

    monkeypatch.setattr(runner, "_graph", _FakeGraph())
    events = [e async for e in runner.stream("q", history=[], identity=None)]

    tokens = "".join(e["data"] for e in events if e["type"] == "token")
    assert "最终回答" in tokens
    assert "发送结果：已通过 Gmail 发送给张三。" in tokens
    assert events[-1]["send_report"] == "已通过 Gmail 发送给张三。"
