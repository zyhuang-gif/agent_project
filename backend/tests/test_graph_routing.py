from app.agent.graph.nodes.coordinator import route_after_coordinator


def test_route_to_knowledge_when_need_retrieval():
    state = {"plan": {"task_type": "knowledge_qa", "need_retrieval": True}}
    assert route_after_coordinator(state) == "knowledge"


def test_route_to_finalize_when_no_retrieval():
    state = {"plan": {"task_type": "knowledge_qa", "need_retrieval": False}}
    assert route_after_coordinator(state) == "finalize"


def test_route_defaults_to_knowledge_when_plan_missing():
    # 没有 plan 时保守走检索（不漏检索）
    assert route_after_coordinator({}) == "knowledge"


import pytest


@pytest.mark.asyncio
async def test_graph_skips_retrieval_when_no_retrieval(monkeypatch):
    """coordinator 判定 need_retrieval=False 时，图不应触发知识检索。"""
    import app.agent.graph.nodes.coordinator as co
    import app.agent.graph.nodes.knowledge as kn
    import app.agent.graph.nodes.finalize as fz

    class _Msg:
        def __init__(self, c):
            self.content = c

    async def _fake_final_invoke(_messages):
        return _Msg("你好，我能帮你查询企业知识。")

    rag_called = {"hit": False}

    async def _fake_get_docs(query, filter_meta=None):
        rag_called["hit"] = True
        return {"documents": [], "citations": [], "summary": "", "error": None}

    class _FakeCoordModel:
        def with_structured_output(self, schema, include_raw=False):
            class _Structured:
                async def ainvoke(self, _messages):
                    return {
                        "raw": _Msg(""),
                        "parsed": schema(
                            task_type="knowledge_qa",
                            need_retrieval=False,
                            reason="闲聊",
                        ),
                        "parsing_error": None,
                    }
            return _Structured()

    class _FakeFinalModel:
        ainvoke = staticmethod(_fake_final_invoke)

    monkeypatch.setattr(co, "chat_model", _FakeCoordModel())
    monkeypatch.setattr(fz, "chat_model", _FakeFinalModel())
    monkeypatch.setattr(kn.rag_service, "get_documents_for_agent", _fake_get_docs)

    from app.agent.graph.build import build_graph
    graph = build_graph()
    result = await graph.ainvoke({"query": "你好啊", "history": [], "trace": []})

    assert rag_called["hit"] is False                 # 检索未被触发
    assert result["plan"]["need_retrieval"] is False
    assert result["final_answer"] == "你好，我能帮你查询企业知识。"


from app.agent.graph.build import route_after_finalize


def test_route_after_finalize_goes_to_send_when_intent():
    assert route_after_finalize({"plan": {"send_intent": True}}) == "send"


def test_route_after_finalize_ends_without_send_intent():
    assert route_after_finalize({"plan": {"send_intent": False}}) == "end"
