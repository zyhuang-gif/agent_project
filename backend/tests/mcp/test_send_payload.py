import pytest

from app.mcp.payload import build_send_payload
from app.mcp.preflight import ResolvedRecipient


@pytest.mark.asyncio
async def test_build_send_payload_wraps_final_answer():
    payload = await build_send_payload(
        source_answer="这是最终报告正文。",
        original_query="根据薪酬制度生成报告",
        recipients=[
            ResolvedRecipient(name="张三", email="zhangsan@example.com", channel="gmail")
        ],
    )

    assert payload.subject.startswith("企业知识库 Agent：")
    assert "根据薪酬制度生成报告" in payload.subject
    assert payload.body_text != "这是最终报告正文。"
    assert "以下内容由企业知识库 Agent 生成" in payload.body_text
    assert "这是最终报告正文。" in payload.body_text
    assert payload.attachments == []
    assert payload.source_answer_preview == "这是最终报告正文。"
