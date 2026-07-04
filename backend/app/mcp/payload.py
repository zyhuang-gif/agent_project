"""把 final_answer 包装成可发送的 send_payload。

设计目的（spec §2.4、§6）：
- 不能裸发 final_answer。P0 无附件，但 subject/body 需要固定的框架文案。
- source_answer_preview 保留原答案摘要，供审计和调试。
- attachments 在 P1 由 send_node 在 preflight 通过后生成 Markdown artifact，
  并把绝对路径写进 tool_args，SendPayload 本身仍是 frozen dataclass，不修改。
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SendPayload:
    subject: str
    body_text: str
    source_answer_preview: str
    attachments: list[str] = field(default_factory=list)


def _compact(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


async def build_send_payload(source_answer: str, original_query: str, recipients: list) -> SendPayload:
    query_preview = _compact(original_query, 42) or "消息"
    subject = f"企业知识库 Agent：{query_preview}"
    body_text = (
        "以下内容由企业知识库 Agent 生成，请在正式使用前按需核对。\n\n"
        f"{source_answer or ''}\n\n"
        "—— 由 LangChain RAG FastAPI Service 自动发送"
    )
    return SendPayload(
        subject=_compact(subject, 80),
        body_text=body_text,
        source_answer_preview=_compact(source_answer, 200),
        attachments=[],
    )
