import json
from typing import Literal

from pydantic import BaseModel

from app.agent.graph._stream import safe_get_stream_writer
from app.agent.graph.state import AgentState
from app.utils.factory import chat_model, get_chat_model

_DEFAULT_CHAT_MODEL = chat_model

# 合法任务类型（与设计文档一致）；非法值归一为 unknown
_TASK_TYPES = {
    "knowledge_qa", "document_compare", "document_generation",
    "report_generation", "knowledge_gap", "unknown",
}

_COORDINATOR_PROMPT = """你是企业知识库 Agent 的任务协调器。判断用户问题属于哪种任务类型，是否需要检索知识库，以及是否要把结果发送出去。

只输出一个 JSON 对象，不要任何额外解释，格式：
{"task_type": "<类型>", "need_retrieval": <true|false>, "reason": "<简短中文理由>", "send_intent": <true|false>, "recipients": ["<联系人>"], "channels": ["gmail|feishu"]}

task_type 取值：
- knowledge_qa：普通知识问答
- document_compare：多文档/新旧版对比
- document_generation：生成申请/说明等文本
- report_generation：生成结构化报告
- knowledge_gap：明显超出知识库范围、需记录缺口
- unknown：无法识别

need_retrieval：除非是与企业知识完全无关的闲聊，否则一律为 true。
send_intent：用户明确说"发给 / 转给 / 抄送 / 通知 / 发到群"时为 true；只是生成报告或回答问题时为 false。
recipients：只抽取用户明确说出的自然语言收件人，如"张三""运营组"；没有则为空数组。
channels：用户明确指定邮件/Gmail 时写 "gmail"，明确指定飞书/群时写 "feishu"；未指定渠道时为空数组，由后端 P0 默认 Gmail。"""

# 兜底 plan：分类失败时保守走检索且不发送
_FALLBACK_PLAN = {
    "task_type": "knowledge_qa",
    "need_retrieval": True,
    "reason": "分类失败，默认走检索",
    "send_intent": False,
    "recipients": [],
    "channels": [],
}


class CoordinatorPlan(BaseModel):
    task_type: Literal[
        "knowledge_qa", "document_compare", "document_generation",
        "report_generation", "knowledge_gap", "unknown",
    ]
    need_retrieval: bool
    reason: str
    send_intent: bool = False
    recipients: list[str] = []
    channels: list[str] = []


def _plan_to_dict(plan_obj: CoordinatorPlan | dict) -> dict:
    """把结构化输出对象归一为图状态中的 plan 字典。"""
    data = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else dict(plan_obj)
    task_type = data.get("task_type")
    if task_type not in _TASK_TYPES:
        task_type = "unknown"
    channels = [str(c).strip().lower() for c in data.get("channels", []) if str(c).strip()]
    channels = [c for c in channels if c in ("gmail", "feishu")]
    recipients = [str(r).strip() for r in data.get("recipients", []) if str(r).strip()]
    return {
        "task_type": task_type,
        "need_retrieval": bool(data.get("need_retrieval", True)),
        "reason": str(data.get("reason", "")),
        "send_intent": bool(data.get("send_intent", False)),
        "recipients": recipients,
        "channels": channels,
    }


async def coordinator_node(state: AgentState) -> dict:
    """判定任务类型与是否需要检索。LLM token 因节点名非 finalize 被 bridge 过滤，不泄漏。"""
    writer = safe_get_stream_writer()
    writer({"kind": "step", "id": "task_understood", "status": "running",
            "level": "info", "detail": "正在识别任务类型", "title": "识别任务类型"})

    # 仅取最近一轮做指代消解（"那它呢"），避免长历史干扰分类的 JSON 输出
    context = ""
    history = state.get("history") or []
    if history and isinstance(history[-1], (list, tuple)) and len(history[-1]) == 2:
        last_u, last_a = history[-1]
        context = f"（上一轮——用户：{last_u}；助手：{last_a}）\n"
    messages = [
        {"role": "system", "content": _COORDINATOR_PROMPT},
        {"role": "user", "content": f"{context}当前问题：{state['query']}"},
    ]
    try:
        model = chat_model if chat_model is not _DEFAULT_CHAT_MODEL else get_chat_model("coordinator")
        structured = model.with_structured_output(CoordinatorPlan, include_raw=True)
        result = await structured.ainvoke(messages)
        if result.get("parsing_error"):
            raise result["parsing_error"]
        msg = result.get("raw")
        plan = _plan_to_dict(result["parsed"])
        plan_status = "done"
        from app.agent.token_utils import extract_total_tokens, estimate_messages_tokens
        coord_tokens = extract_total_tokens(msg) or (
            estimate_messages_tokens(messages) + estimate_messages_tokens([{"content": json.dumps(plan)}])
        )
    except Exception as e:
        from app.core.logger_handler import logger
        logger.error(f"[Coordinator] 分类失败，降级走检索: {e}", exc_info=True)
        plan = dict(_FALLBACK_PLAN)
        plan_status = "failed"
        coord_tokens = 0

    writer({"kind": "step", "id": "task_understood", "status": "done",
            "level": "success", "detail": f"识别为：{plan['task_type']}",
            "title": "已识别任务类型"})

    return {
        "plan": plan,
        "token_usage": coord_tokens,
        "trace": [{"agent": "coordinator", "status": plan_status,
                   "output": json.dumps(plan, ensure_ascii=False)}],
    }


def route_after_coordinator(state: AgentState) -> str:
    """条件边：need_retrieval 为真走 knowledge，否则直接 finalize。缺 plan 时保守走 knowledge。"""
    plan = state.get("plan") or {}
    return "knowledge" if plan.get("need_retrieval", True) else "finalize"
