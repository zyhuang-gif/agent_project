import json
import os
import time
from typing import List, Optional, AsyncGenerator

import httpx
import openai
from langsmith import traceable

from app.agent.agent_middleware import (
    on_agent_start, on_agent_end,
    on_tool_call, on_tool_result,
    on_model_call, on_model_response,
)
from app.agent.agent_tools import TOOLS, TOOL_SCHEMAS, get_rag_citations
from app.agent.graph.runner import graph_runner
from app.core.logger_handler import logger
from app.services import session_manager as sm
from app.utils.auth_utils import RequestIdentity
from app.utils.prompt_loader import load_prompt

# ── token 估算 ────────────────────────────────────────────────────────────────
# 流式过程中用 tiktoken 粗估 token 数（qwen 非精确，仅用于实时跳动），
# 每轮 LLM 调用结束后由 API 返回的精确 usage 校准。
# 估算函数已抽到 app.agent.token_utils（GraphRunner 共用）。
from app.agent.token_utils import (
    estimate_text_tokens as _estimate_text_tokens,
    estimate_messages_tokens as _estimate_messages_tokens,
)


TOOL_STEP_TITLES = {
    "rag_summary_tools": "已检索知识库",
    "reorder_documents_tools": "已完成文档重排序",
    "get_user_info_tools": "已读取用户信息",
    "get_weather_tools": "已调用天气工具",
    "what_time_is_now": "已读取当前时间",
}

TOOL_PLAN_TITLES = {
    "rag_summary_tools": "检索相关知识库",
    "reorder_documents_tools": "重排序候选文档",
    "get_user_info_tools": "读取用户信息",
    "get_weather_tools": "调用天气工具",
    "what_time_is_now": "读取当前时间",
}

TOOL_RUNNING_DETAILS = {
    "rag_summary_tools": "正在检索相关知识库",
    "reorder_documents_tools": "正在重排序候选文档",
    "get_user_info_tools": "正在读取用户信息",
    "get_weather_tools": "正在调用天气工具",
    "what_time_is_now": "正在读取当前时间",
}


def _max_tool_rounds() -> int:
    """工具调用循环的最大轮数上限（防止模型持续请求工具导致无限循环/烧 token）。"""
    try:
        return max(1, int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "8")))
    except (TypeError, ValueError):
        return 8


def _format_agent_step(step: dict) -> dict:
    """把内部工具调用记录转换成前端可展示的进度事件。"""
    tool = step.get("tool") or "unknown_tool"
    output = str(step.get("tool_output") or "")
    failed = "失败" in output or "异常" in output
    detail = output[:120]
    if tool == "rag_summary_tools" and step.get("citation_count") is not None:
        detail = f"已检索 {step['citation_count']} 个文档"
    return {
        "id": f"tool_{tool}",
        "title": TOOL_STEP_TITLES.get(tool, f"已调用工具：{tool}"),
        "status": "done",
        "level": "warning" if failed else "success",
        "detail": detail,
    }


def _build_agent_plan(tool_calls: list[dict] = None, answer_status: str = "todo") -> list[dict]:
    """生成可公开展示的执行计划，不暴露模型隐藏推理。"""
    steps = [
        {
            "id": "task_understood",
            "title": "理解用户问题",
            "status": "done",
            "level": "success",
        }
    ]
    for tc in tool_calls or []:
        tool = tc.get("name") or "unknown_tool"
        steps.append({
            "id": f"tool_{tool}",
            "title": TOOL_PLAN_TITLES.get(tool, f"调用工具：{tool}"),
            "status": "todo",
            "level": "muted",
        })
    steps.append({
        "id": "answer_generated",
        "title": "生成最终回答",
        "status": answer_status,
        "level": "info" if answer_status == "running" else "muted",
    })
    return steps


def _format_step_update(
    step_id: str,
    status: str,
    level: str = None,
    detail: str = None,
    title: str = None,
) -> dict:
    update = {"id": step_id, "status": status}
    if level:
        update["level"] = level
    if detail:
        update["detail"] = detail
    if title:
        update["title"] = title
    return update


class AgentLoop:
    """
    自实现的 Tool Calling Agent 循环。

    设计原则：
    - 实例无状态（history/query 每次传入），全局单例安全复用
    - 使用 OpenAI SDK 访问 DashScope / Ollama 的兼容接口，获得真 token 流
    - 中间件钩子（agent_middleware.py）挂载在关键节点，不侵入主循环
    """

    def __init__(self):
        self._client: Optional[openai.AsyncOpenAI] = None
        self.system_prompt: str = load_prompt("main_prompt")
        self.tools: dict = TOOLS
        self.tool_schemas: list = TOOL_SCHEMAS

    # ── 客户端懒初始化 ────────────────────────────────────────────────────────

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> openai.AsyncOpenAI:
        llm_type = os.getenv("LLM_TYPE", "ALIYUN").upper()
        # 读超时是「两次 chunk 之间」的最大间隔（不是整段流的总时长），
        # 故 60s 不会截断长回答；连接超时单独设短一些；失败重试 2 次。
        timeout = httpx.Timeout(float(os.getenv("LLM_TIMEOUT", "60")), connect=10.0)
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        if llm_type == "OLLAMA":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
            return openai.AsyncOpenAI(
                base_url=base_url, api_key="ollama",
                timeout=timeout, max_retries=max_retries,
            )
        return openai.AsyncOpenAI(
            base_url=os.getenv(
                "ALIYUN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            api_key=os.getenv("ALIYUN_ACCESS_KEY_SECRET"),
            timeout=timeout,
            max_retries=max_retries,
        )

    def _model_name(self) -> str:
        llm_type = os.getenv("LLM_TYPE", "ALIYUN").upper()
        if llm_type == "OLLAMA":
            return os.getenv("OLLAMA_MODEL_NAME", "qwen3:7b")
        return os.getenv("ALIYUN_MODEL_NAME", "qwen3-max")

    # ── 消息构建 ──────────────────────────────────────────────────────────────

    def _build_messages(self, history: list, query: str) -> list:
        messages = [{"role": "system", "content": self.system_prompt}]
        for user_msg, assistant_msg in (history or []):
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": query})
        return messages

    # ── 工具执行 ──────────────────────────────────────────────────────────────

    async def _execute_tool(
        self, name: str, arguments: str, identity: "Optional[RequestIdentity]" = None
    ) -> str:
        tool_fn = self.tools.get(name)
        if tool_fn is None:
            return f"工具 {name} 不存在"
        t0 = time.perf_counter()
        try:
            args = json.loads(arguments) if arguments.strip() else {}
            # rag_summary_tools 需要请求级身份做权限过滤，由调用方显式注入 identity
            # （不走 ContextVar，避免跨异步生成器边界丢值导致越权检索）
            if name == "rag_summary_tools":
                args["identity"] = identity
            return str(await tool_fn(**args))
        except Exception as e:
            logger.error(f"[Tool] {name} 执行异常: {e}", exc_info=True)
            return f"工具执行失败: {e}"
        finally:
            logger.info(
                f"[Timing][Agent] stage=tool_execute tool={name} "
                f"duration={time.perf_counter() - t0:.3f}s"
            )

    # ── 流式执行 ──────────────────────────────────────────────────────────────

    @traceable
    async def stream(
        self, query: str, history: list = None, identity: "Optional[RequestIdentity]" = None
    ) -> AsyncGenerator[dict, None]:
        """
        真 token 级流式执行。

        yield 事件：
          {"type": "token", "data": str}             —— LLM 输出 token
          {"type": "agent_plan", "data": list}        —— 可公开执行计划
          {"type": "agent_step_update", "data": dict} —— 执行计划状态更新
          {"type": "step",  "data": {tool, ...}}      —— 工具调用记录
          {"type": "usage", "tokens": int, ...}      —— 实时 token 估算
          {"type": "done",  "steps": list, "tokens"} —— 全部完成（精确总数）
        """
        yield {
            "type": "agent_plan",
            "data": _build_agent_plan([], answer_status="todo"),
        }
        messages = self._build_messages(history, query)
        steps = []
        on_agent_start(messages)

        committed_tokens = 0  # 已完成各轮 LLM 调用的精确 total_tokens 之和
        collected_citations: list = []  # 工具执行时同步固化的结构化引用

        max_rounds = _max_tool_rounds()
        stream_total_t0 = time.perf_counter()
        for round_idx in range(max_rounds + 1):
            on_model_call(messages)
            if round_idx > 0:
                yield {
                    "type": "agent_step_update",
                    "data": _format_step_update(
                        "answer_generated",
                        "running",
                        "info",
                        "正在基于已完成步骤生成回答",
                    ),
                }
            # 最后一轮禁用工具，强制模型输出文字答复，保证循环必然终止
            last_round = round_idx == max_rounds
            prompt_est = _estimate_messages_tokens(messages)
            # 思考期先发一个初始估算（已消耗的输入 token），让前端立刻有数字开始增长
            yield {"type": "usage", "tokens": committed_tokens + prompt_est, "estimated": True}
            model_t0 = time.perf_counter()
            stream = await self.client.chat.completions.create(
                model=self._model_name(),
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="none" if last_round else "auto",
                stream=True,
                stream_options={"include_usage": True},
            )

            # 累积本轮输出
            content_buf = ""
            tool_calls_buf: dict[int, dict] = {}  # chunk index -> {id, name, args}
            turn_usage_total: Optional[int] = None
            last_emit_len = 0
            first_chunk_at: Optional[float] = None

            async for chunk in stream:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                # include_usage 的最后一帧 choices 为空、仅带 usage
                if getattr(chunk, "usage", None) is not None:
                    turn_usage_total = chunk.usage.total_tokens
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    content_buf += delta.content
                    yield {"type": "token", "data": delta.content}

                    # 节流：内容每增长约 20 字符发一次实时估算
                    if len(content_buf) - last_emit_len >= 20:
                        last_emit_len = len(content_buf)
                        running = committed_tokens + prompt_est + _estimate_text_tokens(content_buf)
                        yield {"type": "usage", "tokens": running, "estimated": True}

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buf:
                            tool_calls_buf[idx] = {"id": "", "name": "", "args": ""}
                        if tc.id:
                            tool_calls_buf[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_buf[idx]["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls_buf[idx]["args"] += tc.function.arguments

            # 本轮结束：用精确 usage 累加；拿不到则用估算兜底
            if turn_usage_total is not None:
                committed_tokens += turn_usage_total
            else:
                committed_tokens += prompt_est + _estimate_text_tokens(content_buf)

            on_model_response(bool(tool_calls_buf), len(content_buf))
            model_duration = time.perf_counter() - model_t0
            first_chunk_duration = (
                first_chunk_at - model_t0 if first_chunk_at is not None else None
            )
            first_chunk_text = (
                f"{first_chunk_duration:.3f}s" if first_chunk_duration is not None else "none"
            )
            logger.info(
                f"[Timing][Agent] stage=model_turn round={round_idx + 1} "
                f"tool_calls={len(tool_calls_buf)} content_chars={len(content_buf)} "
                f"first_chunk={first_chunk_text} duration={model_duration:.3f}s"
            )

            if not tool_calls_buf:
                break  # 无工具调用，流式输出已完成

            yield {
                "type": "agent_plan",
                "data": _build_agent_plan(list(tool_calls_buf.values())),
            }

            # 追加 assistant 消息（含 tool_calls 结构）
            tool_calls_list = [
                {
                    "id": v["id"],
                    "type": "function",
                    "function": {"name": v["name"], "arguments": v["args"]},
                }
                for v in tool_calls_buf.values()
            ]
            messages.append({
                "role": "assistant",
                "content": content_buf or None,
                "tool_calls": tool_calls_list,
            })

            # 顺序执行各工具
            for tc in tool_calls_buf.values():
                on_tool_call(tc["name"], tc["args"])
                yield {
                    "type": "agent_step_update",
                    "data": _format_step_update(
                        f"tool_{tc['name']}",
                        "running",
                        "info",
                        TOOL_RUNNING_DETAILS.get(tc["name"], f"正在调用工具：{tc['name']}"),
                    ),
                }
                result = await self._execute_tool(tc["name"], tc["args"], identity)
                on_tool_result(tc["name"], result)

                # 紧邻工具执行点同步读取 citations 并固化到局部变量；
                # 此处与工具内 set() 处于同一执行段（无 yield/无新 task），ContextVar 可靠
                citation_count = None
                if tc["name"] == "rag_summary_tools":
                    cites = get_rag_citations()
                    if cites:
                        collected_citations = cites
                    citation_count = len(cites or [])

                step = {
                    "tool": tc["name"],
                    "tool_input": tc["args"],
                    "tool_output": result,
                    "citation_count": citation_count,
                }
                steps.append(step)
                yield {"type": "agent_step_update", "data": _format_agent_step(step)}
                yield {"type": "step", "data": step}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        on_agent_end("", steps)
        logger.info(
            f"[Timing][Agent] stage=stream_total rounds={round_idx + 1} "
            f"steps={len(steps)} duration={time.perf_counter() - stream_total_t0:.3f}s"
        )
        yield {
            "type": "agent_step_update",
            "data": _format_step_update(
                "answer_generated",
                "done",
                "success",
                "已生成最终回答",
                "生成最终回答",
            ),
        }
        yield {"type": "done", "steps": steps, "tokens": committed_tokens, "citations": collected_citations}


# ── 全局单例（实例无状态，多请求安全复用）────────────────────────────────────
agent_loop = AgentLoop()


# ── 对外接口 ────────────────────────────────────────────────────────────────

@traceable
async def get_agent_stream_response(
    query: str,
    session_id: str,
    identity: "RequestIdentity",
    **kwargs,
) -> AsyncGenerator[str, None]:
    """SSE 流式调用，yield Server-Sent Events 格式字符串"""
    total_t0 = time.perf_counter()
    user_id = identity.user_id
    history_t0 = time.perf_counter()
    history = await sm.session_manager.get_history(session_id, user_id)
    logger.info(
        f"[Timing][AgentSSE] stage=history_load session={session_id} "
        f"duration={time.perf_counter() - history_t0:.3f}s"
    )
    logger.info(f"[Agent流式] 开始 user={user_id} session={session_id}")

    # 初始帧：告知客户端 session_id
    yield f"data: {json.dumps({'type': 'response', 'content': '', 'session_id': session_id}, ensure_ascii=False)}\n\n"

    full_response: list[str] = []
    steps: list = []
    total_tokens = 0
    citations: list = []
    send_report: Optional[str] = None
    # 累积每条 agent step 的最终态(以 step.id 为 key 合并 update),
    # 以便流结束后随 assistant 消息一起落库,前端切回会话能还原
    agent_steps_state: dict[str, dict] = {}

    engine = os.getenv("AGENT_ENGINE", "loop").strip().lower()
    if engine == "graph":
        event_source = graph_runner.stream(query, history, identity=identity, session_id=session_id)
    else:
        event_source = agent_loop.stream(query, history, identity=identity)

    async for event in event_source:
        if event["type"] == "token":
            full_response.append(event["data"])
            yield f"data: {json.dumps({'type': 'response', 'content': event['data']}, ensure_ascii=False)}\n\n"
        elif event["type"] == "usage":
            yield f"data: {json.dumps({'type': 'usage', 'tokens': event['tokens'], 'estimated': event.get('estimated', True)}, ensure_ascii=False)}\n\n"
        elif event["type"] == "agent_plan":
            for s in event.get("data") or []:
                if isinstance(s, dict) and s.get("id"):
                    agent_steps_state[s["id"]] = dict(s)
            yield f"data: {json.dumps({'type': 'agent_plan', 'data': event['data']}, ensure_ascii=False)}\n\n"
        elif event["type"] == "agent_step_update":
            s = event.get("data") or {}
            if isinstance(s, dict) and s.get("id"):
                agent_steps_state.setdefault(s["id"], {}).update(s)
            yield f"data: {json.dumps({'type': 'agent_step_update', 'data': event['data']}, ensure_ascii=False)}\n\n"
        elif event["type"] == "agent_step":
            s = event.get("data") or {}
            if isinstance(s, dict) and s.get("id"):
                agent_steps_state.setdefault(s["id"], {}).update(s)
            yield f"data: {json.dumps({'type': 'agent_step', 'data': event['data']}, ensure_ascii=False)}\n\n"
        elif event["type"] == "step":
            steps.append(event["data"])
            formatted = _format_agent_step(event['data'])
            if formatted.get("id"):
                agent_steps_state.setdefault(formatted["id"], {}).update(formatted)
            yield f"data: {json.dumps({'type': 'agent_step', 'data': formatted}, ensure_ascii=False)}\n\n"
        elif event["type"] == "done":
            steps = event["steps"]
            total_tokens = event.get("tokens", 0)
            citations = event.get("citations", [])
            send_report = event.get("send_report")

    response = "".join(full_response) or "抱歉，我无法理解您的请求。"
    save_t0 = time.perf_counter()
    final_agent_steps = list(agent_steps_state.values())
    await sm.session_manager.add_message(
        session_id, user_id, query, response,
        citations=citations,
        steps=final_agent_steps,
        send_report=send_report,
    )
    logger.info(
        f"[Timing][AgentSSE] stage=history_save session={session_id} "
        f"duration={time.perf_counter() - save_t0:.3f}s"
    )

    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'citations': citations, 'tokens': total_tokens, 'send_report': send_report}, ensure_ascii=False)}\n\n"
    logger.info(
        f"[Timing][AgentSSE] stage=request_total session={session_id} "
        f"response_chars={len(response)} duration={time.perf_counter() - total_t0:.3f}s"
    )
    logger.info(f"[Agent流式] 完成 session={session_id} steps={len(steps)} citations={len(citations)} tokens={total_tokens}")
