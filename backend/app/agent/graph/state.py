import operator
from typing import Annotated, Optional, TypedDict

from app.utils.auth_utils import RequestIdentity


class AgentState(TypedDict, total=False):
    """LangGraph 全图共享状态。

    约束（见设计文档 §4）：
    - identity 是权限对象，只在内存中流转，绝不持久化（一期不启用 checkpointer）。
    - trace 用 operator.add reducer，并行节点各自 append 不互相覆盖。
    """
    # 输入
    query: str
    history: list                       # [(user, assistant), ...]
    identity: Optional[RequestIdentity]
    session_id: str                     # send 节点写审计要用；不影响主问答

    # Coordinator 产出
    plan: dict                          # {task_type, need_retrieval, reason,
                                        #  send_intent, recipients, channels}

    # Knowledge 产出
    documents: list                     # list[str]
    citations: list                     # list[dict]，复用现有 citations 结构
    is_enough: bool
    max_score: float                    # 检索最高相关度；critic 评估证据 / 判断是否阈值过严

    # Task 产出
    task_messages: list                 # finalize 用的任务专属 LLM messages；空则走默认问答

    # 输出
    final_answer: str

    # Send 产出（P0 新增）
    send_payload: object                # app.mcp.payload.SendPayload
    send_report: str                    # 前端可见的发送结果文案

    # Send Artifact（P1 新增，Markdown 附件；未生成时字段缺失）
    artifact_path: str
    artifact_mime: str
    artifact_name: str

    # token 计量（各节点 append 本次 LLM 调用的 total，reducer 累加）
    token_usage: Annotated[int, operator.add]

    # 轨迹（append-only）
    trace: Annotated[list, operator.add]

    # ── Critic 反馈回路（本计划新增）──
    # critic 触发改写重做时 +1，reducer 累加（与 trace/token_usage 一致，兼容并行）
    revision_count: Annotated[int, operator.add]
    # 最近一次 critic 输出 {verdict, reason, reformulated_query}；每次覆盖（无 reducer）
    critic_verdict: dict
    # critic 给出的改写 query；knowledge 重检索时优先用；每次覆盖（无 reducer）
    reformulated_query: str
