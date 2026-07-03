# MCP 消息通道接入设计（Gmail P0 + 飞书 P2）

> 状态：设计稿 · 待实现
> 作者：黄职勇（与 Claude 结对 brainstorm）
> 日期：2026-07-03

## 1. 背景与目标

现有 RAG Agent 只完成"检索 + 生成"。用户希望在生成产出物（对比、报告、申请）之后，能在同一段对话里让 Agent **自动发送**这些产出物给指定的人或飞书群。

典型话术：

> "把刚才那份薪酬报告发给张三，同时发到运营群。"

本设计要解决的是**接入邮件与飞书两条通道**这件事，选用 **MCP（Model Context Protocol）** 作为工具接入形态，原因：

- Gmail P0 自建最小 MCP server，只暴露发送能力，避免把 archived / 高权限社区包放进主链路；
- 飞书自建 MCP server，能力可控、可复用给其他 Agent（Claude Desktop、Cursor 等）；
- 联系人解析（`contacts`）也做成 MCP server，被 gmail、feishu 两条通道共享。

**非目标**：

- 不做 per-user OAuth：P0 不让每个登录用户分别绑定自己的 Gmail / 飞书账号，统一使用一套系统发件身份（沙箱 Gmail / 公司机器人）；
- 不做完整客户端级幂等 / 去重：P0 不新增 `send_request` 去重表；发送动作靠 preflight、单次 tool 调用约束和审计日志降低重复风险；
- 不接入短信、微信、Slack、Teams（YAGNI）；
- P0 不做附件文件生成 / 附件发送；`final_answer` 只作为发送内容来源，由 `send` 节点生成专用 `send_payload`（subject + body）后发送；
- 不做"读邮箱 / 删邮件 / 管理标签或过滤器"等敏感能力，Gmail MCP server 从实现上就不暴露这些工具。

## 2. 术语与选型澄清

### 2.1 MCP 相关

- **MCP Server**：对外提供工具的一方。本设计里的 gmail（自写最小 Python server）、feishu（自写）、contacts（自写）都是 MCP server。
- **MCP Client**：调用别人 MCP Server 的一方。本项目后端 Agent 就是 Client。
- **`langchain-mcp-adapters`**：把 MCP 工具适配成 LangChain `BaseTool`，可直接被 LangGraph 节点或 `create_react_agent` 消费。

### 2.2 编排范式 vs 通信协议（两者正交，别混）

- **编排范式**（LLM 如何决定下一步做什么）：
  - **ReAct**：LLM 循环"思考 → 行动 → 观察 → …"直到给出最终答案。**LLM 主导流程**。
  - **DAG / 固定流水线**：节点顺序由预定义的图路由决定，LLM 只在节点内做局部决策。**图路由主导流程**。
  - Plan-and-Execute / Reflexion / Multi-Agent 是别的编排范式，本设计不涉及。
- **通信协议**（LLM 如何把"我要调 X 工具"表达出来）：
  - **纯文本解析**：`Action: xxx / Action Input: {...}` 由代码正则解析（ReAct 原论文写法，过时）。
  - **Function Calling / Tool Use**：LLM 直接输出结构化 `tool_calls` 字段（OpenAI / Anthropic / DashScope 现代 API 通用）。
- **LangGraph 的 `create_react_agent`** = **ReAct 编排 + Function Calling 表达**。不是原论文的正则版 ReAct，是现代 API 版。

### 2.3 本项目现状定位

- **loop 引擎（`agent/agent.py`）**：ReAct 编排（手写 while 循环）+ Function Calling 表达。**等价于自写版 `create_react_agent`**。工具协议是 OpenAI schema（`TOOLS` dict + `TOOL_SCHEMAS` list）。
- **graph 引擎（`agent/graph/`）**：DAG 编排（Coordinator → Knowledge → Task/Gap → Finalize），节点内**不调 tool**——现有"任务工具"（compare / report / form）是节点内拼提示，不是 LLM tool_calls。
- **本 P0 的取舍**：**只在 graph 引擎里承载发送能力**（见 § 3）。loop 引擎不注入 MCP 工具；P0 实现时把 `AGENT_ENGINE` 默认值切到 `graph`，loop 收敛/移除另起 spec（避免双引擎双向适配的额外成本）。

### 2.4 P0 发送载荷与安全取舍

- **系统发件身份**：P0 使用统一 Gmail 沙箱账号发出邮件，审计中记录触发发送的 `user_id/session_id`。这不是最终权限模型；正式多人使用前需要 per-user OAuth 或明确的组织级机器人发件策略。
- **轻量防重复**：P0 不建幂等表，但必须保证一次 `send_node` 只调用一次 `gmail_send_email`，并把每次成功写入 `sends.log`。如果后续出现前端重试、后端超时重跑、批量发送等场景，再引入 `send_request_id` + `send_audit` 表做强幂等。
- **专用发送载荷**：P0 无附件，但不能裸发 `final_answer`。`final_answer` 只是内容来源，`send` 节点必须生成 `send_payload`：`subject`、`body_text`（或 `body_markdown`）、`source_answer_preview`、`attachments=[]`。Gmail 工具只接收 `send_payload` 中的 subject/body。

## 3. 总体架构

**P0 只走 graph 引擎，只支持 Gmail 正文发送**——在图上加一个 `send` 节点作为发送能力承载点。`send` 节点先做确定性 preflight（联系人解析、通道可用性、是否需要确认），再基于 `final_answer` 生成专用 `send_payload`（subject + body），最后才把已解析好的收件人和 payload 交给局部 `create_react_agent` 调用 MCP 发送工具。

```text
┌────────────────────────────────────────────────────────────────┐
│ FastAPI backend (existing, port 8000)                          │
│                                                                │
│   AGENT_ENGINE=graph（P0 实现时切为默认）                       │
│                                                                │
│   Coordinator → Knowledge → [Task | Gap] → Finalize            │
│                                    │                           │
│                                    ▼                           │
│                          ┌───────────────────────┐             │
│                          │  send 节点（新增）      │             │
│                          │                       │             │
│                          │  1. deterministic     │             │
│                          │     preflight         │             │
│                          │  2. build send_payload│             │
│                          │  3. create_react_agent│             │
│                          │     with send tools   │             │
│                          │  4. send_report       │             │
│                          └───────────────────────┘             │
│                                    │                           │
│                                    ▼                           │
│                                  END                           │
│                                                                │
│   ┌──────────────────────────────────────┐                     │
│   │ MCPClientManager  (app/mcp/client.py) │                     │
│   │  - startup 读 mcp_servers.yaml         │                     │
│   │  - MultiServerMCPClient 起子进程       │                     │
│   │  - tool_name_prefix=true              │                     │
│   │  - include_tools 白名单过滤            │                     │
│   │  - send_guard ToolCallInterceptor     │                     │
│   │  - shutdown 清理子进程                │                     │
│   └──────────────────────────────────────┘                     │
└────────────────────────────────────────────────────────────────┘
         │ stdio                  │ stdio                 │ stdio(P2)
         ▼                        ▼                       ▼
  ┌────────────────┐       ┌────────────────┐      ┌────────────────┐
  │ gmail MCP      │       │ contacts MCP   │      │ feishu MCP     │
  │ 自写 Python     │       │ 自写 Python     │      │ 自写 Python     │
  │ send_email only│       │ MySQL contact  │      │ P2 启用         │
  └────────────────┘       └────────────────┘      └────────────────┘
   Google OAuth              contact/group 表        app_id/app_secret
```

**关键决定**：

1. **P0 只支持 graph 引擎**——避免为 loop 侧多写一层 OpenAI schema ↔ BaseTool 双向适配；P0 实现时把 `agent.py` 的 `AGENT_ENGINE` 默认值从 `loop` 改为 `graph`，同时保留显式 `AGENT_ENGINE=loop` 的回退开关。
2. **Gmail P0 自建最小 MCP server**——只暴露 `send_email`；不采用 `@gongrzhe/server-gmail-autoauth-mcp` 作为主链路，因为该社区仓库已归档，且工具面过宽（读信、删信、标签等能力不适合靠白名单事后剔除）。
3. **`send` 节点 = DAG 节点 + 局部 ReAct**——主图仍由路由决定是否发送；节点内部先生成专用 `send_payload`，再把已解析、已允许的收件人和 payload 交给局部 `create_react_agent` 调用发送工具。
4. **preflight 必须确定性执行**——联系人解析、通道可用性、是否需要用户确认都在 backend 代码里判断，不交给 LLM 自行决定。模糊命中、多命中、群聊/飞书在 P0 中一律不发送，返回需要确认或暂不支持。
5. **send_guard 用 `ToolCallInterceptor` 实现**——挂在 `langchain-mcp-adapters` 的 `tool_interceptors` 上，负责白名单校验、审计日志和结构化错误；不再手写包一层 `BaseTool wrapper`。
6. **开启 `tool_name_prefix=true`**——MCP 工具进入 LangChain 后统一叫 `gmail_send_email`、`contacts_resolve`、`feishu_send_message_to_group`，避免多 server 工具重名，也避免 OpenAI function name 对 `.` 的兼容风险。
7. **stdio 传输**，本机子进程，不占端口。MCP 官方/adapter 文档也提醒 stdio 在 web server 场景要谨慎；本设计接受 P0 单机原型的取舍，跨机或多副本时再换 Streamable HTTP。
8. **FastAPI startup / shutdown 钩子**沿用现有 `@app.on_event`，不为 MCP 单独迁移 lifespan。

## 3.1 Coordinator 输出扩展

现有 Coordinator 输出的 `plan` dict 需要新增 3 个字段：

| 字段 | 类型 | 语义 |
|------|------|------|
| `send_intent` | bool | 用户话里是否有明确的"发给 / 通知 / 抄送 / 转给"意图 |
| `recipients` | list[str] | 从用户话里抽出的自然语言收件人，如"张三"、"运营组" |
| `channels` | list[str] | 用户明确指定的渠道：`["gmail"]` / `["feishu"]` / `["gmail","feishu"]`；未指定则为空 |

**路由**：`finalize → END` 变成 `finalize →(send_intent?)→ send | END`。

**实现注意**：不是只改 prompt。当前 Coordinator 使用 `with_structured_output(CoordinatorPlan)`，所以必须同步修改：

- `CoordinatorPlan` Pydantic schema；
- `_FALLBACK_PLAN`；
- `_plan_to_dict()` 的字段归一化；
- `test_graph_coordinator_node.py` / routing tests；
- README / `.env.example` 中对 `AGENT_ENGINE=graph` 的说明。

## 3.2 send 节点结构

```python
# app/agent/graph/nodes/send.py（伪代码）
from langgraph.prebuilt import create_react_agent

from app.mcp.client import mcp_manager
from app.mcp.payload import build_send_payload
from app.mcp.preflight import resolve_send_targets
from app.utils.factory import get_chat_model

async def send_node(state: AgentState) -> dict:
    plan = state.get("plan") or {}
    if not plan.get("send_intent"):
        return {}

    # 1) 确定性 preflight：不让 LLM 决定是否可以发
    preflight = await resolve_send_targets(
        recipients=plan.get("recipients") or [],
        channels=plan.get("channels") or [],
        identity=state.get("identity"),
    )
    if not preflight.can_send:
        return {
            "send_report": preflight.user_message,
            "trace": [{"agent": "send", "status": "skipped", "output": preflight.reason}],
        }

    payload = await build_send_payload(
        source_answer=state.get("final_answer") or "",
        original_query=state.get("query") or "",
        recipients=preflight.recipients,
    )

    # P0 只给局部 agent 暴露 Gmail 发送工具，不暴露 contacts/feishu。
    tools = mcp_manager.get_tools(names=["gmail_send_email"])
    agent = create_react_agent(model=get_chat_model("send"), tools=tools)

    task = (
        "使用下面的 send_payload 发送 Gmail。"
        "不要添加附件，不要改写收件人，不要重新生成正文，不要调用未提供的工具。\n"
        f"收件人邮箱：{preflight.gmail_to}\n"
        f"subject：{payload.subject}\n"
        f"body：{payload.body_text}"
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    report = result["messages"][-1].content
    return {
        "send_report": report,
        "trace": [{"agent": "send", "status": "done", "output": report[:200]}],
    }
```

`send` 节点是"**主图的一个 DAG 节点，节点内部是 ReAct**"。但 ReAct 只负责调用已允许的发送工具；是否能发、发给谁、P0 是否支持该通道，都由 preflight 和 interceptor 确定。

## 3.3 send_report 可见性与持久化

当前 GraphRunner 只会把 `finalize` 节点的 token 作为用户可见回答，并在结束时保存 `citations/steps`。因此 P0 必须同时修改 GraphRunner / session metadata：

1. GraphRunner 在 `values` stream 中收集 `send_report`；
2. graph 结束后，如果 `send_report` 非空，向前端追加一段 token：`\n\n发送结果：...`，确保用户当场看到发送状态；
3. `done` frame 带上 `send_report`；
4. `DatabaseSessionManager.add_message()` 的 assistant metadata 增加 `send_report`，`get_session()` 回放时也带回；
5. `send` 节点通过 custom stream 写 `agent_step_update`，前端步骤里显示"准备发送 / 已发送 / 未发送原因"。

这样发送结果不会只存在 LangGraph state 里，也不会在用户刷新会话后丢失。

## 4. 组件与文件清单

### 4.1 后端新增/修改

```text
backend/
├── app/
│   ├── mcp/                             ← 新
│   │   ├── __init__.py
│   │   ├── client.py                    ← MCPClientManager，全局单例
│   │   ├── config_loader.py             ← 读 mcp_servers.yaml
│   │   ├── preflight.py                 ← 确定性解析/确认/通道可用性判断
│   │   ├── payload.py                   ← final_answer → send_payload
│   │   └── send_guard.py                ← ToolCallInterceptor：白名单 + 审计 + typed error
│   ├── config/
│   │   └── mcp_servers.yaml             ← 新
│   ├── agent/
│   │   ├── agent.py                     ← 改：AGENT_ENGINE 默认 graph
│   │   ├── agent_tools.py               ← 不改（loop 引擎不注入 MCP 工具）
│   │   └── graph/
│   │       ├── build.py                 ← 改：路由加 send 节点
│   │       ├── runner.py                ← 改：收集/输出/持久化 send_report
│   │       ├── state.py                 ← 改：AgentState 加 send_payload / send_report
│   │       └── nodes/
│   │           ├── coordinator.py       ← 改：CoordinatorPlan 扩 send 字段
│   │           └── send.py              ← 新：preflight + create_react_agent + MCP tools
│   ├── models/
│   │   └── contact.py                   ← 新，Contact / FeishuGroup ORM（复用 chat_history.Base）
│   ├── router/
│   │   └── contacts.py                  ← 新，联系人 CRUD
│   ├── schemas/
│   │   └── contact.py                   ← 新，Pydantic
│   └── services/
│       ├── contact_service.py           ← 新，DB 层
│       └── database_session_manager.py  ← 改：metadata 增加 send_report
├── mcp_servers/                         ← 新目录
│   ├── gmail/
│   │   ├── __init__.py
│   │   ├── server.py                    ← FastMCP 入口，只暴露 send_email
│   │   ├── gmail_client.py              ← Gmail API 封装
│   │   ├── auth.py                      ← credentials/token 读取与 refresh
│   │   └── README.md                    ← OAuth 准备说明
│   ├── contacts/
│   │   ├── __init__.py
│   │   ├── server.py                    ← FastMCP 入口（读 MySQL）
│   │   └── README.md
│   └── feishu/                          ← P2 新增
│       └── ...
├── scripts/
│   ├── seed_contacts.py                 ← 手动录联系人
│   └── mcp_smoke_test.py                ← 本地手工验收
├── tests/
│   └── mcp/                             ← 新
│       ├── test_mcp_client_boot.py
│       ├── test_gmail_server.py         ← mock Gmail API
│       ├── test_contacts_server.py
│       ├── test_send_guard.py
│       ├── test_send_preflight.py
│       └── test_send_node.py
├── main.py                              ← 改：startup / shutdown 里 MCP 启停
├── pyproject.toml                       ← 改：加依赖
└── .env.example                         ← 改：AGENT_ENGINE / Gmail / MCP env
```

### 4.2 数据库（现有 MySQL 库内新增两张表）

**表 `contact` —— 存"人"（P0 邮件；P2 飞书私聊）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | PK | |
| name | varchar(64) | 显示名，"张三" |
| alias | varchar(255) | 逗号分隔，"老张,zhangsan"；P0 简化，P2 可拆表 |
| email | varchar(255) | Gmail 用；校验时 trim + lower |
| feishu_open_id | varchar(64) | P2 飞书私聊主键 |
| feishu_user_id | varchar(64) | P2 备用 |
| is_active | bool | 软禁用，默认 true |
| note | varchar(255) | |
| created_at / updated_at | | |
| **UNIQUE(name)** | | |
| **INDEX(email)** | | 白名单校验用 |

**表 `feishu_group` —— 存"飞书群"（P2）**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | PK | |
| name | varchar(64) | 显示名，"运营组" |
| chat_id | varchar(64) | `oc_xxx` |
| is_active | bool | 软禁用，默认 true |
| note | varchar(255) | |
| **UNIQUE(name)**、**UNIQUE(chat_id)** | | |

注意：当前项目的 SQLAlchemy `Base` 定义在 `app/models/chat_history.py`。新增 `models/contact.py` 必须复用同一个 `Base`，并确保 `init_db()` 调用 `Base.metadata.create_all` 前已 import 新模型，否则表不会创建。

### 4.3 依赖

后端新增：

- `mcp>=1.12.4` —— 官方 Python SDK（gmail / contacts / feishu server 用）；
- `langchain-mcp-adapters>=0.1.0` —— MCP tool → LangChain `BaseTool` 适配层；
- `google-api-python-client`、`google-auth`、`google-auth-oauthlib` —— 自写 Gmail server 使用 Gmail API。

不再通过 `npx -y @gongrzhe/server-gmail-autoauth-mcp` 拉社区 Gmail server；该方案只保留为 rejected alternative，不进入 P0。

### 4.4 配置

**`.env.example` 追加 / 修改**：

```env
# ── Agent 引擎 ─────────────────────────
AGENT_ENGINE=graph

# ── MCP 总开关 ────────────────────────
MCP_ENABLED=true

# ── Gmail MCP（P0）────────────────────
GMAIL_CREDENTIALS_PATH=./secrets/gmail_credentials.json
GMAIL_TOKEN_PATH=./secrets/gmail_token.json
GMAIL_SENDER_EMAIL=your-sandbox@gmail.com

# ── 飞书自建应用（P2）─────────────────
LARK_APP_ID=cli_xxxxxxxxxxxx
LARK_APP_SECRET=xxxxxxxxxxxxxxxxxxxx
LARK_DOMAIN=https://open.feishu.cn
```

**`mcp_servers.yaml` 示例**：

```yaml
tool_name_prefix: true
servers:
  gmail:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.gmail.server"]
    include_tools: ["send_email"]      # 对 LLM 暴露为 gmail_send_email
    env:
      GMAIL_CREDENTIALS_PATH: "${GMAIL_CREDENTIALS_PATH}"
      GMAIL_TOKEN_PATH: "${GMAIL_TOKEN_PATH}"
      GMAIL_SENDER_EMAIL: "${GMAIL_SENDER_EMAIL}"
  contacts:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.contacts.server"]
  feishu:
    enabled: false                      # P2 启用
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.feishu.server"]
    env:
      LARK_APP_ID: "${LARK_APP_ID}"
      LARK_APP_SECRET: "${LARK_APP_SECRET}"
      LARK_DOMAIN: "${LARK_DOMAIN}"
```

## 5. MCP 工具接口

> 约定：MCP server 内部工具仍叫 `send_email` / `resolve`；进入 LangChain 后因 `tool_name_prefix=true` 暴露为 `gmail_send_email` / `contacts_resolve`。

### 5.1 contacts server（自写）

**`contacts_resolve`**

- 描述：把用户口中的"谁"（姓名 / 别名 / 群名）解析成结构化送达地址。
- 入参：`query: str`
- 出参：

  ```json
  {
    "matches": [
      {
        "kind": "person",
        "name": "张三",
        "email": "zhangsan@example.com",
        "feishu_open_id": "ou_abc",
        "match_type": "exact"
      }
    ],
    "ambiguous": false
  }
  ```

- 匹配规则：`name` 精确 → `alias` 精确 → `name` 模糊 LIKE。
- P0 自动发送条件：必须是单个 `person`，且 `match_type in {"exact", "alias"}`，且有 email。
- 多命中、模糊命中、无 email、group 命中：`send` 节点返回"需要确认/暂不支持"，不发送。

**`contacts_list_all`**

- 描述：列出所有联系人和飞书群。仅调试用。
- P0 默认不暴露给 `send` 节点的局部 agent，只在 smoke/debug 脚本里使用。

### 5.2 gmail server（自写，P0）

**`gmail_send_email`**

- 入参：`to: list[str]`, `subject: str`, `body: str`, `cc?: list[str]`, `bcc?: list[str]`
- 出参：`{"message_id": "..."}`
- P0 不支持 `attachments` 参数；附件能力等 artifact 生成/文件安全校验落地后再进 P1。

`send_guard` interceptor 在工具真正调用 Gmail API 之前校验：

- `to / cc / bcc` 每个邮箱都必须存在于 `contact.email` 白名单且 `is_active=true`；
- 邮箱比较统一 `strip().lower()`；
- 不通过 → 返回结构化 `RECIPIENT_NOT_ALLOWED`，不调用 Gmail API；
- 通过且 Gmail API 成功 → 写 `log/sends.log` 审计。

### 5.3 feishu server（P2）

| 工具 | 入参 | 出参 |
|------|------|------|
| `feishu_send_message_to_user` | `open_id`, `content`, `msg_type?=text\|post` | `{"message_id": "om_..."}` |
| `feishu_send_message_to_group` | `chat_id`, `content`, `msg_type?=text\|post` | `{"message_id": "om_..."}` |
| `feishu_send_file_to_user` | `open_id`, `file_path`, `file_name?` | `{"message_id": "om_...", "file_key": "..."}` |
| `feishu_send_file_to_group` | `chat_id`, `file_path`, `file_name?` | `{"message_id": "om_...", "file_key": "..."}` |

`send_file_to_*` 是原子操作：内部先上传拿 `file_key`，再发消息；不单独暴露 upload。

### 5.4 P0 典型调用链

> 用户："把这份报告发给张三。"

```text
[主图]
1. Coordinator: send_intent=true, recipients=["张三"], channels=[]
2. Knowledge / Task / Finalize: 生成 final_answer 文本
3. route: finalize → send

[send 节点]
4. preflight: contact_service.resolve("张三") → 单个 person + exact/alias + email
5. preflight: P0 默认 channel=gmail，允许发送
6. 局部 create_react_agent 只看到 gmail_send_email
7. build_send_payload(final_answer) → {subject, body_text, attachments=[]}
8. tool_call: gmail_send_email(to=[email], subject=payload.subject, body=payload.body_text)
9. send_guard interceptor 白名单校验 + 成功审计
10. send_report: "已通过 Gmail 发送给张三。"
11. GraphRunner 把 send_report 追加给前端并持久化
```

> 用户："把这份报告发给张三和运营组。"

P0 不做部分发送。因为包含 group/飞书通道，preflight 返回"飞书群发送暂未启用，请去掉群聊或等 P2"，不调用 Gmail，避免用户误以为全部完成。

## 6. 数据流 & 关键决定

1. **发送触发由主图路由决定**：Coordinator 判 `send_intent`，路由 `finalize →(send_intent?)→ send | END`。LLM 不决定是否进入发送节点。
2. **发送许可由 preflight 决定**：收件人解析、通道可用性、是否需要确认都在 backend 确定性执行。LLM 不决定是否允许发送。
3. **P0 无附件，但有 `send_payload`**：`report_tool` / `form_tool` 当前只是 prompt builder，真正输出在 `finalize_node`。因此 P0 不发附件，也不裸发 `final_answer`；必须先生成专用 `send_payload`，再发送 subject/body。artifact 生成、文件落盘、文件大小/路径安全校验、附件发送统一放 P1。
4. **工具白名单 + 工具名前缀**：MCPClientManager 开启 `tool_name_prefix=true`，再应用 `include_tools`；局部 send agent P0 只拿到 `gmail_send_email`。
5. **审计日志**：`log/sends.log`（JSONL），每次成功发送写一行 `{ts, user_id, session_id, channel, recipient, subject_or_preview, message_id}`。写入点是 `send_guard` interceptor 的成功分支。
6. **收件人白名单护栏**：LLM 只能发到 `contact.email` / P2 `contact.feishu_open_id` / P2 `feishu_group.chat_id` 里已录入且 active 的地址。白名单在 interceptor 层强制校验。
7. **发送结果可见且可回放**：`send_report` 既追加到当前 SSE 输出，也进入 assistant message metadata。
8. **幂等 / 重试**：P0 不做完整客户端级去重表；但 `send_node` 必须保证一次请求最多一次真实 `gmail_send_email` 调用，且成功后写审计。Gmail API 失败不自动重试，由 send_report 明确告知失败。P1/P2 如要支持重试，需先引入 `send_request_id` 和去重表。
9. **send 节点串行 tool 调用**：P0 只有 Gmail 单工具；P2 多通道也保持串行，错误恢复更清楚。

## 7. 错误处理

| 错误层 | 触发 | 处理位置 | 用户/LLM 看到 |
|--------|------|----------|---------------|
| 配置 | enabled server 必填 env 缺失 | startup config validation | FastAPI 拒绝启动（fail-fast） |
| 传输 | stdio 子进程崩 / 超时 | MCPClientManager | 该 server 工具下线；send 节点报告发送能力不可用 |
| 认证 | Gmail token 过期且 refresh 失败 | gmail server typed error | `{"error":"AUTH_FAILED","channel":"gmail"}` |
| preflight | 收件人无命中/多命中/模糊命中/群聊 P0 | send preflight | 不发送，返回需要确认或暂不支持 |
| 白名单 | 工具参数里出现未录入邮箱 | send_guard interceptor | `{"error":"RECIPIENT_NOT_ALLOWED"}` |
| 业务 | Gmail API 错误 | gmail server | `{"error":"GMAIL_API_ERROR","code":...}` |
| 参数 | LLM 拼错参数 | Pydantic / tool schema | 结构化错误回喂给局部 agent；最终 send_report 说明失败 |

原则：**配置错误 fail-fast；运行期能力缺失 fail-closed for send、fail-open for 主问答**。也就是说 MCP 起不来不影响普通问答，但只要发送链路不确定，就不发。

### 7.1 飞书 tenant_access_token 缓存（P2）

- 有效期 2h；feishu server 进程内维护缓存 + 到期前 5min 主动刷新。
- 遇 "invalid access_token" 错误码，清 cache 并内部重试一次，仍失败才抛。
- 不用 Redis —— 单进程、无跨机共享需求；未来横向扩再切换（TokenCache 抽象层预留）。

## 8. 测试策略

**单元测试**（pytest，`backend/tests/mcp/`）：

- `test_gmail_server.py` —— mock Gmail API，覆盖 MIME 组装、OAuth token refresh、API 错误转 typed error。
- `test_contacts_server.py` —— 测试 DB，覆盖精确 / 别名 / 模糊 / ambiguous / not found / inactive。
- `test_mcp_client_boot.py` —— 启动能拿 prefixed 工具列表 / include_tools 生效 / 关闭清理 / env 缺失报错。
- `test_send_guard.py` —— 白名单放行 / 拦截 / 大小写与 trim / 成功审计。
- `test_send_preflight.py` —— exact 可发、fuzzy 需确认、group P0 不发、混合 unsupported 不做部分发送。
- `test_send_payload.py` —— 验证 final_answer 到 subject/body 的转换，不裸发原始 final_answer。
- `test_send_node.py` —— mock preflight + mock payload + mock tool，验证 send_report 与 trace。
- `test_graph_runner_send_report.py` —— 验证 send_report 进入 SSE token、done frame、session metadata。
- `test_graph_coordinator_node.py` / routing tests —— 覆盖 `send_intent / recipients / channels` schema。

**手工 smoke**（`scripts/mcp_smoke_test.py`，本地跑不进 CI）：

- 依赖真 Gmail 沙箱账号；
- 依赖 `seed_contacts.py` 先录入自己的 email；
- 走完 `contacts_resolve → gmail_send_email` 文本发送链路；
- 验证邮件正文到达、`sends.log` 有记录。

**Agent 端到端**（P1 前补）：小型 eval 集 5~10 条发送类话术，检查 preflight 决策和 tool 调用签名，不真发。

## 9. 启停与灰度

**FastAPI `@app.on_event("startup")`**（沿用现有钩子，不专门改成 lifespan）：

```text
1. 读 .env（现有）
2. 初始化 DB / Redis（现有）
3. 读 config/mcp_servers.yaml → 展开 env 占位
4. 校验 enabled server 必填 env；缺失则 fail-fast
5. MCPClientManager.start()
     ├─ 逐个 server 起子进程（并行）
     ├─ list_tools() 拿清单
     ├─ 开启 tool_name_prefix
     ├─ 应用 include_tools 白名单
     └─ 注册 send_guard ToolCallInterceptor
6. graph 编译完成；send 节点运行时通过 mcp_manager.get_tools() 取工具
7. 服务对外可用
```

**`@app.on_event("shutdown")`** 相反顺序，`terminate` + 5s 超时后 `kill`。

**引擎影响**：

- `AGENT_ENGINE=graph`（P0 默认）：`send` 节点生效，发送能力可用。
- `AGENT_ENGINE=loop`：不注入 MCP 工具，发送能力禁用；保留为回退开关，不作为 P0 验收路径。

**失败策略**：

- 某个 enabled server 必填 env 缺失 → 拒绝启动。
- 某个 server env 完整但子进程起不来 → 记 log，从工具列表剔除，普通问答继续。
- Gmail 工具不可用且用户请求发送 → send 节点返回"发送能力当前不可用"，不影响 final_answer。
- 全部 MCP server 起不来 → 普通问答继续，所有发送请求 fail-closed。

**灰度 & 回滚**：

- 总开关 `MCP_ENABLED=true|false`；false 时 `MCPClientManager` 空跑，`send` 节点短路返回"发送能力未启用"；
- 每 server 单独 `enabled: true|false`；
- `AGENT_ENGINE=loop` 可作为紧急回退，但不会具备发送能力；
- 回滚可用 `MCP_ENABLED=false`，无需回滚普通 RAG 代码。

## 10. 分期与验收

**P0 —— Graph 发送骨架 + 自写 Gmail 正文发送**

1. 依赖：`mcp`, `langchain-mcp-adapters`, Gmail API 相关库。
2. 建 `contact` / `feishu_group` 表 + service + CRUD router（feishu_group 先为 P2 数据准备）。
3. 自写 `mcp_servers/gmail/`，只暴露 `send_email`。
4. 自写 `mcp_servers/contacts/`，复用 `contact_service`。
5. `MCPClientManager`：stdio、启停、`tool_name_prefix=true`、include_tools、interceptors。
6. `send_guard` interceptor：白名单、typed error、成功审计。
7. `send_preflight`：确定性联系人解析、通道判断、禁止 P0 部分发送。
8. Coordinator schema/prompt/fallback 扩 `send_intent / recipients / channels`。
9. `build.py` 加 `send` 节点和 `finalize → send | END` 路由。
10. `nodes/send.py`：preflight + `send_payload` + 局部 `create_react_agent` + `gmail_send_email`。
11. `GraphRunner` / `DatabaseSessionManager` 支持 `send_report` 可见与回放。
12. `agent.py` 默认 `AGENT_ENGINE=graph`，`.env.example` 明写。
13. 单测：client boot、gmail server、contacts、send_guard、preflight、send_payload、send_node、GraphRunner send_report、Coordinator routing。
14. 手工 smoke：真实 Gmail 沙箱账号文本发送。

**P0 验收**：`AGENT_ENGINE=graph` 下，对话说"把刚才那份报告发给 <已录入联系人>"，Agent 走完 graph 后经 send 节点生成 `send_payload`，再用 `gmail_send_email` 发送 payload 的 subject/body；邮件到达且正文不是未清洗的原始 `final_answer`；前端看到"发送结果"；刷新会话仍能看到发送结果；`sends.log` 有一条记录。

**P1 —— Artifact 与附件发送**

1. 新增 artifact renderer：把 `final_answer` 落成 Markdown/PDF/Docx（先选一种）。
2. `AgentState` 加 `artifact_path` / `artifact_mime` / `artifact_name`。
3. 文件安全校验：路径必须在受控 artifact 目录，限制大小、扩展名、存在性。
4. Gmail `send_email` 扩 `attachments?: list[str]`。
5. `report_tool` / `form_tool` 不直接"返回 artifact_path"，而是由 finalize 后的 artifact renderer 生成文件。
6. 单测和 smoke 覆盖附件到达。

**P2 —— 飞书通道**

1. `mcp_servers/feishu/`（server / lark_client / token_cache）。
2. `test_feishu_server.py`（mock REST）。
3. `send_guard` 扩到 feishu（open_id / chat_id 校验）。
4. preflight 支持 group / explicit feishu channel。
5. 手工 smoke：真 sandbox 应用 send_to_user / send_to_group / send_file_to_group。

**P2 验收**：对话说"发给张三 + 发到运营群"，在所有收件人都明确且通道可用时，邮件/飞书均送达；任一收件人不明确时不做部分发送。

**P3 —— 打磨 & loop 收敛（各自独立 spec）**

1. Agent eval 集加发送类话术。
2. contacts 后台管理 UI。
3. 更细的错误码文案。
4. `send_audit` 落库替代 JSONL（如需要检索）。
5. loop 引擎移除（另起 `docs/specs/2026-07-XX-drop-loop-engine.md`）。

## 11. 用户侧准备（非代码）

上线前需要用户自行准备，本设计不代做：

1. **Gmail P0**：Google Cloud 项目 → 启用 Gmail API → 建 OAuth 客户端 → 下 `credentials.json` → 首次本地授权生成 `token.json` → 放 `secrets/`；建议使用沙箱 Gmail 账号。
2. **联系人 seed**：录入自己 + 一个测试同事的 email 到 `contact` 表；P2 前再补 open_id / chat_id。
3. **飞书 P2**：开放平台 → 创建自建应用 → 开启 `im:message` 和 `im:resource` 权限 → 拿 `app_id` / `app_secret` → bot 加入测试群拿 `chat_id`。

## 12. Open Questions（可延后）

- 联系人管理页 UI 是走 Django admin 还是前端加页面？（P3 决定）
- `sends.log` 是否需要落库（`send_audit` 表）方便检索？（P3 视需要）
- 未来 per-user OAuth 接入时，系统身份、用户身份和 token cache 如何分层？
- P1 artifact 先落 Markdown、PDF 还是 Docx？建议先 Markdown/PDF 二选一，不在 P0 内混入。
- loop 引擎移除的时间点：等 graph 引擎作为默认稳定运行 2 周以上，再另起 `2026-07-XX-drop-loop-engine.md` spec。

## 13. 变更记录

- **2026-07-03 初版**：双引擎都注入 MCP 工具。
- **2026-07-03 修订**：
  - 明确 ReAct（编排范式）与 function calling（通信协议）正交（§2.2）；
  - 澄清 loop = 手写 ReAct、graph = DAG（§2.3）；
  - P0 收敛到只在 graph 引擎里加 `send` 节点承载 MCP 工具，内部用 `create_react_agent`（§3、§3.1、§3.2）；
  - Coordinator 输出扩 `send_intent / recipients / channels`（§3.1）；
  - 数据流改成主图路由到 send 节点（§5.4、§6）；
  - loop 收敛作为后续独立 spec 处理（§10、§12）。
- **2026-07-03 二次修订**：
  - Gmail P0 改为自写最小 Python MCP server，不再依赖 archived 社区 Node 包；
  - P0 收敛为 Gmail 正文发送，artifact/附件进入 P1，飞书进入 P2；
  - 增加 deterministic preflight，禁止模糊/多命中/unsupported recipient 自动发送；
  - `send_guard wrapper` 改为 `ToolCallInterceptor`，并要求 `tool_name_prefix=true`；
  - 明确 `send_report` 必须进入 SSE、done frame 和 session metadata；
  - 明确 P0 实现时 `AGENT_ENGINE` 默认切到 `graph`。
- **2026-07-03 三次修订**：
  - 补充 per-user OAuth 与客户端级幂等/去重的含义和 P0 风险边界；
  - 明确 P0 不裸发 `final_answer`，而是由 `send` 节点生成 `send_payload` 后发送；
  - 增加 `payload.py` / `test_send_payload.py`，并把 P0 验收改为校验 subject/body payload。
