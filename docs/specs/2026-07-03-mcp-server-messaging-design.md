# MCP 消息通道接入设计（Gmail + 飞书）

> 状态：设计稿 · 待实现
> 作者：黄职勇（与 Claude 结对 brainstorm）
> 日期：2026-07-03

## 1. 背景与目标

现有 RAG Agent 只完成"检索 + 生成"。用户希望在生成产出物（对比、报告、申请）之后，能在同一段对话里让 Agent **自动发送**这些产出物给指定的人或飞书群。

典型话术：

> "把刚才那份薪酬报告发给张三，同时发到运营群。"

本设计要解决的是**接入邮件与飞书两条通道**这件事，选用 **MCP（Model Context Protocol）** 作为工具接入形态，原因：

- 复用社区已有 Gmail MCP server，零业务代码；
- 飞书自建 MCP server，能力可控、可复用给其他 Agent（Claude Desktop、Cursor 等）；
- 联系人解析（`contacts`）也做成 MCP server，被 gmail、feishu 两条通道共享。

**非目标**：

- 不做 per-user OAuth（原型阶段全项目共用一套系统身份）；
- 不做客户端级幂等 / 去重；
- 不接入短信、微信、Slack、Teams（YAGNI）；
- 不做"读邮箱""删邮件"等敏感能力，即使社区 gmail server 提供也白名单剔除。

## 2. 术语与选型澄清

### 2.1 MCP 相关

- **MCP Server**：对外提供工具的一方。本设计里的 gmail（社区包）、feishu（自写）、contacts（自写）都是 MCP server。
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
- **本 P0 的取舍**：**只在 graph 引擎里承载发送能力**（见 § 3）。loop 引擎不注入 MCP 工具，未来另起 spec 移除 loop（避免双引擎双向适配的额外成本）。

## 3. 总体架构

**P0 只走 graph 引擎**——在图上加一个 `send` 节点作为"发送能力承载点"。`send` 节点内部用 LangGraph 官方 **`create_react_agent`** 承载全部 MCP 工具，跑一个局部 ReAct 循环把消息发出去；跟主图的 DAG 编排正交组合。

```
┌────────────────────────────────────────────────────────────────┐
│ FastAPI backend (existing, port 8000)                          │
│                                                                │
│   AGENT_ENGINE=graph（P0 只支持此引擎发送）                     │
│                                                                │
│   Coordinator → Knowledge → [Task | Gap] → Finalize            │
│                                    │                           │
│                                    ▼                           │
│                          ┌───────────────────────┐             │
│                          │  send 节点（新增）      │             │
│                          │                       │             │
│                          │  create_react_agent(  │             │
│                          │    llm,               │             │
│                          │    tools = MCP tools) │             │
│                          │                       │             │
│                          │  局部 ReAct 循环：      │             │
│                          │   resolve → send      │             │
│                          └───────────────────────┘             │
│                                    │                           │
│                                    ▼                           │
│                                  END                           │
│                                                                │
│                            ▲                                   │
│                            │ 通过 send_guard wrapper 校验       │
│                            │ 收件人是否在 contacts 表白名单内     │
│   ┌──────────────────────────────────────┐                     │
│   │ MCPClientManager  (app/mcp/client.py) │                     │
│   │  - startup 读 mcp_servers.yaml         │                     │
│   │  - MultiServerMCPClient 起子进程       │                     │
│   │  - 应用 include_tools 白名单           │                     │
│   │  - 挂 send_guard wrapper              │                     │
│   │  - shutdown 清理子进程                │                     │
│   └──────────────────────────────────────┘                     │
│                                                                │
│   AGENT_ENGINE=loop：不注入 MCP 工具，发送能力自动禁用            │
│                     （loop 收敛移除在后续 P2 spec 中做）          │
└────────────────────────────────────────────────────────────────┘
         │ stdio            │ stdio            │ stdio
         ▼                  ▼                  ▼
  ┌──────────────┐   ┌────────────────┐   ┌────────────────┐
  │ gmail MCP    │   │ feishu MCP     │   │ contacts MCP   │
  │ (社区 Node 包)│   │ (自写 Python)   │   │ (自写 Python)   │
  └──────────────┘   └────────────────┘   └────────────────┘
   Google OAuth        app_id/app_secret   MySQL (contact 表)
```

**关键决定**：

1. **P0 只支持 graph 引擎**——避免为 loop 侧多写一层 OpenAI schema ↔ BaseTool 双向适配。
2. **`send` 节点内部用 `create_react_agent`**——发送场景是典型 ReAct（LLM 决定发几个通道、按什么顺序发），比手写 loop 稳，比让主图 DAG 硬编码分支灵活。
3. **contacts 独立 MCP server**，不塞进 feishu——邮件也要用它，抽出来后两条通道共享。
4. **stdio 传输**，本机子进程，不占端口。跨机再换 SSE/HTTP，接口不变。
5. **`send_guard` wrapper** 在 backend 进程里挂到 send 类 BaseTool 上，做收件人白名单校验；不放到 MCP server 内，避免每个 server 都要连 DB。
6. **FastAPI startup / shutdown 钩子**（沿用现有 `@app.on_event`，不为 MCP 特意改成 lifespan）里拉起 / 关闭 MCP 客户端子进程。

## 3.1 Coordinator 输出扩展

现有 Coordinator 输出的 `plan` dict 需要新增 3 个字段：

| 字段            | 类型            | 语义                                                     |
|-----------------|-----------------|---------------------------------------------------------|
| `send_intent`   | bool            | 用户话里是否有"发给 / 通知 / 抄送 / 转给"这类意图              |
| `recipients`    | list[str]       | 从用户话里抽出的自然语言收件人，"张三"、"运营组"、"李四"          |
| `channels`      | list[str]       | `["gmail"]` / `["feishu"]` / `["gmail","feishu"]`；缺省为空 |

**路由**：`finalize → END` 变成 `finalize →(send_intent?)→ send | END`。

**Coordinator prompt 需要扩**（不新写节点，沿用现有 coordinator）：加一段"若用户明确要求发送/通知/转给某人某群，抽取意图和对象；否则字段留空/false"。

## 3.2 send 节点结构

```python
# app/agent/graph/nodes/send.py（伪代码）
from langgraph.prebuilt import create_react_agent
from app.mcp.client import mcp_manager

async def send_node(state: AgentState) -> dict:
    plan = state.get("plan") or {}
    if not plan.get("send_intent"):
        return {}  # 兜底，理论上路由已经过滤
    tools = mcp_manager.get_tools()  # 已挂 send_guard wrapper
    agent = create_react_agent(model=get_chat_model("send"), tools=tools)
    task = (
        f"用户希望把以下内容发送出去。收件人：{plan['recipients']}，"
        f"渠道偏好：{plan.get('channels') or '自动判断'}。\n"
        f"要发送的内容：{state['final_answer']}\n"
        f"附件路径（可选）：{state.get('artifact_path')}\n"
        f"步骤：先用 contacts.resolve 解析每个收件人；"
        f"若是 person 走 gmail 或 feishu；若是 group 走 feishu。"
        f"每个通道各调一次即可，不要重复发送。"
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    return {"send_report": result["messages"][-1].content}
```

`send` 节点是"**主图的一个 DAG 节点，节点内部是 ReAct**"，跟主图正交。

## 4. 组件与文件清单

### 4.1 后端新增/修改

```
backend/
├── app/
│   ├── mcp/                             ← 新
│   │   ├── __init__.py
│   │   ├── client.py                    ← MCPClientManager，全局单例
│   │   ├── config_loader.py             ← 读 mcp_servers.yaml
│   │   └── send_guard.py                ← 收件人白名单 wrapper
│   ├── config/
│   │   └── mcp_servers.yaml             ← 新
│   ├── agent/
│   │   ├── agent_tools.py               ← 不改（loop 引擎不注入 MCP 工具）
│   │   └── graph/
│   │       ├── build.py                 ← 改：路由加 send 节点
│   │       ├── state.py                 ← 改：AgentState 加 send_report / artifact_path
│   │       └── nodes/
│   │           ├── coordinator.py       ← 改：plan 输出扩 send_intent / recipients / channels
│   │           └── send.py              ← 新：create_react_agent + MCP tools
│   ├── models/
│   │   └── contact.py                   ← 新，Contact / FeishuGroup ORM
│   ├── router/
│   │   └── contacts.py                  ← 新，联系人 CRUD
│   ├── schemas/
│   │   └── contact.py                   ← 新，Pydantic
│   ├── services/
│   │   └── contact_service.py           ← 新，DB 层
│   └── tools/                           ← 改：report_tool / form_tool
│                                          输出加 artifact_path
├── mcp_servers/                         ← 新目录
│   ├── feishu/
│   │   ├── __init__.py
│   │   ├── server.py                    ← FastMCP 入口
│   │   ├── lark_client.py               ← 飞书 REST 封装
│   │   ├── token_cache.py               ← 进程内 tenant_access_token 缓存
│   │   └── README.md                    ← 应用创建、权限、chat_id 获取
│   └── contacts/
│       ├── __init__.py
│       ├── server.py                    ← FastMCP 入口（读 MySQL）
│       └── README.md
├── scripts/
│   ├── seed_contacts.py                 ← 手动录联系人
│   └── mcp_smoke_test.py                ← 本地手工验收
├── tests/
│   └── mcp/                             ← 新
│       ├── test_mcp_client_boot.py
│       ├── test_feishu_server.py        ← mock 飞书 REST
│       ├── test_contacts_server.py
│       └── test_send_guard.py
├── main.py                              ← 改：startup / shutdown 里 MCP 启停
├── pyproject.toml                       ← 改：加依赖
└── .env.example                         ← 改：加 Gmail / 飞书 env
```

### 4.2 数据库（现有 MySQL 库内新增两张表）

**表 `contact` —— 存"人"（可发邮件、可发飞书私聊的对象）**

| 字段             | 类型          | 说明                                   |
|-----------------|--------------|---------------------------------------|
| id              | PK           |                                       |
| name            | varchar(64)  | 显示名，"张三"                          |
| alias           | varchar(255) | 逗号分隔，"老张,zhangsan"               |
| email           | varchar(255) | Gmail 用                              |
| feishu_open_id  | varchar(64)  | 飞书私聊主键                            |
| feishu_user_id  | varchar(64)  | 备用                                   |
| note            | varchar(255) |                                       |
| created_at / updated_at |       |                                       |
| **UNIQUE(name)** |              |                                       |

**表 `feishu_group` —— 存"飞书群"（飞书特有的收件对象，邮件通道用不上）**

| 字段        | 类型         | 说明                       |
|-------------|-------------|----------------------------|
| id          | PK          |                            |
| name        | varchar(64) | 显示名，"运营组"             |
| chat_id     | varchar(64) | `oc_xxx`                   |
| note        | varchar(255)|                            |
| **UNIQUE(name)**、**UNIQUE(chat_id)** |     |                            |

### 4.3 依赖

后端新增：

- `mcp>=1.0.0` —— 官方 Python SDK（飞书 / contacts server 用 `FastMCP`）
- `langchain-mcp-adapters>=0.1.0` —— MCP tool → LangChain `BaseTool` 适配层

Gmail 侧用社区 **Node.js** MCP server（如 `@gongrzhe/server-gmail-autoauth-mcp`），通过 `npx` 拉起，**不给 Python 项目加 npm 依赖**——config 里声明 command 就行。

### 4.4 配置

**`.env.example` 追加**：

```
# ── MCP 总开关 ────────────────────────
MCP_ENABLED=true

# ── Gmail MCP ─────────────────────────
GMAIL_CREDENTIALS_PATH=./secrets/gmail_credentials.json
GMAIL_TOKEN_PATH=./secrets/gmail_token.json

# ── 飞书自建应用 ──────────────────────
LARK_APP_ID=cli_xxxxxxxxxxxx
LARK_APP_SECRET=xxxxxxxxxxxxxxxxxxxx
LARK_DOMAIN=https://open.feishu.cn        # 海外：https://open.larksuite.com
```

**`mcp_servers.yaml` 示例**：

```yaml
servers:
  gmail:
    enabled: true
    transport: stdio
    command: npx
    args: ["-y", "@gongrzhe/server-gmail-autoauth-mcp"]
    include_tools: ["send_email"]         # 白名单（若适配器版本不支持此字段，在 MCPClientManager 内部做等价 filter）
    env:
      GMAIL_CREDENTIALS_PATH: "${GMAIL_CREDENTIALS_PATH}"
      GMAIL_TOKEN_PATH: "${GMAIL_TOKEN_PATH}"
  feishu:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.feishu.server"]
    env:
      LARK_APP_ID: "${LARK_APP_ID}"
      LARK_APP_SECRET: "${LARK_APP_SECRET}"
      LARK_DOMAIN: "${LARK_DOMAIN}"
  contacts:
    enabled: true
    transport: stdio
    command: python
    args: ["-m", "mcp_servers.contacts.server"]
```

## 5. MCP 工具接口

### 5.1 contacts server（自写）

**`contacts.resolve`**

- 描述：把用户口中的"谁"（姓名 / 别名 / 群名）解析成结构化的送达地址。命中多个时 `ambiguous=true`，让 LLM 追问。
- 入参：`query: str`
- 出参：

  ```json
  {
    "matches": [
      {"kind": "person", "name": "张三", "email": "zhangsan@x.com", "feishu_open_id": "ou_abc"},
      {"kind": "group",  "name": "运营组", "feishu_chat_id": "oc_xxx"}
    ],
    "ambiguous": false
  }
  ```

- 匹配规则：`name` 精确 → `alias` 命中 → `name` 模糊 LIKE；多命中 `ambiguous=true`。

**`contacts.list_all`**

- 描述：列出所有联系人和飞书群。兜底 / 调试用。
- 无参
- 出参：`{"contacts": [...], "groups": [...]}`

### 5.2 gmail server（社区包，白名单过滤后可见）

**`gmail.send_email`**

- 入参：`to: list[str]`, `subject: str`, `body: str`, `cc?: list[str]`, `bcc?: list[str]`, `attachments?: list[str]`
- 出参：`{"message_id": "..."}`

其他 search / draft / label / trash 等工具**通过 `include_tools` 白名单过滤掉**。

### 5.3 feishu server（自写）

| 工具                             | 入参                                              | 出参                                            |
|----------------------------------|--------------------------------------------------|-------------------------------------------------|
| `feishu.send_message_to_user`   | `open_id`, `content`, `msg_type?=text\|post`     | `{"message_id": "om_..."}`                      |
| `feishu.send_message_to_group`  | `chat_id`, `content`, `msg_type?=text\|post`     | `{"message_id": "om_..."}`                      |
| `feishu.send_file_to_user`      | `open_id`, `file_path`, `file_name?`             | `{"message_id": "om_...", "file_key": "..."}`   |
| `feishu.send_file_to_group`     | `chat_id`, `file_path`, `file_name?`             | `{"message_id": "om_...", "file_key": "..."}`   |

**`send_file_to_*` 是原子操作**——内部先 `im/v1/files` 上传拿 `file_key`，再 `im/v1/messages` 发送。**不单独暴露 upload**，避免 LLM 在两步之间自己保管 `file_key` 出错。

### 5.4 典型调用链

> 用户："把这份报告发给张三和运营组。"

主图 DAG 走到 `finalize` 之后进入 `send` 节点；send 节点内 `create_react_agent` 跑局部 ReAct 循环：

```
[send 节点内部]
1. LLM 思考："先解析收件人。"
2. LLM tool_call: contacts.resolve("张三")     → {kind: "person", email, open_id}
3. LLM tool_call: contacts.resolve("运营组")   → {kind: "group",  chat_id}
4. LLM 思考："张三走邮件（有 email 且 artifact 是 PDF），运营组走飞书群文件。"
5. LLM tool_call: gmail.send_email(to=[email], subject=..., attachments=[artifact_path])
6. LLM tool_call: feishu.send_file_to_group(chat_id, file_path=artifact_path)
7. LLM 输出最终消息："已通过邮件发送给张三，并通过飞书发送到运营组。"
8. send_node 把这段作为 state.send_report 返回；主图路由到 END。
```

`send_guard` wrapper 在 tool 调用真正打到 MCP server **之前**校验收件人 / open_id / chat_id 是否在白名单；不通过则 tool 抛 `RecipientNotAllowedError`，`create_react_agent` 把异常回喂给 LLM，LLM 转告用户"请先在联系人里录入 xxx"。

## 6. 数据流 & 关键决定

1. **发送触发是主图路由决定的，不是 LLM 决定的**：Coordinator 判 `send_intent`，路由 `finalize →(send_intent?)→ send | END`。**send 节点内部**才由 LLM（`create_react_agent`）自主决定用哪些通道、顺序。
2. **附件路径**：`report_tool` / `form_tool` 等工具的产出结构加 `artifact_path: str`（本地绝对路径）；`AgentState` 也加 `artifact_path` 字段透传给 send 节点。—— **本项目原有 tool 需要配合的一处小改。**
3. **幂等 / 重试**：不做客户端级去重；MCP 工具抛异常由 `create_react_agent` 默认行为处理（异常回喂 LLM 自决）。不做自动重试。
4. **审计日志**：`log/sends.log`（JSONL），每次成功发送写一行 `{ts, user_id, channel, recipient, subject_or_preview, message_id}`。落盘目录沿用现有 `log/` 约定。写入点是 `send_guard` wrapper 的成功分支。
5. **收件人白名单护栏（关键）**：LLM 只能发到 `contact.email` / `contact.feishu_open_id` / `feishu_group.chat_id` 里已录入的地址。**在 `send_guard` wrapper 层校验**（不在 MCP server 内），因为 wrapper 在 backend 进程里，能直接查 DB，职责最清。
   - `to / cc / bcc` 每个邮箱都要过白名单；
   - 飞书同理校验 `open_id` / `chat_id`；
   - 不通过 → 抛 `RecipientNotAllowedError`，LLM 可以复述给用户"请先把该联系人录入"。
6. **send 节点是串行 tool 调用**（`create_react_agent` 默认行为），错误恢复更友好。

## 7. 错误处理

| 错误层     | 触发                                     | 处理位置                | LLM 看到                                            |
|-----------|----------------------------------------|------------------------|----------------------------------------------------|
| 配置       | env 缺失                                | startup                | FastAPI 拒绝启动（fail-fast）                        |
| 传输       | stdio 子进程崩 / 超时                    | MCPClientManager       | 该 server 的工具下线，主功能不受影响（fail-open）      |
| 认证       | Gmail token 过期 / 飞书 access_token 失效 | server 内 typed error  | `{"error":"AUTH_FAILED","channel":"gmail",...}`     |
| 白名单     | 收件人未录入                             | send_guard wrapper     | `{"error":"RECIPIENT_NOT_ALLOWED",...}`             |
| 业务       | 飞书 API 错误码                          | feishu server          | `{"error":"LARK_API_ERROR","code":230001,...}`      |
| 参数       | LLM 拼错参数、附件文件不存在              | pydantic 校验          | pydantic 结构化错误回喂给 LLM 自纠                   |

原则：**启动期 fail-fast；运行期 fail-open**；错误必须结构化带 code；绝不吞异常。

### 7.1 飞书 tenant_access_token 缓存

- 有效期 2h；feishu server 进程内维护缓存 + 到期前 5min 主动刷新。
- 遇 "invalid access_token" 错误码，清 cache 并**内部重试一次**，仍失败才抛。
- **不用 Redis** —— 单进程、无跨机共享需求；未来横向扩再切换（TokenCache 抽象层预留）。

## 8. 测试策略

**单元测试**（pytest，`backend/tests/mcp/`）：

- `test_feishu_server.py` —— mock 飞书 REST，覆盖：发文字给用户 payload / 发文件到群 两步组装 / 错误码抛 typed error。
- `test_contacts_server.py` —— 测试 DB，覆盖精确 / 别名 / 模糊 / ambiguous / not found 五分支。
- `test_mcp_client_boot.py` —— 启动能拿工具列表 / 关闭正确清理 / env 缺失可读报错。
- `test_send_guard.py` —— 白名单放行 / 拦截 / 大小写与 trim 边界。

**手工 smoke**（`scripts/mcp_smoke_test.py`，本地跑不进 CI）：

- 依赖真 Gmail 沙箱账号 + 飞书自建应用；
- 依赖 `seed_contacts.py` 先录入一条自己的联系人；
- 走完 `resolve → send_email → send_file_to_user` 完整链路。

**Agent 端到端**（P2 可选）：小型 eval 集 5~10 条"发送类"话术，检查 tool 调用**签名**符合预期，不真发。

## 9. 启停与灰度

**FastAPI `@app.on_event("startup")`**（沿用现有钩子，不专门改成 lifespan）：

```
1. 读 .env（现有）
2. 初始化 DB / Redis（现有）
3. 读 config/mcp_servers.yaml → 展开 env 占位
4. MCPClientManager.start()
     ├─ 逐个 server 起子进程（并行）
     ├─ list_tools() 拿清单
     ├─ 应用 include_tools 白名单
     └─ send 类工具挂 send_guard wrapper
5. graph 构建时通过 mcp_manager.get_tools() 供 send 节点使用
6. 服务对外可用
```

**`@app.on_event("shutdown")`** 相反顺序，`terminate` + 5s 超时后 `kill`。

**引擎影响**：
- `AGENT_ENGINE=graph`（默认）：`send` 节点生效，发送能力可用。
- `AGENT_ENGINE=loop`：不注入 MCP 工具，Coordinator 也不参与判定 `send_intent`，用户说"发给张三"时 loop 引擎会直接把这句话作为聊天回答处理——**发送能力仅在 graph 引擎可用**，用户需要 `AGENT_ENGINE=graph`（P0 阶段这条限制在 README / release note 明写）。

**失败策略**：

- 某个 server 起不来 → 记 log，从 tool 列表剔除，不阻塞启动。
- 全部 server 起不来 → 不阻塞启动，明确警告日志；send 节点收到空工具列表时返回"发送能力当前不可用"。
- **fail-fast 唯一情形**：某 server 声明 `enabled=true` 但必填 env 缺失（如 `LARK_APP_ID` 空）→ 拒绝启动。

**灰度 & 回滚**：

- 总开关 `MCP_ENABLED=true|false`（false 时 `MCPClientManager` 空跑，`send` 节点也短路直接返回）；
- 每 server 单独 `enabled: true|false`；
- 回滚就 revert MCP 相关代码 or `MCP_ENABLED=false`。

## 10. 分期与验收

**P0 —— 骨架 & Gmail 通道（只支持 graph 引擎）**

1. 依赖：`mcp`, `langchain-mcp-adapters`, `langgraph`（已有，`create_react_agent` 由 `langgraph.prebuilt` 提供）
2. 建 `contact` / `feishu_group` 表 + service + CRUD router
3. `MCPClientManager`（stdio、启停、tools 暴露、白名单、send_guard 挂钩）
4. `.env.example` / `mcp_servers.yaml`
5. 接社区 gmail MCP server + `include_tools=["send_email"]`
6. `send_guard` wrapper（`RecipientNotAllowedError`）
7. `report_tool` / `form_tool` 输出加 `artifact_path`；`AgentState` 加 `artifact_path` 字段
8. Coordinator prompt 扩 `send_intent / recipients / channels`；`build.py` 加 `send` 节点和路由
9. `nodes/send.py`（`create_react_agent` 承载 MCP 工具）
10. `sends.log` 审计（在 send_guard 成功分支写入）
11. 单测：`test_mcp_client_boot` + `test_send_guard` + `test_contacts_server` + `test_send_node`（mock tools）
12. 手工 smoke：`scripts/mcp_smoke_test.py`

**P0 验收**：`AGENT_ENGINE=graph` 下对话说"把刚才那份报告发给 <已录入联系人>"，Agent 走完 graph 后经 send 节点，用 `contacts.resolve` + `gmail.send_email` 完成发送，附件到，`sends.log` 有一条记录。

**P1 —— 飞书通道**

1. `mcp_servers/feishu/`（server / lark_client / token_cache）
2. `test_feishu_server.py`（mock REST）
3. `send_guard` 扩到 feishu（open_id / chat_id 校验）
4. 手工 smoke：真 sandbox 应用 send_to_user / send_to_group / send_file_to_group

**P1 验收**：对话说"发给张三 + 发到运营群"，邮件到、飞书私聊到、群消息带附件到。

**P2 —— 打磨 & loop 收敛（各自独立 spec）**

1. Agent eval 集加"发送类"话术
2. contacts 后台管理 UI
3. 更细的错误码文案
4. **loop 引擎移除**（另起 `docs/specs/2026-07-XX-drop-loop-engine.md`，本 spec 不覆盖细节）

## 11. 用户侧准备（非代码）

上线前需要用户自行准备，本设计不代做：

1. **Gmail**：Google Cloud 项目 → 启用 Gmail API → 建 OAuth 客户端 → 下 `credentials.json` → 首次本地授权生成 `token.json` → 放 `secrets/`。
2. **飞书**：开放平台 → 创建自建应用 → 开启 `im:message` 和 `im:resource` 权限 → 拿 `app_id` / `app_secret` → bot 加入测试群拿 `chat_id`。
3. **联系人 seed**：录入自己 + 一个测试同事的 email / open_id 到 `contact` 表；录入一个测试群 `chat_id` 到 `feishu_group` 表。

## 12. Open Questions（可延后）

- 联系人管理页 UI 是走 Django admin 还是前端加页面？（P2 决定）
- `sends.log` 是否需要落库（`send_audit` 表）方便检索？（P2 视需要）
- 未来 per-user OAuth 接入时，`TokenCache` 抽象层如何抽？（真到那步再说）
- **loop 引擎移除的时间点**：等 graph 引擎在 P0/P1 稳定运行 2 周以上，且 `AGENT_ENGINE=graph` 默认打开后，另起 `2026-07-XX-drop-loop-engine.md` spec 走独立 plan 移除。

## 13. 变更记录

- **2026-07-03 初版**：双引擎都注入 MCP 工具。
- **2026-07-03 修订**：
  - 明确 ReAct（编排范式）与 function calling（通信协议）正交（§2.2）；
  - 澄清 loop = 手写 ReAct、graph = DAG（§2.3）；
  - P0 收敛到只在 graph 引擎里加 `send` 节点承载 MCP 工具，内部用 `create_react_agent`（§3、§3.1、§3.2）；
  - Coordinator 输出扩 `send_intent / recipients / channels`（§3.1）；
  - 数据流改成主图路由到 send 节点（§5.4、§6）；
  - loop 收敛作为独立 P2 spec 处理（§10、§12）。
