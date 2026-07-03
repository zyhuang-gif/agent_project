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

- **MCP Server**：对外提供工具的一方。本设计里的 gmail（社区包）、feishu（自写）、contacts（自写）都是 MCP server。
- **MCP Client**：调用别人 MCP Server 的一方。本项目后端 Agent 就是 Client。
- **`langchain-mcp-adapters`**：把 MCP 工具适配成 LangChain `BaseTool`。**MCP 客户端层与 LangChain / LangGraph 无关**——同一批 `BaseTool` 既能塞给 loop 引擎，也能塞给 graph 引擎，未来多 agent 图也用同一批工具。

## 3. 总体架构

```
┌────────────────────────────────────────────────────────┐
│ FastAPI backend (existing, port 8000)                  │
│                                                        │
│  ┌────────────────────────────────────────────────┐    │
│  │ Agent 引擎（loop 或 graph 二选一）              │    │
│  │                                                │    │
│  │  工具集 = [                                    │    │
│  │    rag_summary,           ← 已有              │    │
│  │    compare / form / report, ← 已有            │    │
│  │    ─── 以下由 MCP Client 动态加载 ───          │    │
│  │    gmail.send_email,                          │    │
│  │    feishu.send_message_to_user,               │    │
│  │    feishu.send_message_to_group,              │    │
│  │    feishu.send_file_to_user,                  │    │
│  │    feishu.send_file_to_group,                 │    │
│  │    contacts.resolve,                          │    │
│  │    contacts.list_all,                         │    │
│  │  ]                                             │    │
│  └────────────────────────────────────────────────┘    │
│              ↑ 都是 LangChain BaseTool                 │
│                                                        │
│  ┌────────────────────────────────────────────────┐    │
│  │ MCPClientManager  (app/mcp/client.py)          │    │
│  │  - startup 读 config/mcp_servers.yaml          │    │
│  │  - MultiServerMCPClient 起子进程                │    │
│  │  - 应用 include_tools 白名单                    │    │
│  │  - 挂"收件人白名单 wrapper"                     │    │
│  │  - shutdown 清理子进程                          │    │
│  └────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────┘
         │ stdio            │ stdio            │ stdio
         ▼                  ▼                  ▼
  ┌──────────────┐   ┌────────────────┐   ┌────────────────┐
  │ gmail MCP    │   │ feishu MCP     │   │ contacts MCP   │
  │ (社区 Node 包)│   │ (自写 Python)   │   │ (自写 Python)   │
  └──────────────┘   └────────────────┘   └────────────────┘
   Google OAuth        app_id/app_secret   MySQL (contact 表)
```

**关键决定**：

1. **contacts 独立 MCP server**，不塞进 feishu——邮件也要用它，抽出来后两条通道共享。
2. **loop / graph 两个引擎共用一份工具**，注入点分别是 `agent_tools.py`（loop）和 graph 构建处。
3. **stdio 传输**，本机子进程，不占端口。跨机再换 SSE/HTTP，接口不变。
4. **FastAPI lifespan** 里 startup 拉起 MCP 客户端，shutdown 关子进程。

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
│   │   ├── agent_tools.py               ← 改：并入 MCP 工具
│   │   └── graph/…                      ← 改：graph 引擎处同样并入
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
├── main.py                              ← 改：lifespan 里 MCP 启停
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

```
1. contacts.resolve("张三")     → {email, open_id}
2. contacts.resolve("运营组")   → {chat_id}
3. （report_tool 早前已生成 file_path，通过 artifact_path 字段带在上下文里）
4. gmail.send_email(to=[email], subject=..., attachments=[file_path])
5. feishu.send_file_to_group(chat_id=..., file_path=..., file_name="xxx报告.pdf")
```

## 6. 数据流 & 关键决定

1. **附件路径**：`report_tool` / `form_tool` 等工具的产出结构加 `artifact_path: str`（本地绝对路径）。Agent 在上下文里把这个路径带到 `send_email` / `send_file_to_group`。—— **本项目原有 tool 需要配合的一处小改。**
2. **幂等 / 重试**：不做客户端级去重；MCP 工具抛异常由 LangGraph `ToolNode` 默认行为处理（异常回喂 LLM 自决）。不做自动重试。
3. **审计日志**：`log/sends.log`（JSONL），每次成功发送写一行 `{ts, user_id, channel, recipient, subject_or_preview, message_id}`。落盘目录沿用现有 `log/` 约定。
4. **收件人白名单护栏（关键）**：LLM 只能发到 `contact.email` / `contact.feishu_open_id` / `feishu_group.chat_id` 里已录入的地址。**在 `send_guard` wrapper 层校验**（不在 MCP server 内），因为 wrapper 在 backend 进程里，能直接查 DB，职责最清。
   - `to / cc / bcc` 每个邮箱都要过白名单；
   - 飞书同理校验 `open_id` / `chat_id`；
   - 不通过 → 抛 `RecipientNotAllowedError`，LLM 可以复述给用户"请先把该联系人录入"。
5. **调用是串行**（不特意并行），错误恢复更友好。

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

**FastAPI lifespan（startup）**：

```
1. 读 .env（现有）
2. 初始化 DB / Redis（现有）
3. 读 config/mcp_servers.yaml → 展开 env 占位
4. MCPClientManager.start()
     ├─ 逐个 server 起子进程（并行）
     ├─ list_tools() 拿清单
     ├─ 应用 include_tools 白名单
     └─ send 类工具挂 send_guard wrapper
5. Agent 层从 MCPClientManager.get_tools() 拉工具
6. 服务对外可用
```

**shutdown 相反顺序**，`terminate` + 5s 超时后 `kill`。

**失败策略**：

- 某个 server 起不来 → 记 log，从 tool 列表剔除，不阻塞启动。
- 全部 server 起不来 → 不阻塞启动，明确警告日志。
- **fail-fast 唯一情形**：某 server 声明 `enabled=true` 但必填 env 缺失（如 `LARK_APP_ID` 空）→ 拒绝启动。

**灰度 & 回滚**：

- 总开关 `MCP_ENABLED=true|false`；
- 每 server 单独 `enabled: true|false`；
- 回滚就 revert MCP 相关代码 or `MCP_ENABLED=false`。

## 10. 分期与验收

**P0 —— 骨架 & Gmail 通道**

1. 依赖：`mcp`, `langchain-mcp-adapters`
2. 建 `contact` / `feishu_group` 表 + service + CRUD router
3. `MCPClientManager`（stdio、启停、tools 暴露、白名单）
4. `.env.example` / `mcp_servers.yaml`
5. 接社区 gmail MCP server + `include_tools=["send_email"]`
6. `send_guard` wrapper（`RecipientNotAllowedError`）
7. `report_tool` / `form_tool` 输出加 `artifact_path`
8. `sends.log` 审计
9. 单测：`test_mcp_client_boot` + `test_send_guard` + `test_contacts_server`
10. 手工 smoke：`scripts/mcp_smoke_test.py`

**P0 验收**：对话说"把刚才那份报告发给 <已录入联系人>"，Agent 能 `resolve → send_email`，附件到，`sends.log` 有一条记录。

**P1 —— 飞书通道**

1. `mcp_servers/feishu/`（server / lark_client / token_cache）
2. `test_feishu_server.py`（mock REST）
3. `send_guard` 扩到 feishu
4. 手工 smoke：真 sandbox 应用 send_to_user / send_to_group / send_file_to_group

**P1 验收**：对话说"发给张三 + 发到运营群"，邮件到、飞书私聊到、群消息带附件到。

**P2 —— 打磨（可选）**

1. Agent eval 集加"发送类"话术
2. contacts 后台管理 UI
3. 更细的错误码文案

## 11. 用户侧准备（非代码）

上线前需要用户自行准备，本设计不代做：

1. **Gmail**：Google Cloud 项目 → 启用 Gmail API → 建 OAuth 客户端 → 下 `credentials.json` → 首次本地授权生成 `token.json` → 放 `secrets/`。
2. **飞书**：开放平台 → 创建自建应用 → 开启 `im:message` 和 `im:resource` 权限 → 拿 `app_id` / `app_secret` → bot 加入测试群拿 `chat_id`。
3. **联系人 seed**：录入自己 + 一个测试同事的 email / open_id 到 `contact` 表；录入一个测试群 `chat_id` 到 `feishu_group` 表。

## 12. Open Questions（可延后）

- 联系人管理页 UI 是走 Django admin 还是前端加页面？（P2 决定）
- `sends.log` 是否需要落库（`send_audit` 表）方便检索？（P2 视需要）
- 未来 per-user OAuth 接入时，`TokenCache` 抽象层如何抽？（真到那步再说）
