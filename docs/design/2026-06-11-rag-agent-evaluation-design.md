# RAG / Agent 评估体系设计

> 状态：设计已评审，待实现。
> 关联：[求职视角项目评估与优化方向](../求职视角项目评估与优化方向.md)（本设计是其中「路径 A：补评估」的落地方案）、
> [项目弱点与竞品差距批判](../项目弱点与竞品差距批判.md)（评估直接回应其中「零评估体系」「HyDE 强制」「一次问答 5 次 LLM」三条批判）。
> 语料复用：[docs/test-corpus](../test-corpus/README.md)。

---

## 1. 背景与目标

项目目前**没有任何 RAG / Agent 质量评估**：检索准不准、回答有没有编、critic/HyDE/rerank 这些机制到底有没有用，全凭主观感觉；`RAG_GAP_THRESHOLD=0.75` 等阈值是肉眼观察拍的。这在 LLM 应用岗位是专业性红线——「做 LLM 应用却不做评估」等于「写代码不写测试」。

`docs/test-corpus/` 里已有一份**人工编写的测试问题清单**（分 5 类：knowledge_qa / document_compare / report_generation / document_generation / knowledge_gap），很多问题已标注预期答案（如「2025 版应答 550 元」）。它是 golden set 的雏形，但是 markdown 散文、未结构化、不能自动跑分。

**目标**：把它升级成一套**配置驱动、可重复、支持消融对比**的评估体系，能一条命令跑出「不同配置下的三层指标对比表 + 成本对比表」。

**成功标准**：
- 能产出一张 `summary.md`，并排展示 baseline / +critic / +hyde / +rerank 等配置在三层指标上的差异——这是简历/面试要讲的核心物证。
- 能用数据回答面试三连：「你怎么证明它好？」「critic 改完是变好还是变坏？」「为什么默认关 HyDE？」
- 评估器自身有单元测试，结果可重复。

## 2. 非目标（YAGNI 边界）

- 不做评估平台 / 可视化面板 / 历史趋势数据库 / 多模型矩阵 UI。
- 不追求统计显著（n≈40 是**指示性**样本，报告中如实标注）。
- 不做在线评估 / 生产流量回放。
- P0 不强制接 CI（runner 设计成可被 CI 调用，但门禁是 P2 可选项）。

## 3. 设计决策摘要（已评审确认）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 评估范围 | **三层全评**：检索层 + 回答层 + Agent 编排层 | 编排层（路由/缺口）是本项目差异化优势，量化「别人没有的东西」 |
| 裁判机制 | **混合裁判**：程序化断言 + LLM-as-judge + grounding 检查 | 事实点用程序判（便宜稳定可进 CI），开放式输出用 LLM 判 |
| 实现形态 | **配置驱动的可重复 runner + 消融对比** | 既能本地一键出对比表讲故事，又显工程能力，工作量可控 |
| 配置注入 | **子进程隔离 + 环境变量** | 绕开 import-time 常量污染，零状态泄漏 |
| 被测调用 | **in-process 直调**（子进程内 import，不起 web 服务） | 更快，且易拿中间结果（召回列表、路由轨迹） |
| LLM-judge 模型 | **阿里云千问 `qwen3-max`**（被测 chat 是 DeepSeek V4，judge 用千问 → 不同家族），temperature=0，`config.py` 可配 | 规避 self-enhancement bias，且复用已有阿里云 key |
| 数据集规模 | 由 ~15 扩到 **40~50 题**（每类 8~10） | n 太小无意义；扩充是半天人工活 |

## 4. 架构与目录

评估代码作为**独立工具层**，放在 `backend/eval/`（与 `app/`、`tests/` 平级，不混入运行时）：

```
backend/eval/
├── datasets/                    # golden set（结构化），语料复用 docs/test-corpus/
│   ├── retrieval.jsonl          #   检索/问答层标注
│   ├── tasks.jsonl              #   对比/报告/申请（开放式）
│   └── routing.jsonl            #   编排层：预期 coordinator 路由 + 是否该触发缺口
├── system_under_test.py         # 被测适配层：给定配置 → 返回每题(检索结果, 回答, 路由轨迹, token, 延迟)
├── judges/
│   ├── assertion_judge.py       #   程序化断言（数值/关键词命中）
│   ├── llm_judge.py             #   LLM-as-judge（开放式输出按 rubric 逐点核对）
│   └── grounding_judge.py       #   faithfulness：回答能否对齐到引用文档
├── metrics.py                   # 三层指标计算
├── config.py                    # 评估配置：消融开关组合、judge 模型、被测端点、k 值
├── runner.py                    # 编排：加载数据集→按配置起子进程跑→收集→裁判→算指标→出报告
├── seed_corpus.py               # 幂等灌库：test-corpus → 固定 eval 知识库
├── report.py                    # 报告生成（markdown 对比表 + json 明细）
└── reports/                     # 产物（每次跑一个带时间戳子目录）
```

**模块边界原则**：数据集（纯数据）/ 被测适配（只喂问题拿输出）/ 裁判（只给输出打分）/ 指标（只聚合）/ 报告（只渲染）五者解耦，各自可单测、可替换。换 judge 模型只动 `llm_judge.py`；加指标只动 `metrics.py`。

## 5. 数据集 schema

把散文问题清单转成机器可读 jsonl，每条一题。

### 5.1 retrieval.jsonl（检索 / 问答层）

```jsonc
{
  "id": "qa-001",
  "question": "一线城市出差住宿费每晚上限是多少？",
  "type": "knowledge_qa",
  "expected_doc": "02-差旅与报销管理办法-2025版.md",  // 检索命中标尺 → recall/MRR
  "answer_assertions": {                                  // 回答事实标尺（程序化断言）
    "must_include": ["550"],
    "must_not_include": ["450"]                           // 旧版数字，答错会命中
  },
  "should_refuse": false                                  // 是否本就该拒答
}
```

### 5.2 routing.jsonl（编排层）

```jsonc
{ "id": "rt-012", "question": "公司的股权激励计划细则？",
  "expected_route": "knowledge_gap",      // 预期 coordinator 分类
  "expect_gap_triggered": true }          // 预期触发缺口记录
```

### 5.3 tasks.jsonl（开放式输出，交 LLM-judge）

```jsonc
{ "id": "cmp-001", "question": "对比差旅报销 2023 版和 2025 版的差异？",
  "type": "document_compare",
  "rubric_points": ["市内交通80→120", "住宿一线450→550", "审批额度5000→8000",
                    "报销时限30→60天", "打款5→3工作日"],   // judge 逐点核对覆盖率
  "expected_docs": ["01-差旅与报销管理办法-2023版.md", "02-差旅与报销管理办法-2025版.md"] }
```

**要点**：`expected_doc` 是检索层 ground truth，`answer_assertions` 是回答层事实 ground truth，`rubric_points` 是开放式覆盖率 ground truth——**三层各有标尺，互不耦合**。负样本（`should_refuse` / `expect_gap_triggered`）单独标，用来量化缺口闭环和拒答的准确性。

## 6. 三层指标定义

| 层 | 指标 | 计算 | ground truth |
|---|---|---|---|
| **检索层** | recall@k | top-k 召回是否含 `expected_doc` | `expected_doc` |
| | hit-rate@k | 二值命中率 | `expected_doc` |
| | MRR | `expected_doc` 排名的倒数均值 | `expected_doc` |
| **回答层** | 事实正确率 | `must_include` 全中且 `must_not_include` 全不中的比例 | `answer_assertions` |
| | faithfulness | grounding judge：回答事实陈述能在引用文档找到依据的比例 | 引用文档 |
| | rubric 覆盖率 | LLM-judge 逐点核对覆盖的点数 / 总点数 | `rubric_points` |
| | 拒答正确率 | `should_refuse=true` 的题是否正确拒答（不编） | `should_refuse` |
| **编排层** | 路由准确率 | `predicted_route == expected_route` 比例 + 混淆矩阵 | `expected_route` |
| | 缺口触发 P/R | `expect_gap_triggered` 的精确率 / 召回率 | `expect_gap_triggered` |

**检索层适配点**：被测系统现在只把 top-3 当 citations 吐出来；评 recall@k 需要从 `get_documents_for_agent` 多暴露**完整召回列表 + 来源 filename + 排名**。这是个小改动，列入 P0。

## 7. 混合裁判

### 7.1 按题型分流

每题按 `type` 决定裁判：

- `knowledge_qa` → **assertion_judge**（程序化、零成本、可进 CI）+ **grounding_judge**
- `document_compare / report_generation / document_generation` → **llm_judge**（按 `rubric_points` 逐点核对覆盖率）+ **grounding_judge**
- `knowledge_gap` → 编排层校验（该不该触发缺口）
- 所有题 → 路由校验（coordinator 分类对不对）

### 7.2 三个裁判的职责

- **assertion_judge**：纯字符串/数值匹配 `answer_assertions`。确定性、零成本、可重复，是回答层的「测试断言」。
- **llm_judge**：对开放式输出，输入 `(question, answer, rubric_points)`，强制返回结构化 JSON `{covered_points: [...], score, reasoning}`。**逐点核对而非整体打分**，把主观「写得好不好」降维成客观「覆盖了哪几点」。
- **grounding_judge**：faithfulness。把回答拆成事实性陈述，逐句判断能否在该题引用的文档片段里找到依据，输出对齐比例。可用 LLM 判，也可先用片段重叠的轻量版打底。

### 7.3 LLM-judge 可信度控制（关键，面试加分点）

1. **judge 模型与被测生成模型分离**：被测 chat 是 **DeepSeek V4**（finalize 用 pro），judge 用**阿里云千问 `qwen3-max`** —— 不同家族，规避 self-enhancement bias。`config.py` 中 judge 模型独立可配。（注意：不要图省事让 judge 也用 deepseek-v4-pro，那和 finalize 同模型，自评偏袒。）
2. **temperature=0**：降低波动。
3. **强制结构化输出**：`{score, covered_points[], reasoning}`，不要自由文本打分。
4. **逐点 rubric**：主观降维成客观覆盖率。
5. **人工校准一致性**：抽 10 条人工打分，与 LLM-judge 算 agreement，**写进报告**——这是「评估的评估」，强差异化信号。

> 边界提醒：再强的 judge 也有 verbosity / positional bias。模型强是锦上添花，上面 1~5 的方法论才是可信度地基，二者都要做。

## 8. 消融对比运行模型（核心）

### 8.1 配置矩阵

一个「评估配置」= 一组环境变量开关：

```
baseline : AGENT_ENGINE=graph  AGENT_CRITIC_ENABLE=false  RAG_HYDE_ENABLE=false  RERANK_ENABLE=true
+critic  : AGENT_CRITIC_ENABLE=true
+hyde    : RAG_HYDE_ENABLE=true
-rerank  : RERANK_ENABLE=false        # 对照「关掉 rerank」
```

runner 接受配置矩阵，逐配置跑全量数据集，最后并排成对比表。

### 8.2 import-time 常量约束（必须处理的坑）

现有开关两种读法：
- **运行时函数内读**（好）：`AGENT_CRITIC_ENABLE`、`AGENT_ENGINE`、`AGENT_CRITIC_MAX_REVISIONS`、`RAG_GENERATION_MODE`。
- **模块顶层 import 时读死**（坑）：`_GAP_THRESHOLD`、`_CONFIDENCE_THRESHOLD`（`rag_service.py`）、`RERANKER_TYPE`（`reorder_service.py`）。

import-time 读取 → 同进程改 env 再 re-import 不生效 → in-process 注入不可靠。

**解法（采用）：子进程隔离。** 每个配置起一个子进程，先 set env 再启动 → 子进程 import 时即读到正确开关 → 跑完整个数据集吐 json → 父进程收集。干净、零污染、天然隔离，代价仅每配置一次进程启动开销（可接受）。

> 可选优化（非 P0）：把那几个 import-time 常量重构成运行时读取，顺手还设计债。

### 8.3 要补的能力开关（评估逼出来的整改）

| 开关 | 现状 | 动作 | 阶段 | 附带收益 |
|---|---|---|---|---|
| `RAG_HYDE_ENABLE` | **无**，HyDE 在 `retrieve_document` 硬编码强制 | 新增，默认可配 | **P0** | 解决批判「HyDE 强制」 |
| `RERANK_ENABLE`（或 reorder 加 `NONE` 模式） | 只能切 LOCAL/ALIYUN，不能关 | 新增 bypass | P1 | 量化 rerank 真实增益 |

### 8.4 被测调用方式

子进程内**直接 import** `app` 的 graph / rag_service 跑，**不打 HTTP**——更快，且易拿中间结果（召回列表、路由轨迹）。要求 MySQL / Redis / 向量库 / embedding / rerank / LLM 真实依赖在跑（评估必须真实）。

> **被测模型栈（评估需真实在跑）**：chat = **DeepSeek V4**（分角色：coordinator / knowledge_gap / rag = flash，finalize = pro）；embedding = 千问 `text-embedding-v4`；rerank = 千问 `qwen3-vl-rerank`。因此评估环境需同时具备 **DeepSeek key + 阿里云千问 key**。judge 用千问 `qwen3-max`，与被测 chat（DeepSeek）不同家族。

### 8.5 语料准备

`seed_corpus.py` 评估前用 `test-corpus` **幂等灌一个固定的 eval 知识库**（company 公开范围），保证每次评估语料一致、可复现。

## 9. 端到端数据流

```
① seed_corpus.py 幂等灌库（test-corpus → 固定 eval KB）
② runner 读 配置矩阵 + 数据集
③ 对每个配置：起子进程(注入 env) → 逐题调被测适配层
       → 收集 {检索结果, 回答, 路由轨迹, token, 延迟} → 写中间 json
④ 父进程对所有结果跑裁判（按题型分流）→ metrics 算三层指标
⑤ report.py 渲染：
       summary.md    （配置 × 指标 对比表）
       cost.md       （配置 × 延迟/token 成本表）
       details.json  （每题每配置原始输出+分数）
       regressions.md（失败/退步样本，便于 debug）
⑥ 可选(P2)：关键指标 assert 进 pytest 做回归门禁
```

## 10. 产物

每次评估输出到 `reports/<timestamp>/`：

- **summary.md**：面试要用的那张三层指标对比表。
- **cost.md**：延迟 + token 成本表。HyDE / map-reduce 的开关对比直接出成本差异——用数据回击「一次问答打 5 次 LLM」：「开 HyDE 召回涨 X%，但延迟涨 Y%、token 涨 Z%，故默认关」。**用数据做工程决策的铁证。**
- **details.json**：全量明细，复盘用。
- **regressions.md**：哪些题在哪个配置下挂了/退步了。

## 11. 分阶段

### P0（一个周末，核心故事）
- 数据集结构化：retrieval + tasks 两类，~30 题。
- `seed_corpus.py` 幂等灌库。
- 子进程 runner + 配置矩阵。
- 检索层指标（recall@k / hit-rate / MRR）+ 回答层事实断言（assertion_judge）。
- 检索层适配点：`get_documents_for_agent` 暴露完整召回列表。
- 新增 `RAG_HYDE_ENABLE` 开关。
- 产出 `summary.md` + `cost.md`：baseline / +critic / +hyde 三配置对比。

### P1（差异化加分）
- 开放式 LLM-judge（rubric 覆盖率）+ grounding faithfulness。
- 编排层：路由准确率（混淆矩阵）+ 缺口触发精确/召回。
- LLM-judge 人工校准一致性（10 条）。
- 新增 `RERANK_ENABLE` bypass。
- **阈值扫描**：`RAG_GAP_THRESHOLD` 从 0.6~0.85 扫出曲线 → 把「肉眼拍 0.75」变成「数据选 0.75」。
- 数据集补到 40~50 题（含 routing.jsonl）。

### P2（可选）
- CI 回归门禁（关键指标低于基线则 fail）。
- over-HTTP smoke（起服务打 `/api/agent/query/stream` 做端到端真实性校验）。
- import-time 常量重构成运行时读取。

## 12. 评估器自身的测试

`judges/` 与 `metrics.py` 用合成输入做单元测试（已知输入 → 已知分数），保证评估器无 bug。放 `backend/tests/` 下，纳入现有 pytest。**给评估代码也写测试**——专业性信号。

## 13. 风险与取舍

| 风险 | 缓解 |
|---|---|
| LLM-judge 成本/波动 | temperature=0 + 逐点 rubric + 强模型 + 人工校准 |
| import-time 常量污染配置 | 子进程隔离 |
| 评估依赖真实中间件，环境重 | `seed_corpus.py` 幂等 + 文档写明前置依赖 |
| 数据集小（n≈40），非统计显著 | 报告如实标注样本量，定位为「指示性」 |
| judge 与被测同模型的自评偏袒 | judge 默认用不同家族强模型 |

## 14. 成功标准（验收）

- 一条命令跑完一个配置矩阵，产出 `summary.md` + `cost.md`。
- 对比表能清楚显示 critic / hyde / rerank 各自对三层指标和成本的影响。
- 阈值扫描曲线能为 `RAG_GAP_THRESHOLD` 的取值给出数据支撑。
- 评估器单测通过；同一配置重复跑，程序化指标完全一致、LLM-judge 指标在小波动内。
- README 能引用对比表，简历能写出「用 N 题标注集做三层评估 + 消融，量化每次改动收益」。
