# RAG/Agent 评估体系 P0 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 复选框跟踪。
>
> **关联设计**：[docs/design/2026-06-11-rag-agent-evaluation-design.md](../design/2026-06-11-rag-agent-evaluation-design.md)（本计划只实现其中 **P0**）。
>
> **版本控制说明**：实现代码**需要 commit**（按项目中文 conventional commit 风格，如 `test:` / `feat:`）。但**设计/计划文档不要 commit**——`docs/` 下本计划、设计 spec 及两份评估文档保持 untracked。
> 每个 Task 测试通过后，commit 该 Task 涉及的**代码文件**：用 `git add <具体文件>` 精确添加，**禁止 `git add -A` / `git add .`**（否则会把 `docs/` 设计文档一并误提交）。

**Goal:** 给项目补一套配置驱动、可重复的评估管线，一条命令跑出 baseline / +critic / +hyde 三配置在「检索层 + 回答层事实断言」上的对比表与成本表。

**Architecture:** 评估代码独立放 `backend/eval/`，不混入运行时。纯逻辑模块（schema/judges/metrics/report/config）走 TDD 单测；被测系统通过 `GraphRunner().stream(...)` 真实驱动；消融配置用**子进程 + 环境变量**注入（绕开 import-time 常量污染），每个配置一个子进程跑全量数据集吐 json，父进程聚合算指标出报告。检索层指标直接复用 `done.citations` 的 filename+排名。

**Tech Stack:** Python 3.12、pytest（asyncio_mode=auto）、LangGraph（被测）、子进程隔离、jsonl 数据集。被测模型栈：chat=DeepSeek V4，embedding/rerank=千问（评估需真实依赖在跑）。

---

## 文件结构

```
backend/eval/
├── __init__.py                  # 空包标记
├── datasets/
│   └── retrieval.jsonl          # golden set（P0：knowledge_qa，带 expected_doc + answer_assertions）
├── schema.py                    # EvalCase dataclass + load_cases()
├── judges/
│   ├── __init__.py
│   └── assertion_judge.py       # check_assertions()（纯函数）
├── metrics.py                   # recall_at_k / mrr / aggregate()（纯函数）
├── report.py                    # render_summary / render_cost（纯函数）
├── config.py                    # EvalConfig + CONFIG_MATRIX（纯数据）
├── system_under_test.py         # run_dataset()：真实驱动 GraphRunner
├── seed_corpus.py               # 幂等灌库（test-corpus → 向量库）
└── runner.py                    # 父编排 + 子进程 worker 入口

backend/app/rag/rag_service.py   # 修改：新增 RAG_HYDE_ENABLE 开关
backend/tests/                   # 新增评估器单测
├── test_eval_schema.py
├── test_eval_assertion_judge.py
├── test_eval_metrics.py
├── test_eval_report.py
├── test_eval_config.py
├── test_eval_sut.py
└── test_rag_hyde_toggle.py
```

每个文件单一职责：数据（schema/datasets）/ 裁判（judges）/ 指标（metrics）/ 渲染（report）/ 配置（config）/ 被测驱动（system_under_test）/ 灌库（seed_corpus）/ 编排（runner）解耦，可独立测试与替换。

---

### Task 1: 评估包骨架 + 数据集 schema

**Files:**
- Create: `backend/eval/__init__.py`（空文件）
- Create: `backend/eval/judges/__init__.py`（空文件）
- Create: `backend/eval/schema.py`
- Create: `backend/eval/datasets/retrieval.jsonl`
- Test: `backend/tests/test_eval_schema.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_schema.py
import json
from pathlib import Path
from eval.schema import EvalCase, load_cases


def test_load_cases_parses_fields(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({
        "id": "qa-001",
        "question": "一线城市出差住宿费每晚上限是多少？",
        "type": "knowledge_qa",
        "expected_doc": "02-差旅与报销管理办法-2025版.md",
        "answer_assertions": {"must_include": ["550"], "must_not_include": ["450"]},
        "should_refuse": False,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    cases = load_cases(p)
    assert len(cases) == 1
    c = cases[0]
    assert isinstance(c, EvalCase)
    assert c.id == "qa-001"
    assert c.expected_doc == "02-差旅与报销管理办法-2025版.md"
    assert c.answer_assertions["must_include"] == ["550"]
    assert c.should_refuse is False


def test_load_cases_skips_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('\n\n{"id":"x","question":"q","type":"knowledge_qa"}\n\n', encoding="utf-8")
    cases = load_cases(p)
    assert len(cases) == 1
    assert cases[0].answer_assertions == {}   # 缺字段给默认


def test_real_dataset_loads_and_is_nonempty():
    cases = load_cases(Path(__file__).parent.parent / "eval" / "datasets" / "retrieval.jsonl")
    assert len(cases) >= 8
    assert all(c.id and c.question and c.type for c in cases)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_schema.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'eval'`）

- [ ] **Step 3: 建空包文件**

`backend/eval/__init__.py` 和 `backend/eval/judges/__init__.py` 均为空文件（0 字节）。

- [ ] **Step 4: 写 schema.py**

```python
# backend/eval/schema.py
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalCase:
    id: str
    question: str
    type: str                              # knowledge_qa / document_compare / ...
    expected_doc: Optional[str] = None     # 检索命中标尺（文件名）
    answer_assertions: dict = field(default_factory=dict)  # {"must_include":[], "must_not_include":[]}
    should_refuse: bool = False
    history: list = field(default_factory=list)            # [[user, assistant], ...]


def load_cases(path) -> list[EvalCase]:
    """从 jsonl 加载评估用例，跳过空行。"""
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        cases.append(EvalCase(
            id=raw["id"],
            question=raw["question"],
            type=raw["type"],
            expected_doc=raw.get("expected_doc"),
            answer_assertions=raw.get("answer_assertions", {}),
            should_refuse=raw.get("should_refuse", False),
            history=raw.get("history", []),
        ))
    return cases
```

- [ ] **Step 5: 写初始数据集 retrieval.jsonl**

每行一个 JSON。下面 8 条取自 `docs/test-corpus/README.md` 的真实问题清单与预期答案，文件名与 `docs/test-corpus/` 一致。**后续按相同格式补到 ~30 题**（每篇语料挖 4~5 个事实型问题，标 `expected_doc` 与 `answer_assertions`）。

```jsonl
{"id":"qa-001","question":"一线城市出差住宿费每晚上限是多少？","type":"knowledge_qa","expected_doc":"02-差旅与报销管理办法-2025版.md","answer_assertions":{"must_include":["550"],"must_not_include":["450"]},"should_refuse":false}
{"id":"qa-002","question":"工龄满12年每年有几天年假？","type":"knowledge_qa","expected_doc":"03-考勤与请假管理制度.md","answer_assertions":{"must_include":["10"]},"should_refuse":false}
{"id":"qa-003","question":"远程办公申请需要满足哪些条件？","type":"knowledge_qa","expected_doc":"04-远程办公管理规定.md","answer_assertions":{"must_include":["申请"]},"should_refuse":false}
{"id":"qa-004","question":"试用期员工有带薪年假吗？","type":"knowledge_qa","expected_doc":"05-员工入职与转正管理办法.md","answer_assertions":{"must_include":["转正"]},"should_refuse":false}
{"id":"qa-005","question":"笔记本电脑多久更换一次？","type":"knowledge_qa","expected_doc":"06-IT设备与办公用品管理规定.md","answer_assertions":{"must_include":["4"]},"should_refuse":false}
{"id":"qa-006","question":"2025版差旅报销的审批额度上限是多少？","type":"knowledge_qa","expected_doc":"02-差旅与报销管理办法-2025版.md","answer_assertions":{"must_include":["8000"],"must_not_include":["5000"]},"should_refuse":false}
{"id":"qa-007","question":"报销时限是多少天？","type":"knowledge_qa","expected_doc":"02-差旅与报销管理办法-2025版.md","answer_assertions":{"must_include":["60"]},"should_refuse":false}
{"id":"qa-008","question":"公司的股权激励计划细则是怎样的？","type":"knowledge_qa","expected_doc":null,"answer_assertions":{},"should_refuse":true}
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_schema.py -v`
Expected: PASS（3 passed）

---

### Task 2: 程序化断言裁判 assertion_judge

**Files:**
- Create: `backend/eval/judges/assertion_judge.py`
- Test: `backend/tests/test_eval_assertion_judge.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_assertion_judge.py
from eval.judges.assertion_judge import check_assertions


def test_must_include_all_present_passes():
    assert check_assertions("住宿上限为550元", {"must_include": ["550"]}) is True


def test_must_include_missing_fails():
    assert check_assertions("住宿上限较高", {"must_include": ["550"]}) is False


def test_must_not_include_hit_fails():
    assert check_assertions("旧版是450元", {"must_include": [], "must_not_include": ["450"]}) is False


def test_empty_assertions_passes():
    assert check_assertions("任意回答", {}) is True


def test_multiple_must_include_partial_fails():
    assert check_assertions("只有8000", {"must_include": ["8000", "60"]}) is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_assertion_judge.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 assertion_judge.py**

```python
# backend/eval/judges/assertion_judge.py
def check_assertions(answer: str, assertions: dict) -> bool:
    """事实型回答的程序化断言：must_include 全中且 must_not_include 全不中。

    空 assertions 视为通过（该题不做事实断言，交其它裁判）。
    """
    text = answer or ""
    must = assertions.get("must_include", []) or []
    must_not = assertions.get("must_not_include", []) or []
    if not all(s in text for s in must):
        return False
    if any(s in text for s in must_not):
        return False
    return True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_assertion_judge.py -v`
Expected: PASS（5 passed）

---

### Task 3: 检索/回答指标 metrics

**Files:**
- Create: `backend/eval/metrics.py`
- Test: `backend/tests/test_eval_metrics.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_metrics.py
from eval.metrics import recall_at_k, mrr, aggregate


def test_recall_hit_within_k():
    assert recall_at_k(["a.md", "b.md", "c.md"], "b.md", k=3) == 1.0


def test_recall_miss_outside_k():
    assert recall_at_k(["a.md", "b.md", "c.md"], "b.md", k=1) == 0.0


def test_recall_no_expected_returns_none():
    # expected_doc 为 None（如纯拒答题）→ 该指标不适用
    assert recall_at_k(["a.md"], None, k=3) is None


def test_mrr_rank_two():
    assert mrr(["a.md", "b.md"], "b.md") == 0.5


def test_mrr_not_found():
    assert mrr(["a.md"], "z.md") == 0.0


def test_aggregate_means_ignore_none():
    per_case = [
        {"recall@3": 1.0, "mrr": 1.0, "assert_pass": True},
        {"recall@3": 0.0, "mrr": 0.0, "assert_pass": False},
        {"recall@3": None, "mrr": None, "assert_pass": True},   # 拒答题不计检索
    ]
    agg = aggregate(per_case)
    assert agg["recall@3"] == 0.5          # (1+0)/2，None 不计入
    assert agg["mrr"] == 0.5
    assert agg["assert_pass_rate"] == 2 / 3
    assert agg["n"] == 3
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 metrics.py**

```python
# backend/eval/metrics.py
from typing import Optional


def recall_at_k(ranked_filenames: list, expected_doc: Optional[str], k: int) -> Optional[float]:
    """expected_doc 是否落在 top-k 召回内。expected_doc 为 None 时返回 None（不适用）。"""
    if expected_doc is None:
        return None
    return 1.0 if expected_doc in (ranked_filenames or [])[:k] else 0.0


def mrr(ranked_filenames: list, expected_doc: Optional[str]) -> Optional[float]:
    """expected_doc 在召回排名的倒数。未命中得 0；expected_doc 为 None 返回 None。"""
    if expected_doc is None:
        return None
    for i, f in enumerate(ranked_filenames or [], 1):
        if f == expected_doc:
            return 1.0 / i
    return 0.0


def _mean_ignore_none(values: list) -> Optional[float]:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def aggregate(per_case: list) -> dict:
    """把每题指标聚合成数据集级指标。None 不计入均值。"""
    return {
        "n": len(per_case),
        "recall@3": _mean_ignore_none([c.get("recall@3") for c in per_case]),
        "mrr": _mean_ignore_none([c.get("mrr") for c in per_case]),
        "assert_pass_rate": (
            sum(1 for c in per_case if c.get("assert_pass")) / len(per_case)
            if per_case else None
        ),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py -v`
Expected: PASS（6 passed）

---

### Task 4: 报告渲染 report

**Files:**
- Create: `backend/eval/report.py`
- Test: `backend/tests/test_eval_report.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_report.py
from eval.report import render_summary, render_cost


def test_summary_has_header_and_config_rows():
    matrix = {
        "baseline": {"n": 8, "recall@3": 0.75, "mrr": 0.62, "assert_pass_rate": 0.5},
        "+critic":  {"n": 8, "recall@3": 0.875, "mrr": 0.70, "assert_pass_rate": 0.625},
    }
    md = render_summary(matrix)
    assert "| 配置 |" in md
    assert "baseline" in md
    assert "+critic" in md
    assert "0.750" in md          # 数值格式化到 3 位
    assert "0.875" in md


def test_cost_table_renders_tokens_and_latency():
    cost = {
        "baseline": {"avg_tokens": 1200.0, "avg_latency_s": 3.4},
        "+hyde":    {"avg_tokens": 1800.0, "avg_latency_s": 5.1},
    }
    md = render_cost(cost)
    assert "avg_tokens" in md or "平均 token" in md
    assert "1200" in md
    assert "5.1" in md
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_report.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 report.py**

```python
# backend/eval/report.py
def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_summary(matrix: dict) -> str:
    """matrix: {config_name: {n, recall@3, mrr, assert_pass_rate}} → markdown 对比表。"""
    lines = [
        "# 评估对比表（检索层 + 回答层事实断言）",
        "",
        "| 配置 | n | recall@3 | MRR | 事实断言通过率 |",
        "|---|---|---|---|---|",
    ]
    for name, m in matrix.items():
        lines.append(
            f"| {name} | {_fmt(m.get('n'))} | {_fmt(m.get('recall@3'))} | "
            f"{_fmt(m.get('mrr'))} | {_fmt(m.get('assert_pass_rate'))} |"
        )
    lines.append("")
    lines.append("> 样本量小（指示性，非统计显著）。None/— 表示该指标对该题不适用。")
    return "\n".join(lines)


def render_cost(cost: dict) -> str:
    """cost: {config_name: {avg_tokens, avg_latency_s}} → markdown 成本表。"""
    lines = [
        "# 成本对比表（延迟 / token）",
        "",
        "| 配置 | 平均 token (avg_tokens) | 平均延迟秒 (avg_latency_s) |",
        "|---|---|---|",
    ]
    for name, c in cost.items():
        lines.append(f"| {name} | {_fmt(c.get('avg_tokens'))} | {_fmt(c.get('avg_latency_s'))} |")
    return "\n".join(lines)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_report.py -v`
Expected: PASS（2 passed）

---

### Task 5: 消融配置矩阵 config

**Files:**
- Create: `backend/eval/config.py`
- Test: `backend/tests/test_eval_config.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_config.py
from eval.config import EvalConfig, CONFIG_MATRIX


def test_matrix_has_three_p0_configs():
    names = [c.name for c in CONFIG_MATRIX]
    assert names == ["baseline", "+critic", "+hyde"]


def test_all_configs_use_graph_engine():
    assert all(c.env.get("AGENT_ENGINE") == "graph" for c in CONFIG_MATRIX)


def test_baseline_disables_critic_and_hyde():
    base = next(c for c in CONFIG_MATRIX if c.name == "baseline")
    assert base.env["AGENT_CRITIC_ENABLE"] == "false"
    assert base.env["RAG_HYDE_ENABLE"] == "false"


def test_critic_config_enables_only_critic():
    c = next(c for c in CONFIG_MATRIX if c.name == "+critic")
    assert c.env["AGENT_CRITIC_ENABLE"] == "true"
    assert c.env["RAG_HYDE_ENABLE"] == "false"


def test_hyde_config_enables_only_hyde():
    c = next(c for c in CONFIG_MATRIX if c.name == "+hyde")
    assert c.env["RAG_HYDE_ENABLE"] == "true"
    assert c.env["AGENT_CRITIC_ENABLE"] == "false"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_config.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 config.py**

```python
# backend/eval/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    name: str
    env: dict          # 该配置要注入的环境变量（覆盖在当前 env 之上）


# P0 消融矩阵：在 graph 引擎下，逐个打开 critic / hyde 与 baseline 对照。
# 每个配置都显式写全三个开关，避免继承上一配置的残留。
_GRAPH = {"AGENT_ENGINE": "graph"}

CONFIG_MATRIX = [
    EvalConfig("baseline", {**_GRAPH, "AGENT_CRITIC_ENABLE": "false", "RAG_HYDE_ENABLE": "false"}),
    EvalConfig("+critic",  {**_GRAPH, "AGENT_CRITIC_ENABLE": "true",  "RAG_HYDE_ENABLE": "false"}),
    EvalConfig("+hyde",    {**_GRAPH, "AGENT_CRITIC_ENABLE": "false", "RAG_HYDE_ENABLE": "true"}),
]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_config.py -v`
Expected: PASS（5 passed）

---

### Task 6: 给 HyDE 加 RAG_HYDE_ENABLE 开关（改 app）

**Files:**
- Modify: `backend/app/rag/rag_service.py`（`retrieve_document` 内的 HyDE 调用 + 新增 `_hyde_enabled`）
- Test: `backend/tests/test_rag_hyde_toggle.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_rag_hyde_toggle.py
import pytest
from app.rag import rag_service as rs_mod
from app.rag.rag_service import rag_service, _hyde_enabled


def test_hyde_enabled_default_true(monkeypatch):
    monkeypatch.delenv("RAG_HYDE_ENABLE", raising=False)
    assert _hyde_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "FALSE"])
def test_hyde_disabled_values(monkeypatch, val):
    monkeypatch.setenv("RAG_HYDE_ENABLE", val)
    assert _hyde_enabled() is False


@pytest.mark.asyncio
async def test_retrieve_document_skips_hyde_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_HYDE_ENABLE", "false")

    # 假检索器：ainvoke 返回空子块，避免触达真实向量库
    class _FakeRetriever:
        async def ainvoke(self, q):
            _FakeRetriever.last_query = q
            return []
    fake = _FakeRetriever()
    monkeypatch.setattr(rag_service, "retriever", fake)

    # 监视 HyDE 是否被调用
    called = {"n": 0}
    async def _spy(query):
        called["n"] += 1
        return "HYPO:" + query
    monkeypatch.setattr(rag_service, "generate_hypothetical_document", _spy)

    docs = await rag_service.retrieve_document("一线城市住宿上限")
    assert called["n"] == 0                      # 关 HyDE → 不生成假设文档
    assert fake.last_query == "一线城市住宿上限"  # 直接用原 query 检索
    assert docs == []


@pytest.mark.asyncio
async def test_retrieve_document_uses_hyde_when_enabled(monkeypatch):
    monkeypatch.setenv("RAG_HYDE_ENABLE", "true")

    class _FakeRetriever:
        async def ainvoke(self, q):
            _FakeRetriever.last_query = q
            return []
    fake = _FakeRetriever()
    monkeypatch.setattr(rag_service, "retriever", fake)

    async def _spy(query):
        return "HYPO:" + query
    monkeypatch.setattr(rag_service, "generate_hypothetical_document", _spy)

    await rag_service.retrieve_document("一线城市住宿上限")
    assert fake.last_query == "HYPO:一线城市住宿上限"   # 用假设文档检索
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_rag_hyde_toggle.py -v`
Expected: FAIL（`ImportError: cannot import name '_hyde_enabled'`）

- [ ] **Step 3: 改 rag_service.py**

在 `backend/app/rag/rag_service.py` 顶部、`_GAP_THRESHOLD = ...`（约第 28 行）之后，新增开关函数：

```python
def _hyde_enabled() -> bool:
    """HyDE 开关（运行时读，便于子进程 env 注入与测试 monkeypatch）。
    默认 true，保持原有行为；设 false/0/no 则跳过假设文档生成，直接用原 query 检索。
    """
    return os.getenv("RAG_HYDE_ENABLE", "true").strip().lower() not in ("false", "0", "no")
```

然后在 `retrieve_document` 中，把原来这段（约第 118-120 行）：

```python
        total_t0 = time.perf_counter()
        logger.info(f"【HyDE】开始处理查询: {query}")
        hypothetical_doc = await self.generate_hypothetical_document(query)
```

改为：

```python
        total_t0 = time.perf_counter()
        logger.info(f"【HyDE】开始处理查询: {query}")
        if _hyde_enabled():
            hypothetical_doc = await self.generate_hypothetical_document(query)
        else:
            hypothetical_doc = query
            logger.info("【HyDE】RAG_HYDE_ENABLE=false，跳过假设文档，直接用原 query 检索")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_rag_hyde_toggle.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: 回归确认未破坏既有 RAG 测试**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/ -k "rag or graph" -q`
Expected: 全部 PASS（无新增失败）

---

### Task 7: 被测系统驱动 system_under_test

**Files:**
- Create: `backend/eval/system_under_test.py`
- Test: `backend/tests/test_eval_sut.py`

把 `GraphRunner().stream(...)` 的事件流解析成结构化结果。解析逻辑（纯函数 `parse_events`）单独可测；真实驱动 `run_dataset` 调它。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_sut.py
from eval.system_under_test import parse_events


def test_parse_events_extracts_answer_citations_tokens():
    events = [
        {"type": "token", "data": "住宿"},
        {"type": "token", "data": "上限550元"},
        {"type": "done", "steps": [{"agent": "knowledge"}, {"agent": "finalize"}],
         "tokens": 1234, "citations": [
             {"filename": "02-差旅与报销管理办法-2025版.md", "score": 0.92},
             {"filename": "01-差旅与报销管理办法-2023版.md", "score": 0.55}]},
    ]
    r = parse_events(events)
    assert r["answer"] == "住宿上限550元"
    assert r["ranked_filenames"] == ["02-差旅与报销管理办法-2025版.md",
                                     "01-差旅与报销管理办法-2023版.md"]
    assert r["tokens"] == 1234
    assert r["trace_agents"] == ["knowledge", "finalize"]


def test_parse_events_handles_missing_done():
    r = parse_events([{"type": "token", "data": "x"}])
    assert r["answer"] == "x"
    assert r["ranked_filenames"] == []
    assert r["tokens"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 system_under_test.py**

```python
# backend/eval/system_under_test.py
import time


def parse_events(events: list) -> dict:
    """把 GraphRunner.stream 的事件流解析成结构化结果。"""
    answer = "".join(e["data"] for e in events if e.get("type") == "token")
    done = next((e for e in reversed(events) if e.get("type") == "done"), {})
    citations = done.get("citations", []) or []
    return {
        "answer": answer,
        "ranked_filenames": [c.get("filename") for c in citations],
        "tokens": done.get("tokens", 0) or 0,
        "trace_agents": [s.get("agent") for s in (done.get("steps", []) or [])],
    }


async def run_one(runner, question: str, history: list) -> dict:
    """用已编译的 runner 跑一题，附带延迟。"""
    t0 = time.perf_counter()
    events = [e async for e in runner.stream(question, history=history or [], identity=None)]
    result = parse_events(events)
    result["latency_s"] = time.perf_counter() - t0
    return result


async def run_dataset(cases: list) -> list:
    """对一组 EvalCase 真实驱动被测系统（当前进程的 env 决定配置）。
    每个配置在子进程内调用本函数，故这里只建一次 GraphRunner（按当前 env 编译）。
    """
    from app.agent.graph.runner import GraphRunner
    runner = GraphRunner()
    out = []
    for c in cases:
        r = await run_one(runner, c.question, c.history)
        out.append({"id": c.id, **r})
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py -v`
Expected: PASS（2 passed）

---

### Task 8: 幂等灌库 seed_corpus

**Files:**
- Create: `backend/eval/seed_corpus.py`
- 手动验证（依赖真实向量库/MySQL/embedding，不写自动化单测）

`vector_store.get_document` 接受带 `.filename` 与 async `.read()` 的 file-like 列表，内部按 MD5 去重 → 天然幂等。把 `docs/test-corpus/*.md` 灌进去。`user_id` 固定为评估专用 id，`kb_id=None`（个人范围即可——评估用 `identity=None` 检索全库，不过滤权限）。

- [ ] **Step 1: 写 seed_corpus.py**

```python
# backend/eval/seed_corpus.py
"""幂等把 docs/test-corpus 灌入向量库，供评估检索。

用法（在 backend 目录）：
    .\.venv\Scripts\python.exe -m eval.seed_corpus
重复执行安全：vector_store 按 MD5 去重，已灌的文件自动跳过。
"""
import asyncio
import os
from pathlib import Path

from app.rag.vector_store import VectorStoreService

EVAL_USER_ID = "eval-bot"
CORPUS_DIR = Path(__file__).resolve().parents[2] / "docs" / "test-corpus"


class _LocalFile:
    """伪 UploadFile：vector_store.get_document 只用到 .filename 与 await .read()。"""
    def __init__(self, path: Path):
        self.filename = path.name
        self._path = path

    async def read(self) -> bytes:
        return await asyncio.to_thread(self._path.read_bytes)


async def main():
    files = [_LocalFile(p) for p in sorted(CORPUS_DIR.glob("*.md"))]
    if not files:
        raise SystemExit(f"未在 {CORPUS_DIR} 找到 .md 语料")
    vs = VectorStoreService()
    result = await vs.get_document(files=files, user_id=EVAL_USER_ID, kb_id=None)
    print(f"[seed] processed={result['processed']} duplicates={result['duplicates']}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 真实灌库（需 MySQL/Redis/向量库/embedding 在跑）**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.seed_corpus`
Expected: 打印 `[seed] processed=[...6 个 doc_id...] duplicates=[]`

- [ ] **Step 3: 再跑一次验证幂等**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.seed_corpus`
Expected: 打印 `processed=[] duplicates=['01-...md', ...6 个文件名]`（全部去重跳过）

---

### Task 9: 子进程编排 runner（父进程 + worker 入口）

**Files:**
- Create: `backend/eval/runner.py`
- 端到端手动验证（依赖真实模型，跑真实评估）

`runner.py` 一个文件两用：父进程 `main()` 逐配置起子进程；子进程 `_worker()` 在注入的 env 下跑全量数据集、把每题结果写 json。父进程读回 json、算指标、出报告。先做纯函数 `score_results`（可测），再做编排（手动验证）。

- [ ] **Step 1: 写 score_results 的失败测试**

```python
# backend/tests/test_eval_metrics.py 末尾追加
from eval.runner import score_results
from eval.schema import EvalCase


def test_score_results_combines_metrics_and_cost():
    cases = [
        EvalCase(id="qa-001", question="q1", type="knowledge_qa",
                 expected_doc="A.md", answer_assertions={"must_include": ["550"]}),
        EvalCase(id="qa-008", question="q2", type="knowledge_qa",
                 expected_doc=None, answer_assertions={}, should_refuse=True),
    ]
    raw = [
        {"id": "qa-001", "answer": "上限550元", "ranked_filenames": ["A.md", "B.md"],
         "tokens": 1000, "latency_s": 2.0, "trace_agents": []},
        {"id": "qa-008", "answer": "知识库没有依据", "ranked_filenames": ["X.md"],
         "tokens": 1400, "latency_s": 3.0, "trace_agents": []},
    ]
    metrics, cost = score_results(cases, raw)
    assert metrics["recall@3"] == 1.0          # 唯一有 expected 的题命中
    assert metrics["assert_pass_rate"] == 1.0  # 两题断言都过（qa-008 空断言=过）
    assert cost["avg_tokens"] == 1200.0
    assert cost["avg_latency_s"] == 2.5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py::test_score_results_combines_metrics_and_cost -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'eval.runner'`）

- [ ] **Step 3: 写 runner.py**

```python
# backend/eval/runner.py
"""评估编排：父进程逐配置起子进程，子进程跑全量数据集，父进程聚合出报告。

用法（backend 目录，需 seed_corpus 已灌库、真实模型在跑）：
    .\.venv\Scripts\python.exe -m eval.runner
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from eval.config import CONFIG_MATRIX
from eval.judges.assertion_judge import check_assertions
from eval.metrics import recall_at_k, mrr, aggregate
from eval.report import render_summary, render_cost
from eval.schema import load_cases

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "datasets" / "retrieval.jsonl"
REPORTS = EVAL_DIR / "reports"


def score_results(cases: list, raw: list) -> tuple:
    """把子进程产出的每题原始结果，算成 (聚合指标 dict, 成本 dict)。"""
    by_id = {r["id"]: r for r in raw}
    per_case = []
    for c in cases:
        r = by_id.get(c.id, {})
        ranked = r.get("ranked_filenames", [])
        per_case.append({
            "recall@3": recall_at_k(ranked, c.expected_doc, k=3),
            "mrr": mrr(ranked, c.expected_doc),
            "assert_pass": check_assertions(r.get("answer", ""), c.answer_assertions),
        })
    metrics = aggregate(per_case)
    toks = [r.get("tokens", 0) for r in raw] or [0]
    lats = [r.get("latency_s", 0.0) for r in raw] or [0.0]
    cost = {"avg_tokens": sum(toks) / len(toks), "avg_latency_s": sum(lats) / len(lats)}
    return metrics, cost


async def _worker(out_path: str):
    """子进程：按当前 env 编译被测系统，跑全量数据集，写 json。"""
    from eval.system_under_test import run_dataset
    cases = load_cases(DATASET)
    raw = await run_dataset(cases)
    Path(out_path).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def _run_config_subprocess(cfg, tmp_dir: Path) -> list:
    """父进程：起一个注入 env 的子进程跑某配置，读回原始结果。"""
    out_path = tmp_dir / f"{cfg.name}.json"
    env = {**os.environ, **cfg.env}
    proc = subprocess.run(
        [sys.executable, "-m", "eval.runner", "--worker", "--out", str(out_path)],
        env=env, cwd=str(EVAL_DIR.parent),   # cwd=backend
    )
    if proc.returncode != 0:
        raise RuntimeError(f"配置 {cfg.name} 子进程失败，returncode={proc.returncode}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def main():
    cases = load_cases(DATASET)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPORTS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_matrix, cost_matrix, details = {}, {}, {}
    for cfg in CONFIG_MATRIX:
        print(f"[runner] 跑配置 {cfg.name} env={cfg.env}")
        raw = _run_config_subprocess(cfg, out_dir)
        metrics, cost = score_results(cases, raw)
        metrics_matrix[cfg.name] = metrics
        cost_matrix[cfg.name] = cost
        details[cfg.name] = raw

    (out_dir / "summary.md").write_text(render_summary(metrics_matrix), encoding="utf-8")
    (out_dir / "cost.md").write_text(render_cost(cost_matrix), encoding="utf-8")
    (out_dir / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[runner] 完成。报告在 {out_dir}")
    print(render_summary(metrics_matrix))
    print(render_cost(cost_matrix))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="子进程模式：跑全量数据集吐 json")
    ap.add_argument("--out", help="worker 输出 json 路径")
    args = ap.parse_args()
    if args.worker:
        asyncio.run(_worker(args.out))
    else:
        main()
```

- [ ] **Step 4: 跑 score_results 测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py -v`
Expected: PASS（含新增的 test_score_results...）

- [ ] **Step 5: 端到端真实跑（需 seed_corpus 已灌、DeepSeek+千问 key 在位）**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.runner`
Expected: 控制台打印三配置对比表与成本表，`backend/eval/reports/<timestamp>/` 下生成 `summary.md` / `cost.md` / `details.json`。检查 summary.md：baseline / +critic / +hyde 三行齐全，recall@3 与断言通过率为合理数值（非全 0）。

- [ ] **Step 6: 全量评估器单测回归**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_*.py tests/test_rag_hyde_toggle.py -q`
Expected: 全部 PASS。

---

## 自检（计划 vs 设计 P0）

- **数据集结构化** → Task 1（schema + retrieval.jsonl，8 条种子 + 扩充指引）。✅
- **seed_corpus 幂等灌库** → Task 8。✅
- **子进程 runner + 配置矩阵** → Task 5（矩阵）+ Task 9（子进程编排）。✅
- **检索层指标（recall@k/MRR）** → Task 3 + Task 9 score_results。✅
- **回答层事实断言** → Task 2 + Task 9。✅
- **检索层适配点** → 复用 `done.citations`（Task 7 parse_events），无需改 app（比设计更省）。✅
- **RAG_HYDE_ENABLE 开关** → Task 6。✅
- **summary.md + cost.md** → Task 4 + Task 9。✅

**与 P0 范围差异**：设计 P0 提到「改 `get_documents_for_agent` 暴露完整召回列表」——实现中发现 `done.citations` 已含 top-3 filename+排名，足够 P0 的 recall@3/MRR，故**不改 app**，更低风险。若 P1 需要 k>3，再扩 citations 或加检索 probe。

**类型一致性核对**：`EvalCase` 字段（id/question/type/expected_doc/answer_assertions/should_refuse/history）在 schema/sut/runner 各处引用一致；`parse_events` 输出键（answer/ranked_filenames/tokens/trace_agents）与 `run_one`(+latency_s)、`score_results` 消费一致；指标键（recall@3/mrr/assert_pass(_rate)）跨 metrics/runner/report 一致。✅
