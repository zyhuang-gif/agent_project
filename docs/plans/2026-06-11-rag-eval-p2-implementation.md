# RAG/Agent 评估体系 P2 实现计划（开放式质量评估：LLM-judge + grounding）

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 复选框跟踪。
>
> **关联**：[设计文档](../design/2026-06-11-rag-agent-evaluation-design.md)、[P0](./2026-06-11-rag-eval-p0-implementation.md)、[P1](./2026-06-11-rag-eval-p1-implementation.md)。本计划在 P0/P1 管线上扩展。
>
> **版本控制**：实现代码**需要 commit**（中文 conventional）；`docs/` 与 `backend/eval/reports/` **不要 commit**。每个 Task 测试通过后 `git add <具体文件>` 提交，**禁止 `git add -A` / `git add .`**。
>
> **前置**：建议先完成「快速增强」（recall@1 + gitignore）再做本计划，避免改 report/metrics 冲突。

**Goal:** 让评估覆盖**开放式任务质量**——对比/报告/申请这三类 P0/P1 只评了"路由对不对"、没评"内容好不好"的任务，用 **LLM-judge（qwen3-max，与被测 DeepSeek 不同家族）按 rubric 逐点核对覆盖率**评质量，并加 **grounding faithfulness**（回答是否有文档依据），最后用**人工校准**证明 judge 可信。

**Architecture:** 新增 `tasks.jsonl`（带 rubric_points）；`eval/judges/llm_judge.py` 构造独立的 qwen3-max judge（temperature=0、结构化输出、逐点核对）；`grounding_judge.py` 判忠实度；runner 对开放式任务跑被测→喂 judge→算覆盖率/faithfulness；报告加列；`calibrate.py` 离线算 judge 与人工的一致性。

**Tech Stack:** 同 P1 + `langchain_community` 的 `ChatTongyi`（qwen3-max）、pydantic 结构化输出。judge 真实调用在 Task 7 验证；单测全程 mock judge 模型。

**范围**：本计划聚焦开放式质量评估（LLM-judge + grounding + 校准）。设计文档剩余的**阈值扫描（RAG_GAP_THRESHOLD 0.6~0.85）、rerank bypass 开关**列为末尾「P3 可选」，不在本计划展开。

---

## 文件结构

```
backend/eval/schema.py                 # 修改：EvalCase 加 rubric_points
backend/eval/datasets/tasks.jsonl      # 新建：对比/报告/申请，带 rubric_points
backend/eval/judges/llm_judge.py       # 新建：qwen3-max judge + rubric 覆盖率
backend/eval/judges/grounding_judge.py # 新建：faithfulness
backend/eval/judge_model.py            # 新建：judge 模型工厂（独立可配）
backend/eval/runner.py                 # 修改：开放式任务接入 + 覆盖率/faithfulness 指标
backend/eval/report.py                 # 修改：加开放式覆盖率/faithfulness 列
backend/eval/calibrate.py              # 新建：judge vs 人工 一致性
backend/tests/                         # 新增/改对应单测
```

---

### Task 1: schema 加 rubric_points + tasks.jsonl

**Files:** Modify `backend/eval/schema.py`；Create `backend/eval/datasets/tasks.jsonl`；Test `backend/tests/test_eval_schema.py`

- [ ] **Step 1: 追加失败测试**

```python
# backend/tests/test_eval_schema.py 末尾追加
def test_tasks_dataset_has_rubric_points():
    from pathlib import Path
    from eval.schema import load_cases
    cases = load_cases(Path(__file__).parent.parent / "eval" / "datasets" / "tasks.jsonl")
    assert len(cases) >= 4
    assert all(c.rubric_points for c in cases)            # 每条都有评分点
    assert {c.type for c in cases} >= {"document_compare", "report_generation",
                                       "document_generation"}
```

- [ ] **Step 2: 跑确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_schema.py -q --basetemp=.\.pytmp`
Expected: FAIL（`AttributeError: 'rubric_points'` 或文件不存在）

- [ ] **Step 3: EvalCase 加字段**

`schema.py` 的 `EvalCase` 追加：
```python
    rubric_points: list = field(default_factory=list)   # 开放式输出的逐点评分要点
```
`load_cases` 构造里追加：
```python
            rubric_points=raw.get("rubric_points", []),
```

- [ ] **Step 4: 新建 tasks.jsonl**

`backend/eval/datasets/tasks.jsonl`（rubric_points 取自 README 对比清单与制度要点；申请类按制度约束列要点）：
```jsonl
{"id":"cmp-001","question":"对比差旅报销办法2023版和2025版的差异","type":"document_compare","rubric_points":["市内交通80→120","住宿一线450→550","审批额度5000→8000","报销时限30→60天","打款5→3工作日"],"expected_docs":["01-差旅与报销管理办法-2023版.md","02-差旅与报销管理办法-2025版.md"]}
{"id":"cmp-002","question":"新版报销制度在审批额度和报销时限上比旧版有什么变化","type":"document_compare","rubric_points":["审批额度5000→8000","报销时限30→60天"],"expected_docs":["01-差旅与报销管理办法-2023版.md","02-差旅与报销管理办法-2025版.md"]}
{"id":"rpt-001","question":"根据考勤与请假制度，生成一份请假类型与待遇的要点报告","type":"report_generation","rubric_points":["年假","病假","事假","加班/调休","各类假期的待遇或天数"],"expected_docs":["03-考勤与请假管理制度.md"]}
{"id":"gen-001","question":"帮我写一份远程办公申请，事由是照顾老人，时间下周一到周三","type":"document_generation","rubric_points":["申请人/事由明确","起止时间下周一到周三","符合每周≤2天限制或说明","格式为可提交的申请正文"],"expected_docs":["04-远程办公管理规定.md"]}
```

> 注意 gen-001 的 rubric "每周≤2天"是制度硬约束（远程办公一周最多 2 天），但申请写的是周一到周三共 3 天——这是**故意的冲突题**，考验模型是否会指出超限。judge 按"符合限制或明确说明"评，二者满足其一即覆盖。

- [ ] **Step 5: 跑确认通过 + commit**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_schema.py -q --basetemp=.\.pytmp` → PASS
```bash
git add backend/eval/schema.py backend/eval/datasets/tasks.jsonl backend/tests/test_eval_schema.py
git commit -m "feat: 评估新增开放式任务数据集与 rubric_points"
```

---

### Task 2: judge 模型工厂（独立 qwen3-max）

**Files:** Create `backend/eval/judge_model.py`；Test `backend/tests/test_eval_judge_model.py`

judge 必须与被测（DeepSeek）**不同家族**——用阿里云千问 `qwen3-max`，复用现有阿里云 key。模型构造与被测的全局 `chat_model` 解耦，单独在此创建。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_judge_model.py
import os
from eval.judge_model import judge_model_name, build_judge_model


def test_default_judge_is_qwen3_max(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    assert judge_model_name() == "qwen3-max"


def test_judge_model_override(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "qwen-plus")
    assert judge_model_name() == "qwen-plus"


def test_build_judge_model_is_tongyi(monkeypatch):
    monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "sk-test")
    monkeypatch.setenv("ALIYUN_BASE_URL", "https://example/v1")
    m = build_judge_model()
    # ChatTongyi 实例，模型名与 temperature 正确
    assert m.model_name == "qwen3-max" or getattr(m, "model", None) == "qwen3-max"
```

- [ ] **Step 2: 跑确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_judge_model.py -q --basetemp=.\.pytmp`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 judge_model.py**

```python
# backend/eval/judge_model.py
"""LLM-judge 模型：独立用阿里云千问 qwen3-max（与被测 DeepSeek 不同家族，规避自评偏袒）。"""
import os
from langchain_community.chat_models.tongyi import ChatTongyi


def judge_model_name() -> str:
    return os.getenv("EVAL_JUDGE_MODEL", "qwen3-max")


def build_judge_model():
    """构造 judge 模型；temperature=0 降波动。"""
    return ChatTongyi(
        model=judge_model_name(),
        api_key=os.getenv("ALIYUN_JUDGE_API_KEY") or os.getenv("ALIYUN_ACCESS_KEY_SECRET"),
        base_url=os.getenv("ALIYUN_BASE_URL"),
        temperature=0,
    )
```

- [ ] **Step 4: 跑确认通过 + commit**

Run: `...pytest tests/test_eval_judge_model.py -q --basetemp=.\.pytmp` → PASS
```bash
git add backend/eval/judge_model.py backend/tests/test_eval_judge_model.py
git commit -m "feat: 评估新增独立 qwen3-max judge 模型工厂"
```

---

### Task 3: LLM-judge rubric 覆盖率

**Files:** Create `backend/eval/judges/llm_judge.py`；Test `backend/tests/test_eval_llm_judge.py`

把"打分"拆成：纯函数 `coverage_ratio`（可测）+ 真实调用 `judge_coverage`（注入 model，测试用 fake）。结构化输出强制 judge 返回"覆盖了哪几点"。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_llm_judge.py
import pytest
from eval.judges.llm_judge import coverage_ratio, judge_coverage, CoverageVerdict


def test_coverage_ratio():
    assert coverage_ratio(["a", "b"], total=4) == 0.5
    assert coverage_ratio([], total=3) == 0.0
    assert coverage_ratio(["a"], total=0) is None   # 无 rubric 不适用


class _FakeStruct:
    def __init__(self, verdict): self._v = verdict
    async def ainvoke(self, messages): return self._v


class _FakeJudge:
    def __init__(self, verdict): self._v = verdict
    def with_structured_output(self, schema): return _FakeStruct(self._v)


@pytest.mark.asyncio
async def test_judge_coverage_counts_covered():
    verdict = CoverageVerdict(covered_points=["审批额度5000→8000", "报销时限30→60天"],
                              reasoning="覆盖两点")
    model = _FakeJudge(verdict)
    ratio = await judge_coverage(model, "对比问题", "回答文本",
                                 ["审批额度5000→8000", "报销时限30→60天", "打款5→3工作日"])
    assert ratio == pytest.approx(2 / 3)
```

- [ ] **Step 2: 跑确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_llm_judge.py -q --basetemp=.\.pytmp`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 llm_judge.py**

```python
# backend/eval/judges/llm_judge.py
"""LLM-judge：开放式输出按 rubric_points 逐点核对覆盖率。逐点核对而非整体打分，降 judge 波动。"""
from typing import Optional
from pydantic import BaseModel


class CoverageVerdict(BaseModel):
    covered_points: list[str]   # 回答确实覆盖到的 rubric 要点（原样回填）
    reasoning: str


def coverage_ratio(covered_points: list, total: int) -> Optional[float]:
    """覆盖率 = 覆盖点数 / rubric 总点数。total=0（无 rubric）返回 None。"""
    if not total:
        return None
    return len(covered_points or []) / total


_SYSTEM = (
    "你是严格的评审。给定一个问题、一段回答、和一组评分要点(rubric)。"
    "逐条判断回答是否实质覆盖了每个要点，只把确实覆盖到的要点原样列入 covered_points。"
    "不要臆测、不要把未覆盖的算作覆盖。"
)


async def judge_coverage(model, question: str, answer: str, rubric_points: list) -> Optional[float]:
    """用 judge 模型核对覆盖率。model 由调用方注入（生产用 build_judge_model()）。"""
    if not rubric_points:
        return None
    user = (f"问题：{question}\n\n回答：{answer}\n\n"
            f"评分要点：\n" + "\n".join(f"- {p}" for p in rubric_points))
    structured = model.with_structured_output(CoverageVerdict)
    verdict = await structured.ainvoke(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}])
    # 只统计确实属于 rubric 的覆盖点（防 judge 编造点）
    covered = [p for p in verdict.covered_points if p in rubric_points]
    return coverage_ratio(covered, total=len(rubric_points))
```

- [ ] **Step 4: 跑确认通过 + commit**

Run: `...pytest tests/test_eval_llm_judge.py -q --basetemp=.\.pytmp` → PASS
```bash
git add backend/eval/judges/llm_judge.py backend/tests/test_eval_llm_judge.py
git commit -m "feat: 评估新增 LLM-judge rubric 覆盖率裁判"
```

---

### Task 4: grounding faithfulness

**Files:** Create `backend/eval/judges/grounding_judge.py`；Test `backend/tests/test_eval_grounding.py`

faithfulness = 回答中**有文档依据的事实陈述占比**。judge 返回"无依据的陈述"，纯函数算分。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_grounding.py
import pytest
from eval.judges.grounding_judge import faithfulness_score, judge_faithfulness, GroundingVerdict


def test_faithfulness_score():
    assert faithfulness_score(total_claims=4, unsupported=1) == 0.75
    assert faithfulness_score(total_claims=0, unsupported=0) is None  # 无可判 → 不适用


class _FakeStruct:
    def __init__(self, v): self._v = v
    async def ainvoke(self, m): return self._v


class _FakeJudge:
    def __init__(self, v): self._v = v
    def with_structured_output(self, s): return _FakeStruct(self._v)


@pytest.mark.asyncio
async def test_judge_faithfulness():
    v = GroundingVerdict(total_claims=4, unsupported_claims=["编造的第5条规定"], reasoning="一条无依据")
    score = await judge_faithfulness(_FakeJudge(v), "回答", ["文档片段1", "文档片段2"])
    assert score == 0.75
```

- [ ] **Step 2: 跑确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_grounding.py -q --basetemp=.\.pytmp`
Expected: FAIL

- [ ] **Step 3: 写 grounding_judge.py**

```python
# backend/eval/judges/grounding_judge.py
"""grounding faithfulness：回答的事实陈述是否都能在检索文档里找到依据。"""
from typing import Optional
from pydantic import BaseModel


class GroundingVerdict(BaseModel):
    total_claims: int                 # 回答里可核查的事实性陈述数
    unsupported_claims: list[str]     # 其中在文档里找不到依据的
    reasoning: str


def faithfulness_score(total_claims: int, unsupported: int) -> Optional[float]:
    if not total_claims:
        return None
    return max(0.0, (total_claims - unsupported) / total_claims)


_SYSTEM = (
    "你是事实核查员。给定一段回答和若干文档片段。先数出回答中可核查的事实性陈述总数，"
    "再列出其中无法在文档片段中找到依据的陈述。不要把常识或措辞性语句算作事实陈述。"
)


async def judge_faithfulness(model, answer: str, documents: list) -> Optional[float]:
    if not (answer or "").strip():
        return None
    ctx = "\n\n".join(f"【片段{i}】{d}" for i, d in enumerate(documents or [], 1)) or "（无检索文档）"
    user = f"回答：{answer}\n\n文档片段：\n{ctx}"
    structured = model.with_structured_output(GroundingVerdict)
    v = await structured.ainvoke(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}])
    return faithfulness_score(v.total_claims, len(v.unsupported_claims or []))
```

- [ ] **Step 4: 跑确认通过 + commit**

Run: `...pytest tests/test_eval_grounding.py -q --basetemp=.\.pytmp` → PASS
```bash
git add backend/eval/judges/grounding_judge.py backend/tests/test_eval_grounding.py
git commit -m "feat: 评估新增 grounding faithfulness 裁判"
```

---

### Task 5: 开放式任务接入 runner + 指标

**Files:** Modify `backend/eval/system_under_test.py`、`backend/eval/runner.py`；Test `backend/tests/test_eval_metrics.py`

被测 `run_one` 已能拿 `answer` 与检索文档名；开放式任务还需 `documents`（grounding 用）。`parse_events` 已有 citations，但 grounding 需要文档正文——P2 用 `chunk_preview`（citations 里已有）近似。runner 对 `tasks.jsonl` 的题，跑被测→喂两个 judge→把覆盖率/faithfulness 纳入 `_score_single`。

- [ ] **Step 1: parse_events 暴露文档预览（追加测试）**

```python
# backend/tests/test_eval_sut.py 末尾追加
def test_parse_events_exposes_doc_previews():
    from eval.system_under_test import parse_events
    events = [{"type": "done", "steps": [], "tokens": 1, "plan": {"task_type": "document_compare"},
               "citations": [{"filename": "a.md", "chunk_preview": "片段A", "score": 0.9}]}]
    r = parse_events(events)
    assert r["doc_previews"] == ["片段A"]
```

- [ ] **Step 2: 跑确认失败 → 改 parse_events**

`system_under_test.py` 的 `parse_events` 返回字典追加：
```python
        "doc_previews": [c.get("chunk_preview", "") for c in citations],
```

- [ ] **Step 3: judge 打分接入（runner 异步打分 + score）**

在 `runner.py` 顶部追加：
```python
from eval.judge_model import build_judge_model
from eval.judges.llm_judge import judge_coverage
from eval.judges.grounding_judge import judge_faithfulness
```

新增异步函数 `judge_open_tasks`（对开放式题逐条调 judge，把覆盖率/faithfulness 回填进 raw 项）：
```python
async def judge_open_tasks(cases: list, raw: list) -> None:
    """就地给 raw 里属于开放式任务（有 rubric_points）的题补 coverage / faithfulness。"""
    by_id = {c.id: c for c in cases}
    model = build_judge_model()
    for r in raw:
        c = by_id.get(r["id"])
        if not c or not c.rubric_points:
            continue
        r["coverage"] = await judge_coverage(model, c.question, r.get("answer", ""), c.rubric_points)
        r["faithfulness"] = await judge_faithfulness(model, r.get("answer", ""), r.get("doc_previews", []))
```

在 `_worker` 里、写 json 之前调用它：
```python
async def _worker(out_path: str):
    from eval.system_under_test import run_dataset
    cases = load_cases(DATASET) + load_cases(ROUTING) + load_cases(TASKS)
    raw = await run_dataset(cases)
    await judge_open_tasks(cases, raw)
    Path(out_path).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
```
并在路径常量区加：
```python
TASKS = EVAL_DIR / "datasets" / "tasks.jsonl"
```
`main()` 里 `cases = load_cases(DATASET) + load_cases(ROUTING)` 也改为再 `+ load_cases(TASKS)`。

- [ ] **Step 4: _score_single 纳入开放式指标（追加测试）**

```python
# backend/tests/test_eval_metrics.py 末尾追加
def test_score_single_aggregates_open_metrics():
    from eval.runner import _score_single
    from eval.schema import EvalCase
    cases = [EvalCase(id="cmp-001", question="q", type="document_compare",
                      rubric_points=["a", "b", "c", "d"])]
    raw = [{"id": "cmp-001", "answer": "x", "ranked_filenames": [], "tokens": 0,
            "latency_s": 0.0, "route": "document_compare", "gap_triggered": False,
            "coverage": 0.5, "faithfulness": 1.0}]
    agg = _score_single(cases, raw)
    assert agg["coverage"] == 0.5
    assert agg["faithfulness"] == 1.0
```

在 `_score_single` 里，`aggregate(per_case)` 之后追加（对有值的题取均值）：
```python
    covs = [r.get("coverage") for r in raw if r.get("coverage") is not None]
    faiths = [r.get("faithfulness") for r in raw if r.get("faithfulness") is not None]
    agg["coverage"] = statistics.fmean(covs) if covs else None
    agg["faithfulness"] = statistics.fmean(faiths) if faiths else None
```
并在 `score_runs` 的 `metric_keys` 末尾加 `"coverage", "faithfulness"`。

- [ ] **Step 5: 跑确认通过 + commit**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py tests/test_eval_metrics.py -q --basetemp=.\.pytmp` → PASS
```bash
git add backend/eval/system_under_test.py backend/eval/runner.py backend/tests/test_eval_sut.py backend/tests/test_eval_metrics.py
git commit -m "feat: 评估接入开放式任务的 LLM-judge 覆盖率与 faithfulness"
```

---

### Task 6: 报告加开放式列

**Files:** Modify `backend/eval/report.py`；Test `backend/tests/test_eval_report.py`

- [ ] **Step 1: 追加断言**：在 `render_summary` 测试的 matrix 每个配置加 `"coverage": _ms(0.8,0.0), "faithfulness": _ms(0.9,0.0)`，断言 `"rubric覆盖率" in md` 与 `"faithfulness" in md or "忠实度" in md`。

- [ ] **Step 2: 跑确认失败 → 改 render_summary**：表头与每行末尾追加两列：
```python
        # 表头追加： rubric覆盖率 | 忠实度
        # 每行追加： {_ms(m.get('coverage'))} | {_ms(m.get('faithfulness'))}
```
（在现有列后插入 `| rubric覆盖率 | 忠实度 |` 表头与对应分隔符，行尾追加 `f"{_ms(m.get('coverage'))} | {_ms(m.get('faithfulness'))} |"`。）

- [ ] **Step 3: 跑确认通过 + commit**

```bash
git add backend/eval/report.py backend/tests/test_eval_report.py
git commit -m "feat: 评估报告增加 rubric 覆盖率与忠实度列"
```

---

### Task 7: 人工校准（judge 可信度证明）

**Files:** Create `backend/eval/calibrate.py`；Test `backend/tests/test_eval_calibrate.py`

抽若干题，对比 judge 覆盖率与人工标注，算一致性（agreement）。纯聚合逻辑可测；真实跑读人工标注文件。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_calibrate.py
from eval.calibrate import agreement


def test_agreement_within_tolerance():
    # (judge_score, human_score) 对，容差 0.25 内算一致
    pairs = [(1.0, 1.0), (0.5, 0.6), (0.0, 0.5)]   # 第三对差 0.5 不一致
    assert agreement(pairs, tol=0.25) == 2 / 3
```

- [ ] **Step 2: 跑确认失败 → 写 calibrate.py**

```python
# backend/eval/calibrate.py
"""LLM-judge 人工校准：算 judge 与人工标注的一致率，证明 judge 可信。

人工标注放 backend/eval/datasets/calibration.jsonl，每行：
  {"id": "cmp-001", "human_coverage": 0.8}
用法：先跑一次 eval.runner 拿 judge 分，再人工标 10 条，跑本脚本算 agreement。
"""
from typing import Optional


def agreement(pairs: list, tol: float = 0.25) -> Optional[float]:
    """pairs: [(judge_score, human_score)]；两者差 ≤ tol 视为一致。返回一致率。"""
    judged = [(j, h) for j, h in pairs if j is not None and h is not None]
    if not judged:
        return None
    return sum(1 for j, h in judged if abs(j - h) <= tol) / len(judged)
```

- [ ] **Step 3: 跑确认通过 + commit**

```bash
git add backend/eval/calibrate.py backend/tests/test_eval_calibrate.py
git commit -m "feat: 评估新增 LLM-judge 人工校准一致性"
```

---

### Task 8: 端到端真实跑 + 验证（需环境）

需 MySQL/Redis/向量库 + DeepSeek key（被测）+ 阿里云千问 key（judge）。

- [ ] **Step 1**: `cd backend; .\.venv\Scripts\python.exe -m eval.runner`
- [ ] **Step 2 验证**: 新 summary.md 出现 **rubric覆盖率 / 忠实度** 列且有数值；开放式题（cmp/rpt/gen）被 judge 评分；details.json 里这些题有 `coverage`/`faithfulness` 字段。
- [ ] **Step 3 校准（可选）**: 人工标 `calibration.jsonl` 后，确认 judge 与人工 agreement ≥ 0.7（写进 README/报告作为 judge 可信度证据）。
- [ ] **Step 4**: `Remove-Item -Recurse -Force .\.pytmp -ErrorAction SilentlyContinue`

---

## 自检（计划 vs 目标）

- 开放式质量评估 → Task 1（rubric 数据）+ Task 3（覆盖率）+ Task 5（接入）+ Task 6（报告）。✅
- judge 与被测不同家族 → Task 2（qwen3-max）。✅
- faithfulness → Task 4 + Task 5。✅
- judge 可信度 → Task 7 人工校准。✅

**类型一致性**：`rubric_points`(schema) → `judge_coverage`/`_score_single`；`coverage`/`faithfulness`(judge 输出) → raw → `_score_single` → `score_runs.metric_keys` → `report._ms`；`doc_previews`(parse_events) → `judge_faithfulness`。键名一致。

**改 app 范围**：本计划**不改 app**（judge 与开放式评估全在 eval 层；被测复用 P1 已暴露的 done.plan/citations）。

**留给 P3（可选）**：阈值扫描（RAG_GAP_THRESHOLD 0.6~0.85 扫曲线，为 0.75 取值给数据支撑）、rerank bypass 开关（reorder_service 加 RERANK_ENABLE，量化 rerank 真实增益）。
```
