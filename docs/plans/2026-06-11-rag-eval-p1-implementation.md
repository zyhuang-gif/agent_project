# RAG/Agent 评估体系 P1 实现计划（恢复有效性 + 编排层量化）

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 复选框跟踪。
>
> **关联**：[设计文档](../design/2026-06-11-rag-agent-evaluation-design.md)、[P0 实现计划](./2026-06-11-rag-eval-p0-implementation.md)（本计划在 P0 管线上扩展）。
>
> **版本控制**：实现代码**需要 commit**（中文 conventional 风格）；`docs/` 文档与 `backend/eval/reports/` **不要 commit**。每个 Task 测试通过后 commit 该 Task 的代码文件，用 `git add <具体文件>`，**禁止 `git add -A` / `git add .`**。

**Goal:** 解决 P0 验收暴露的两个评估有效性问题——①检索指标天花板（三配置全 1.0、失去区分度）②小样本指标抖动（同配置两次跑 ±0.125）——并新增**编排层量化**（coordinator 路由准确率 + 知识缺口触发精确/召回），让"critic 有效""缺口闭环有效"这两个差异化故事拿到稳定证据。

**Architecture:** 在 P0 的子进程 runner 上扩展：①数据集加难题 + 新增 routing 标注；②给 `GraphRunner` done 帧暴露 `plan`，让评估能拿到路由判定；③`parse_events` 提取路由与缺口信号；④新增编排层指标；⑤runner 支持同配置多跑取均值±标准差；⑥报告增强。

**Tech Stack:** 同 P0。本计划**不引入 LLM-judge**（开放式输出质量评估留 P2），全部指标走确定性信号（检索命中、路由分类、缺口触发），稳定可重复。

**范围取舍（重要）**：本计划聚焦"恢复有效性 + 编排层"。设计文档 P1 里的**开放式 LLM-judge（rubric 覆盖率）、grounding faithfulness、阈值扫描、rerank bypass、人工校准**留到 **P2 另一份计划**——它们是相对独立、依赖 LLM 调用的子系统，单独成篇更清晰。

---

## 文件结构

```
backend/eval/schema.py              # 修改：EvalCase 加 expected_route / expect_gap_triggered
backend/eval/datasets/retrieval.jsonl  # 修改：加难题（易混淆/模糊），由你对照语料补充
backend/eval/datasets/routing.jsonl    # 新建：路由 + 缺口期望标注（本计划给全 12 条）
backend/app/agent/graph/runner.py   # 修改：done 帧暴露 plan
backend/eval/system_under_test.py   # 修改：parse_events 提取 route / gap_triggered
backend/eval/metrics.py             # 修改：route_accuracy / gap_precision_recall
backend/eval/runner.py              # 修改：多跑取均值，编排指标纳入 score_results
backend/eval/report.py              # 修改：编排列 + mean±std
backend/tests/                      # 修改/新增对应单测
```

---

### Task 1: 数据集扩展（schema 字段 + routing.jsonl + 难题）

**Files:**
- Modify: `backend/eval/schema.py`
- Create: `backend/eval/datasets/routing.jsonl`
- Modify: `backend/eval/datasets/retrieval.jsonl`（加难题）
- Test: `backend/tests/test_eval_schema.py`（追加用例）

- [ ] **Step 1: 追加失败测试**

```python
# backend/tests/test_eval_schema.py 末尾追加
def test_load_cases_parses_routing_fields(tmp_path):
    import json
    from eval.schema import load_cases
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({
        "id": "rt-001", "question": "对比2023和2025版报销差异",
        "type": "document_compare", "expected_route": "document_compare",
        "expect_gap_triggered": False,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    c = load_cases(p)[0]
    assert c.expected_route == "document_compare"
    assert c.expect_gap_triggered is False


def test_routing_dataset_loads():
    from pathlib import Path
    from eval.schema import load_cases
    cases = load_cases(Path(__file__).parent.parent / "eval" / "datasets" / "routing.jsonl")
    assert len(cases) >= 10
    assert all(c.expected_route for c in cases)
    # 至少覆盖 5 类路由
    assert {c.expected_route for c in cases} >= {
        "knowledge_qa", "document_compare", "report_generation",
        "document_generation", "knowledge_gap"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_schema.py -q --basetemp=.\.pytmp`
Expected: FAIL（`AttributeError: ... 'expected_route'` 或 routing.jsonl 不存在）

- [ ] **Step 3: 给 EvalCase 加两个字段**

在 `backend/eval/schema.py` 的 `EvalCase` 里，`history` 字段之后追加：

```python
    expected_route: Optional[str] = None          # coordinator 预期路由
    expect_gap_triggered: Optional[bool] = None    # 是否预期触发知识缺口
```

并在 `load_cases` 的 `EvalCase(...)` 构造里追加两行：

```python
            expected_route=raw.get("expected_route"),
            expect_gap_triggered=raw.get("expect_gap_triggered"),
```

- [ ] **Step 4: 新建 routing.jsonl**

`backend/eval/datasets/routing.jsonl`，每行一题。下列 12 条取自 `docs/test-corpus/README.md` 的 5 类清单，路由与缺口期望明确：

```jsonl
{"id":"rt-001","question":"一线城市出差住宿费每晚上限是多少？","type":"knowledge_qa","expected_route":"knowledge_qa","expect_gap_triggered":false}
{"id":"rt-002","question":"工龄满12年每年有几天年假？","type":"knowledge_qa","expected_route":"knowledge_qa","expect_gap_triggered":false}
{"id":"rt-003","question":"对比一下差旅报销办法2023版和2025版有哪些差异？","type":"document_compare","expected_route":"document_compare","expect_gap_triggered":false}
{"id":"rt-004","question":"新版报销制度在审批额度和报销时限上比旧版有什么变化？","type":"document_compare","expected_route":"document_compare","expect_gap_triggered":false}
{"id":"rt-005","question":"根据考勤与请假制度，生成一份请假类型与待遇的要点报告。","type":"report_generation","expected_route":"report_generation","expect_gap_triggered":false}
{"id":"rt-006","question":"帮我整理一份公司各类假期的对照报告。","type":"report_generation","expected_route":"report_generation","expect_gap_triggered":false}
{"id":"rt-007","question":"帮我写一份远程办公申请，事由是家中老人需要照顾，时间下周一到周三。","type":"document_generation","expected_route":"document_generation","expect_gap_triggered":false}
{"id":"rt-008","question":"帮我起草一份病假申请，因感冒发烧请假两天。","type":"document_generation","expected_route":"document_generation","expect_gap_triggered":false}
{"id":"rt-009","question":"公司的员工持股/股权激励计划细则是怎样的？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
{"id":"rt-010","question":"境外出差的签证费和国际机票怎么报销？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
{"id":"rt-011","question":"公司有没有补充商业医疗保险？保额多少？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
{"id":"rt-012","question":"陪产假和育儿假各有几天？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
```

- [ ] **Step 5: 给 retrieval.jsonl 加难题（对照语料核对，恢复检索区分度）**

P0 数据集 8 题太直白导致 recall/MRR 天花板。**追加 ~10 条更难的题**，重点制造区分度。下面给 3 条模板 + 一条易混淆样例；**其余请你打开 `docs/test-corpus/` 对照正文补充并核对 `expected_doc` 与 `answer_assertions`**（这部分需领域核对，不要让模型凭空编数值）：

```jsonl
{"id":"qa-009","question":"出差到二线城市住一晚最多能报多少钱？","type":"knowledge_qa","expected_doc":"02-差旅与报销管理办法-2025版.md","answer_assertions":{"must_include":[]},"should_refuse":false}
{"id":"qa-010","question":"加班调休最晚要在多久内用完？","type":"knowledge_qa","expected_doc":"03-考勤与请假管理制度.md","answer_assertions":{"must_include":[]},"should_refuse":false}
{"id":"qa-011","question":"员工离职时领用的办公设备要怎么处理？","type":"knowledge_qa","expected_doc":"06-IT设备与办公用品管理规定.md","answer_assertions":{"must_include":[]},"should_refuse":false}
```

> 难题设计要点：①用口语化/同义改写而非照抄标题（考验检索鲁棒性）；②引入易混淆项（如只问"住宿上限"不指版本，看是否答 2025 的 550 而非 2023 的 450）；③`must_include` 填你在语料里核对到的真实数值/关键词。**填完把每条的 must_include 补上**（模板里留空是占位，正式跑前必须补全，否则断言恒过、失去意义）。

- [ ] **Step 6: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_schema.py -q --basetemp=.\.pytmp`
Expected: PASS

- [ ] **Step 7: commit**

```bash
git add backend/eval/schema.py backend/eval/datasets/routing.jsonl backend/eval/datasets/retrieval.jsonl backend/tests/test_eval_schema.py
git commit -m "feat: 评估数据集加路由标注与难题，schema 支持编排层字段"
```

---

### Task 2: GraphRunner done 帧暴露 plan（改 app）

**Files:**
- Modify: `backend/app/agent/graph/runner.py`
- Test: `backend/tests/test_graph_runner_plan.py`

`coordinator` 把路由写进 `state.plan`（`{task_type, need_retrieval, reason}`），但 `GraphRunner` 的 `done` 帧没暴露它。加一个 `plan` 字段（向后兼容，前端忽略未知字段）。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_graph_runner_plan.py
import pytest
import app.agent.graph.nodes.coordinator as co
import app.agent.graph.nodes.knowledge as kn
import app.agent.graph.nodes.finalize as fz
from app.agent.graph.nodes.coordinator import CoordinatorPlan
from app.agent.graph.runner import GraphRunner


class _Raw:
    usage_metadata = {"total_tokens": 10}


class _StructFake:
    def __init__(self, parsed): self._p = parsed
    async def ainvoke(self, messages):
        return {"raw": _Raw(), "parsed": self._p, "parsing_error": None}


class _ModelFake:
    def __init__(self, parsed): self._p = parsed
    def with_structured_output(self, schema, include_raw=False):
        return _StructFake(self._p)


class _FinalizeMsg:
    content = "回答"


class _FinalizeModel:
    async def ainvoke(self, messages): return _FinalizeMsg()


@pytest.mark.asyncio
async def test_done_frame_exposes_plan_task_type(monkeypatch):
    monkeypatch.setenv("AGENT_CRITIC_ENABLE", "false")
    monkeypatch.setattr(co, "chat_model", _ModelFake(
        CoordinatorPlan(task_type="document_compare", need_retrieval=True, reason="对比")))
    monkeypatch.setattr(fz, "chat_model", _FinalizeModel())

    async def _fake_get(query, filter_meta=None):
        return {"documents": ["x"], "citations": [{"filename": "a.md", "score": 0.9}],
                "is_enough": True, "max_score": 0.9}
    monkeypatch.setattr(kn.rag_service, "get_documents_for_agent", _fake_get)

    runner = GraphRunner()
    events = [e async for e in runner.stream("对比2023和2025版", history=[], identity=None)]
    done = next(e for e in reversed(events) if e["type"] == "done")
    assert done.get("plan", {}).get("task_type") == "document_compare"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_graph_runner_plan.py -q --basetemp=.\.pytmp`
Expected: FAIL（`done["plan"]` 缺失 → AttributeError/KeyError 或断言失败）

- [ ] **Step 3: 改 runner.py**

在 `GraphRunner.stream` 里，找到初始化 `final_trace: list = []`（约第 61 行）那一段，紧随其后加一行：

```python
        final_plan: dict = {}
```

在 `values` 模式收集块里（`if payload.get("trace") is not None:` 那几行附近，约第 74-75 行）追加：

```python
                    if payload.get("plan") is not None:
                        final_plan = payload["plan"]
```

最后把 `done` 帧（约第 127 行）从：

```python
        yield {"type": "done", "steps": trace_steps, "tokens": final_tokens, "citations": final_citations}
```

改为：

```python
        yield {"type": "done", "steps": trace_steps, "tokens": final_tokens,
               "citations": final_citations, "plan": final_plan}
```

- [ ] **Step 4: 跑测试 + 回归确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_graph_runner_plan.py tests/ -k "graph" -q --basetemp=.\.pytmp`
Expected: PASS（新测试过 + 现有 graph 测试不回归）

- [ ] **Step 5: commit**

```bash
git add backend/app/agent/graph/runner.py backend/tests/test_graph_runner_plan.py
git commit -m "feat(graph): done 帧暴露 coordinator plan，供评估读路由"
```

---

### Task 3: parse_events 提取路由与缺口信号

**Files:**
- Modify: `backend/eval/system_under_test.py`
- Test: `backend/tests/test_eval_sut.py`（追加用例）

- [ ] **Step 1: 追加失败测试**

```python
# backend/tests/test_eval_sut.py 末尾追加
def test_parse_events_extracts_route_and_gap():
    from eval.system_under_test import parse_events
    events = [
        {"type": "token", "data": "已记录缺口"},
        {"type": "done", "steps": [{"agent": "coordinator"}, {"agent": "knowledge"},
                                   {"agent": "knowledge_gap"}],
         "tokens": 100, "citations": [], "plan": {"task_type": "knowledge_gap"}},
    ]
    r = parse_events(events)
    assert r["route"] == "knowledge_gap"
    assert r["gap_triggered"] is True


def test_parse_events_no_gap_when_absent():
    from eval.system_under_test import parse_events
    events = [{"type": "done", "steps": [{"agent": "coordinator"}, {"agent": "knowledge"},
                                         {"agent": "finalize"}],
              "tokens": 50, "citations": [], "plan": {"task_type": "knowledge_qa"}}]
    r = parse_events(events)
    assert r["route"] == "knowledge_qa"
    assert r["gap_triggered"] is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py -q --basetemp=.\.pytmp`
Expected: FAIL（`KeyError: 'route'`）

- [ ] **Step 3: 改 parse_events**

在 `backend/eval/system_under_test.py` 的 `parse_events` 里，`done = next(...)` 之后、`return` 的字典里追加 `route` 与 `gap_triggered`：

```python
def parse_events(events: list) -> dict:
    """把 GraphRunner.stream 的事件流解析成结构化结果。"""
    answer = "".join(e["data"] for e in events if e.get("type") == "token")
    done = next((e for e in reversed(events) if e.get("type") == "done"), {})
    citations = done.get("citations", []) or []
    trace_agents = [s.get("agent") for s in (done.get("steps", []) or [])]
    return {
        "answer": answer,
        "ranked_filenames": [c.get("filename") for c in citations],
        "tokens": done.get("tokens", 0) or 0,
        "trace_agents": trace_agents,
        "route": (done.get("plan") or {}).get("task_type"),
        "gap_triggered": "knowledge_gap" in trace_agents,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py -q --basetemp=.\.pytmp`
Expected: PASS

- [ ] **Step 5: commit**

```bash
git add backend/eval/system_under_test.py backend/tests/test_eval_sut.py
git commit -m "feat: 评估解析路由与知识缺口触发信号"
```

---

### Task 4: 编排层指标（路由准确率 + 缺口 P/R）

**Files:**
- Modify: `backend/eval/metrics.py`
- Test: `backend/tests/test_eval_metrics.py`（追加用例）

- [ ] **Step 1: 追加失败测试**

```python
# backend/tests/test_eval_metrics.py 末尾追加
def test_route_accuracy():
    from eval.metrics import route_accuracy
    pairs = [("knowledge_qa", "knowledge_qa"), ("document_compare", "knowledge_qa"),
             ("knowledge_gap", "knowledge_gap")]
    # (predicted, expected)
    assert route_accuracy(pairs) == 2 / 3


def test_route_accuracy_ignores_none_expected():
    from eval.metrics import route_accuracy
    assert route_accuracy([("knowledge_qa", None), ("a", "a")]) == 1.0  # 只算有 expected 的


def test_gap_precision_recall():
    from eval.metrics import gap_precision_recall
    # (predicted_triggered, expected_triggered)
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    p, r = gap_precision_recall(pairs)
    assert p == 0.5   # TP=1, FP=1
    assert r == 0.5   # TP=1, FN=1


def test_gap_pr_no_expected_returns_none():
    from eval.metrics import gap_precision_recall
    p, r = gap_precision_recall([(False, None), (True, None)])
    assert p is None and r is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py -q --basetemp=.\.pytmp`
Expected: FAIL（`ImportError: cannot import name 'route_accuracy'`）

- [ ] **Step 3: 给 metrics.py 追加函数**

```python
# backend/eval/metrics.py 末尾追加
def route_accuracy(pairs: list) -> Optional[float]:
    """pairs: [(predicted_route, expected_route)]。只统计 expected 非 None 的题。"""
    judged = [(p, e) for p, e in pairs if e is not None]
    if not judged:
        return None
    return sum(1 for p, e in judged if p == e) / len(judged)


def gap_precision_recall(pairs: list) -> tuple:
    """pairs: [(predicted_triggered: bool, expected_triggered: bool|None)]。
    只统计 expected 非 None 的题；返回 (precision, recall)，无样本时 (None, None)。"""
    judged = [(bool(p), bool(e)) for p, e in pairs if e is not None]
    if not judged:
        return None, None
    tp = sum(1 for p, e in judged if p and e)
    fp = sum(1 for p, e in judged if p and not e)
    fn = sum(1 for p, e in judged if not p and e)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return precision, recall
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py -q --basetemp=.\.pytmp`
Expected: PASS

- [ ] **Step 5: commit**

```bash
git add backend/eval/metrics.py backend/tests/test_eval_metrics.py
git commit -m "feat: 评估新增路由准确率与缺口精确召回指标"
```

---

### Task 5: runner 多跑取均值 + 编排指标接入

**Files:**
- Modify: `backend/eval/runner.py`
- Test: `backend/tests/test_eval_metrics.py`（追加 score_results 用例）

把 `score_results` 升级为：①接收**多次**运行的原始结果（list of runs），对每个指标算 mean±std，压住 LLM 抖动；②纳入编排层指标。`_worker` 用同一份 `routing.jsonl` + `retrieval.jsonl` 合并跑；runner 主流程每配置跑 `EVAL_REPEAT`（默认 2）次。

- [ ] **Step 1: 追加失败测试**

```python
# backend/tests/test_eval_metrics.py 末尾追加
def test_score_runs_aggregates_mean_std():
    from eval.runner import score_runs
    from eval.schema import EvalCase
    cases = [EvalCase(id="qa-001", question="q", type="knowledge_qa",
                      expected_doc="A.md", answer_assertions={"must_include": ["550"]},
                      expected_route="knowledge_qa", expect_gap_triggered=False)]
    # 两次运行：第一次断言过，第二次断言不过 → assert_pass_rate mean=0.5
    run1 = [{"id": "qa-001", "answer": "550", "ranked_filenames": ["A.md"], "tokens": 100,
             "latency_s": 1.0, "route": "knowledge_qa", "gap_triggered": False}]
    run2 = [{"id": "qa-001", "answer": "无", "ranked_filenames": ["A.md"], "tokens": 200,
             "latency_s": 3.0, "route": "knowledge_qa", "gap_triggered": False}]
    metrics, cost = score_runs(cases, [run1, run2])
    assert metrics["recall@3"]["mean"] == 1.0
    assert metrics["assert_pass_rate"]["mean"] == 0.5
    assert metrics["assert_pass_rate"]["std"] == 0.5     # |0.5-1| 与 |0.5-0|
    assert metrics["route_accuracy"]["mean"] == 1.0
    assert cost["avg_tokens"]["mean"] == 150.0
    assert cost["avg_latency_s"]["mean"] == 2.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py::test_score_runs_aggregates_mean_std -q --basetemp=.\.pytmp`
Expected: FAIL（`ImportError: cannot import name 'score_runs'`）

- [ ] **Step 3: 改 runner.py**

在 `backend/eval/runner.py` 顶部 import 区追加：

```python
import statistics
from eval.metrics import route_accuracy, gap_precision_recall
```

把 `DATASET = ...` 那行下面追加 routing 数据集路径：

```python
ROUTING = EVAL_DIR / "datasets" / "routing.jsonl"
EVAL_REPEAT = int(os.getenv("EVAL_REPEAT", "2"))
```

新增 `_score_single`（单次运行算每题指标 + 单次聚合）与 `score_runs`（多次取 mean±std），替换原 `score_results`：

```python
def _mean_std(values: list) -> dict:
    nums = [v for v in values if v is not None]
    if not nums:
        return {"mean": None, "std": None}
    return {"mean": statistics.fmean(nums),
            "std": statistics.pstdev(nums) if len(nums) > 1 else 0.0}


def _score_single(cases: list, raw: list) -> dict:
    """单次运行 → 该次的聚合指标（标量）。"""
    by_id = {r["id"]: r for r in raw}
    per_case = []
    route_pairs, gap_pairs = [], []
    for c in cases:
        r = by_id.get(c.id, {})
        ranked = r.get("ranked_filenames", [])
        per_case.append({
            "recall@3": recall_at_k(ranked, c.expected_doc, k=3),
            "mrr": mrr(ranked, c.expected_doc),
            "assert_pass": check_assertions(r.get("answer", ""), c.answer_assertions),
        })
        route_pairs.append((r.get("route"), c.expected_route))
        gap_pairs.append((r.get("gap_triggered", False), c.expect_gap_triggered))
    agg = aggregate(per_case)
    gp, gr = gap_precision_recall(gap_pairs)
    agg["route_accuracy"] = route_accuracy(route_pairs)
    agg["gap_precision"] = gp
    agg["gap_recall"] = gr
    agg["avg_tokens"] = statistics.fmean([r.get("tokens", 0) for r in raw]) if raw else 0.0
    agg["avg_latency_s"] = statistics.fmean([r.get("latency_s", 0.0) for r in raw]) if raw else 0.0
    return agg


def score_runs(cases: list, runs: list) -> tuple:
    """runs: [single_raw, ...]（同配置多次）→ (指标 mean±std dict, 成本 mean±std dict)。"""
    singles = [_score_single(cases, raw) for raw in runs]
    metric_keys = ["recall@3", "mrr", "assert_pass_rate",
                   "route_accuracy", "gap_precision", "gap_recall"]
    metrics = {k: _mean_std([s.get(k) for s in singles]) for k in metric_keys}
    metrics["n"] = singles[0]["n"] if singles else 0
    metrics["repeat"] = len(runs)
    cost = {
        "avg_tokens": _mean_std([s["avg_tokens"] for s in singles]),
        "avg_latency_s": _mean_std([s["avg_latency_s"] for s in singles]),
    }
    return metrics, cost
```

把 `_worker` 改成合并加载两个数据集：

```python
async def _worker(out_path: str):
    from eval.system_under_test import run_dataset
    cases = load_cases(DATASET) + load_cases(ROUTING)
    raw = await run_dataset(cases)
    Path(out_path).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
```

把 `_run_config_subprocess` 改成跑 `EVAL_REPEAT` 次、返回多次结果：

```python
def _run_config_subprocess(cfg, tmp_dir: Path) -> list:
    env = {**os.environ, **cfg.env}
    runs = []
    for i in range(EVAL_REPEAT):
        out_path = tmp_dir / f"{cfg.name}.run{i}.json"
        proc = subprocess.run(
            [sys.executable, "-m", "eval.runner", "--worker", "--out", str(out_path)],
            env=env, cwd=str(EVAL_DIR.parent))
        if proc.returncode != 0:
            raise RuntimeError(f"配置 {cfg.name} 第 {i} 次子进程失败")
        runs.append(json.loads(out_path.read_text(encoding="utf-8")))
    return runs
```

把 `main()` 里加载与打分改成：

```python
def main():
    cases = load_cases(DATASET) + load_cases(ROUTING)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPORTS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_matrix, cost_matrix, details = {}, {}, {}
    for cfg in CONFIG_MATRIX:
        print(f"[runner] 跑配置 {cfg.name} ×{EVAL_REPEAT} env={cfg.env}")
        runs = _run_config_subprocess(cfg, out_dir)
        metrics, cost = score_runs(cases, runs)
        metrics_matrix[cfg.name] = metrics
        cost_matrix[cfg.name] = cost
        details[cfg.name] = runs

    (out_dir / "summary.md").write_text(render_summary(metrics_matrix), encoding="utf-8")
    (out_dir / "cost.md").write_text(render_cost(cost_matrix), encoding="utf-8")
    (out_dir / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[runner] 完成。报告在 {out_dir}")
    print(render_summary(metrics_matrix))
    print(render_cost(cost_matrix))
```

> 注意：原 `score_results` 已被 `score_runs` + `_score_single` 取代。删除旧 `score_results` 函数及 P0 里 `test_score_results_combines_metrics_and_cost` 那个测试（它的契约已变）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py -q --basetemp=.\.pytmp`
Expected: PASS（含 score_runs 新测试；旧 score_results 测试已删除）

- [ ] **Step 5: commit**

```bash
git add backend/eval/runner.py backend/tests/test_eval_metrics.py
git commit -m "feat: 评估支持多跑取均值并接入编排层指标"
```

---

### Task 6: 报告增强（编排列 + mean±std）

**Files:**
- Modify: `backend/eval/report.py`
- Test: `backend/tests/test_eval_report.py`（替换用例，适配新结构）

- [ ] **Step 1: 改测试以匹配新结构**

```python
# backend/tests/test_eval_report.py 全文替换
from eval.report import render_summary, render_cost


def _ms(mean, std):
    return {"mean": mean, "std": std}


def test_summary_has_orchestration_columns_and_meanstd():
    matrix = {
        "baseline": {"n": 20, "repeat": 2,
                     "recall@3": _ms(0.9, 0.0), "mrr": _ms(0.85, 0.02),
                     "assert_pass_rate": _ms(0.8, 0.1),
                     "route_accuracy": _ms(0.95, 0.0),
                     "gap_precision": _ms(1.0, 0.0), "gap_recall": _ms(0.75, 0.0)},
        "+critic": {"n": 20, "repeat": 2,
                    "recall@3": _ms(0.9, 0.0), "mrr": _ms(0.9, 0.0),
                    "assert_pass_rate": _ms(0.95, 0.05),
                    "route_accuracy": _ms(0.95, 0.0),
                    "gap_precision": _ms(1.0, 0.0), "gap_recall": _ms(1.0, 0.0)},
    }
    md = render_summary(matrix)
    assert "路由准确率" in md
    assert "缺口" in md
    assert "0.900±0.000" in md       # mean±std 格式
    assert "baseline" in md and "+critic" in md


def test_cost_table_meanstd():
    cost = {"baseline": {"avg_tokens": _ms(1200.0, 50.0), "avg_latency_s": _ms(3.4, 0.2)}}
    md = render_cost(cost)
    assert "1200.000±50.000" in md or "1200.0" in md
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_report.py -q --basetemp=.\.pytmp`
Expected: FAIL

- [ ] **Step 3: 改 report.py**

```python
# backend/eval/report.py 全文替换
def _ms(d) -> str:
    """格式化 {mean, std} 为 'mean±std'；None 给 —。"""
    if not isinstance(d, dict):
        return "—" if d is None else str(d)
    mean, std = d.get("mean"), d.get("std")
    if mean is None:
        return "—"
    return f"{mean:.3f}±{(std or 0.0):.3f}"


def render_summary(matrix: dict) -> str:
    lines = [
        "# 评估对比表（检索 + 回答 + 编排，mean±std）",
        "",
        "| 配置 | n | 重复 | recall@3 | MRR | 事实断言 | 路由准确率 | 缺口精确率 | 缺口召回率 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in matrix.items():
        lines.append(
            f"| {name} | {m.get('n','—')} | {m.get('repeat','—')} | "
            f"{_ms(m.get('recall@3'))} | {_ms(m.get('mrr'))} | {_ms(m.get('assert_pass_rate'))} | "
            f"{_ms(m.get('route_accuracy'))} | {_ms(m.get('gap_precision'))} | {_ms(m.get('gap_recall'))} |"
        )
    lines.append("")
    lines.append("> 样本量小（指示性，非统计显著）。mean±std 为同配置多次运行的均值与总体标准差；— 表示不适用。")
    return "\n".join(lines)


def render_cost(cost: dict) -> str:
    lines = [
        "# 成本对比表（延迟 / token，mean±std）",
        "",
        "| 配置 | 平均 token | 平均延迟秒 |",
        "|---|---|---|",
    ]
    for name, c in cost.items():
        lines.append(f"| {name} | {_ms(c.get('avg_tokens'))} | {_ms(c.get('avg_latency_s'))} |")
    return "\n".join(lines)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_report.py -q --basetemp=.\.pytmp`
Expected: PASS

- [ ] **Step 5: 全量评估器单测回归**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_*.py tests/test_rag_hyde_toggle.py tests/test_graph_runner_plan.py -q --basetemp=.\.pytmp`
Expected: 全部 PASS

- [ ] **Step 6: commit**

```bash
git add backend/eval/report.py backend/tests/test_eval_report.py
git commit -m "feat: 评估报告增加编排层指标与 mean±std"
```

---

### Task 7: 端到端真实跑 + 验证（需环境）

**Files:** 无（运行验证）

需 MySQL/Redis/向量库 + DeepSeek/千问 key 在跑，且 `seed_corpus` 已灌库（P1 没改语料，沿用即可）。

- [ ] **Step 1: 跑评估**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.runner`
Expected: 控制台打印新表（含路由准确率/缺口精确召回列、mean±std），`backend/eval/reports/<新时间戳>/` 生成 summary.md / cost.md / details.json。

- [ ] **Step 2: 验证恢复有效性**

检查 summary.md：
- 加难题后，检索层 recall@3 / MRR **不再全是 1.000**（出现配置间差异即说明区分度恢复）；
- assert_pass_rate 带上了 std（量化了抖动）；
- 新增的**路由准确率**列有数值（coordinator 分类质量）；
- **缺口精确/召回**列有数值（rt-009~012 应触发缺口 → 召回应较高）。

- [ ] **Step 3: 清理临时目录**

Run: `cd backend; Remove-Item -Recurse -Force .\.pytmp -ErrorAction SilentlyContinue`

---

## 自检（计划 vs 目标）

- **天花板效应** → Task 1 Step 5 加难题（恢复检索区分度）。✅
- **指标抖动** → Task 5 多跑取均值 + std。✅
- **编排层量化（路由准确率）** → Task 2（暴露 plan）+ Task 3（解析 route）+ Task 4（route_accuracy）+ Task 6（报告列）。✅
- **编排层量化（缺口 P/R）** → Task 3（gap_triggered，现成 trace 信号）+ Task 4（gap_precision_recall）+ Task 6。✅
- **routing 数据** → Task 1 给全 12 条。✅

**类型一致性核对**：`parse_events` 新增 `route`/`gap_triggered` → `_score_single` 消费 `r.get("route")`/`r.get("gap_triggered")`；`EvalCase.expected_route`/`expect_gap_triggered` → `_score_single` 的 `route_pairs`/`gap_pairs`；`score_runs` 输出 `{mean,std}` → `report._ms` 消费。键名 `route_accuracy`/`gap_precision`/`gap_recall` 跨 metrics/runner/report 一致。✅

**改 app 范围**：仅 `backend/app/agent/graph/runner.py` 一处（done 帧加 plan 字段，向后兼容）。

**留给 P2 的（设计文档 P1 余项）**：开放式 LLM-judge（qwen3-max，rubric 覆盖率）、grounding faithfulness、阈值扫描（RAG_GAP_THRESHOLD 0.6~0.85）、rerank bypass 开关、LLM-judge 人工校准一致性。
