# RAG/Agent 评估体系 P3 实现计划（阈值标定 + rerank bypass）

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 复选框跟踪。
>
> **关联**：[设计文档](../design/2026-06-11-rag-agent-evaluation-design.md)、[P2 计划](./2026-06-11-rag-eval-p2-implementation.md)。
>
> **版本控制**：实现代码 commit（中文 conventional）；`docs/` 与 `backend/eval/reports/` 不 commit；`git add <具体文件>`，禁止 `git add -A` / `.`。

**Goal:** ①用数据给 `RAG_GAP_THRESHOLD` 重新定标——用户刚把 reranker 从 `qwen3-vl-rerank` 换成 `qwen3-rerank`，旧阈值 0.75 是按 vl 的 `relevance_score` 分布标的，新模型尺度可能不同，缺口触发会失准；②新增 `RERANK_ENABLE` 开关，量化 rerank 的真实增益。

**Architecture:** 关键洞察——`citations[0].score` 就是缺口判定用的 `max_score`，且 `is_enough = max_score >= RAG_GAP_THRESHOLD` 是纯比较。所以**阈值扫描离线做**：跑一次评估拿每题 `max_score`，再对一组阈值纯比较算缺口 P/R/F1 曲线，**不必为每个阈值重烧 LLM**。先加 rerank 健康检查（确认新模型有效 + 看分数分布），再做扫描。rerank bypass 改 `reorder_service`（Task 5，改 app）。

**Tech Stack:** 同 P2。Task 1-4 不改 app（仅 eval 层 + parse_events）；Task 5 改 `reorder_service.py`（可选）。

---

## 文件结构

```
backend/eval/rerank_healthcheck.py   # 新建：验证 reranker 模型有效 + dump 分数分布
backend/eval/system_under_test.py    # 修改：parse_events 暴露 max_score
backend/eval/threshold_sweep.py      # 新建：离线阈值扫描，出缺口 P/R/F1 曲线 + 推荐阈值
backend/app/rag/reorder_service.py   # 修改（Task 5 可选）：RERANK_ENABLE 开关
backend/eval/config.py               # 修改（Task 5 可选）：加 -rerank 配置
backend/tests/                       # 新增对应单测
```

---

### Task 1: rerank 健康检查（先排除"模型名失效→静默全缺口"）

**Files:** Create `backend/eval/rerank_healthcheck.py`；手动验证（依赖真实 DashScope）

换了模型名后，若 `qwen3-rerank` 无效，`reorder` 静默失败→所有 `similarity=0`→所有问题被判缺口。本脚本先排雷，并打印新模型的分数尺度（为阈值扫描提供直觉）。

- [ ] **Step 1: 写 rerank_healthcheck.py**

```python
# backend/eval/rerank_healthcheck.py
"""rerank 健康检查：验证当前 RERANKER 配置可用，并打印一组样例的相关度分数。

用法（backend 目录，需 DashScope key + 网络）：
    .\.venv\Scripts\python.exe -m eval.rerank_healthcheck
判读：success=True 且相关文档分数明显高于无关文档 → 模型有效；
      若 success=False 或分数全 0/全等 → 模型名失效或调用异常，需查 ALIYUN_RERANKER_MODEL_NAME。
"""
import asyncio
from app.rag.reorder_service import reorder_service

_SAMPLES = [
    ("一线城市出差住宿费上限",
     ["一线城市出差住宿费每晚上限550元", "笔记本电脑每4年更换一次", "年假按工龄计算"]),
    ("远程办公申请条件",
     ["远程办公需转正满3个月且绩效不低于B", "差旅报销时限60天", "病假需提供证明"]),
]


async def main():
    for query, docs in _SAMPLES:
        res = await reorder_service.reorder_documents(query, docs)
        print(f"\nquery={query!r} success={res['success']} error={res.get('error','')}")
        for item in res.get("documents", []):
            print(f"  {item['similarity']:.4f}  {item['document'][:30]}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 跑健康检查**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.rerank_healthcheck`
Expected: 每个 query 打印 `success=True`，且与 query 相关的文档分数明显高于无关文档。
- 若 `success=False`：`qwen3-rerank` 模型名在该账号无效 → 停下来报告，先解决模型名（本计划后续依赖 rerank 正常）。
- 记录相关文档的分数区间（如 0.6~0.9 或 0.3~0.7），Task 4 选阈值时参考。

- [ ] **Step 3: commit**

```bash
git add backend/eval/rerank_healthcheck.py
git commit -m "feat: 评估新增 reranker 健康检查脚本"
```

---

### Task 2: parse_events 暴露 max_score

**Files:** Modify `backend/eval/system_under_test.py`；Test `backend/tests/test_eval_sut.py`

`citations[0].score`（reorder 后 top-1 相关度）= 缺口判定的 `max_score`。暴露它供离线扫描。

- [ ] **Step 1: 追加失败测试**

```python
# backend/tests/test_eval_sut.py 末尾追加
def test_parse_events_exposes_max_score():
    from eval.system_under_test import parse_events
    events = [{"type": "done", "steps": [], "tokens": 1, "plan": {"task_type": "knowledge_qa"},
               "citations": [{"filename": "a.md", "score": 0.83},
                             {"filename": "b.md", "score": 0.41}]}]
    r = parse_events(events)
    assert r["max_score"] == 0.83


def test_parse_events_max_score_none_when_no_citations():
    from eval.system_under_test import parse_events
    r = parse_events([{"type": "done", "steps": [], "tokens": 1, "citations": [],
                       "plan": {"task_type": "knowledge_gap"}}])
    assert r["max_score"] is None
```

- [ ] **Step 2: 跑确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py -q --basetemp=.\.pytmp`
Expected: FAIL（`KeyError: 'max_score'`）

- [ ] **Step 3: 改 parse_events**

在 `system_under_test.py` 的 `parse_events` 返回字典里追加（`citations` 已在函数内取到）：
```python
        "max_score": max((c.get("score") for c in citations), default=None),
```

- [ ] **Step 4: 跑确认通过 + commit**

Run: `...pytest tests/test_eval_sut.py -q --basetemp=.\.pytmp` → PASS
```bash
git add backend/eval/system_under_test.py backend/tests/test_eval_sut.py
git commit -m "feat: 评估暴露检索 max_score 供阈值标定"
```

---

### Task 3: 离线阈值扫描

**Files:** Create `backend/eval/threshold_sweep.py`；Test `backend/tests/test_eval_threshold_sweep.py`

纯函数 `sweep`（可测）+ 脚本入口（跑一次评估收集 max_score → 扫阈值）。判定规则：`max_score is None or max_score < threshold ⇒ 判缺口`。用 `routing.jsonl` 的 `expect_gap_triggered` 当标尺，复用 `metrics.gap_precision_recall`。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_threshold_sweep.py
from eval.threshold_sweep import sweep, best_threshold


def test_sweep_computes_pr_f1_per_threshold():
    # (max_score, expect_gap)：低分应触发缺口
    pairs = [(0.9, False), (0.85, False), (0.4, True), (0.3, True)]
    rows = sweep(pairs, thresholds=[0.5, 0.8])
    # 阈值 0.5：predicted_gap = score<0.5 → 后两条触发，完美 → P=R=F1=1
    assert rows[0.5]["precision"] == 1.0 and rows[0.5]["recall"] == 1.0
    # 阈值 0.8：0.4/0.3 触发(对) + 无误触发 → 仍完美；0.85/0.9 不触发(对)
    assert rows[0.8]["recall"] == 1.0


def test_best_threshold_picks_max_f1():
    rows = {0.5: {"f1": 0.8}, 0.7: {"f1": 0.95}, 0.9: {"f1": 0.6}}
    assert best_threshold(rows) == 0.7
```

- [ ] **Step 2: 跑确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_threshold_sweep.py -q --basetemp=.\.pytmp`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 threshold_sweep.py**

```python
# backend/eval/threshold_sweep.py
"""离线阈值扫描：用每题的 max_score 与缺口期望，对一组 RAG_GAP_THRESHOLD 算缺口 P/R/F1，
给阈值取值（尤其换 reranker 后）提供数据支撑。

用法（backend 目录，需评估环境）：
    .\.venv\Scripts\python.exe -m eval.threshold_sweep
"""
import asyncio
from typing import Optional

from eval.metrics import gap_precision_recall
from eval.schema import load_cases
from eval.system_under_test import run_dataset
from eval.runner import ROUTING

_THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def _f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if not p or not r:
        return 0.0 if (p is not None and r is not None) else None
    return 2 * p * r / (p + r)


def sweep(pairs: list, thresholds: list) -> dict:
    """pairs: [(max_score, expect_gap)]。每个阈值算 predicted_gap=(ms is None or ms<t) 的 P/R/F1。"""
    rows = {}
    for t in thresholds:
        gap_pairs = [((ms is None or ms < t), eg) for ms, eg in pairs]
        p, r = gap_precision_recall(gap_pairs)
        rows[t] = {"precision": p, "recall": r, "f1": _f1(p, r)}
    return rows


def best_threshold(rows: dict) -> Optional[float]:
    """取 F1 最高的阈值。"""
    cand = [(t, v.get("f1")) for t, v in rows.items() if v.get("f1") is not None]
    return max(cand, key=lambda x: x[1])[0] if cand else None


def render(rows: dict) -> str:
    lines = ["# RAG_GAP_THRESHOLD 阈值扫描（缺口 P/R/F1）", "",
             "| 阈值 | 缺口精确率 | 缺口召回率 | F1 |", "|---|---|---|---|"]
    for t in sorted(rows):
        v = rows[t]
        def f(x): return "—" if x is None else f"{x:.3f}"
        lines.append(f"| {t:.2f} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} |")
    bt = best_threshold(rows)
    lines += ["", f"> 推荐阈值（F1 最高）：**{bt:.2f}**。换 reranker 后据此更新 .env 的 RAG_GAP_THRESHOLD。"]
    return "\n".join(lines)


async def main():
    cases = load_cases(ROUTING)               # routing.jsonl 带 expect_gap_triggered
    raw = await run_dataset(cases)            # 当前 env（生产配置）跑一次，拿 max_score
    by_id = {r["id"]: r for r in raw}
    pairs = [(by_id.get(c.id, {}).get("max_score"), c.expect_gap_triggered) for c in cases]
    rows = sweep(pairs, _THRESHOLDS)
    print(render(rows))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 跑确认通过 + commit**

Run: `...pytest tests/test_eval_threshold_sweep.py -q --basetemp=.\.pytmp` → PASS
```bash
git add backend/eval/threshold_sweep.py backend/tests/test_eval_threshold_sweep.py
git commit -m "feat: 评估新增离线阈值扫描"
```

---

### Task 4: 真实跑 + 用新 reranker 重标定（需环境）

需 MySQL/Redis/向量库 + DeepSeek key + 阿里云千问 key，且 `seed_corpus` 已灌库。

- [ ] **Step 1**: 先确认 Task 1 健康检查 `success=True`。
- [ ] **Step 2**: 跑扫描 `cd backend; .\.venv\Scripts\python.exe -m eval.threshold_sweep`，得到 P/R/F1 曲线表与推荐阈值。
- [ ] **Step 3**: 把 `backend/.env` 的 `RAG_GAP_THRESHOLD` 改为推荐值（若与 0.75 不同），重启后端。
- [ ] **Step 4**: 重跑 `.\.venv\Scripts\python.exe -m eval.runner`，确认新阈值下缺口精确/召回更平衡（F1 提升）。把扫描表与新阈值记进 README（"换 qwen3-rerank 后用扫描把阈值从 0.75 调到 X"是个有数据的工程决策故事）。

---

### Task 5（可选）: RERANK_ENABLE 开关，量化 rerank 增益

> **改 app**，且需处理"关 rerank 后缺口判定无分数"的问题，复杂度高于前 4 个 Task，按需做。

**Files:** Modify `backend/app/rag/reorder_service.py`、`backend/eval/config.py`；Test `backend/tests/test_rerank_toggle.py`

- [ ] **Step 1**: `reorder_service.py` 顶部加运行时开关：
```python
def _rerank_enabled() -> bool:
    return os.getenv("RERANK_ENABLE", "true").strip().lower() not in ("false", "0", "no")
```
在 `reorder_documents` 开头（取到 `documents` 后、`_get_scorer` 前）加 bypass 分支：
```python
        if not _rerank_enabled():
            # 关 rerank：保留检索原序，用按位次递减的占位相关度（让下游 max_score 判定可运行）
            n = len(documents)
            scored = [{"document": d, "similarity": (n - i) / n} for i, d in enumerate(documents)]
            return {"success": True, "documents": scored, "error": ""}
```
> 注意：bypass 时 `similarity` 是占位值（非真实相关度），故 **no-rerank 配置下缺口/置信类指标不可比，主要看 recall@k / MRR**。在报告或 README 注明这一点。

- [ ] **Step 2**: 测试（纯开关逻辑 + bypass 返回原序）：
```python
# backend/tests/test_rerank_toggle.py
import pytest
from app.rag.reorder_service import reorder_service, _rerank_enabled


def test_rerank_enabled_default_true(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLE", raising=False)
    assert _rerank_enabled() is True


@pytest.mark.asyncio
async def test_bypass_preserves_order(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLE", "false")
    res = await reorder_service.reorder_documents("q", ["doc1", "doc2", "doc3"])
    assert res["success"] is True
    assert [d["document"] for d in res["documents"]] == ["doc1", "doc2", "doc3"]
    assert res["documents"][0]["similarity"] >= res["documents"][1]["similarity"]
```

- [ ] **Step 3**: `config.py` 的 `CONFIG_MATRIX` 可加一条对照（与 +critic 同基线、关 rerank）：
```python
    EvalConfig("-rerank", {**_GRAPH, "AGENT_CRITIC_ENABLE": "true",
                           "RAG_HYDE_ENABLE": "false", "RERANK_ENABLE": "false"}),
```

- [ ] **Step 4**: 跑测试 + 回归：
Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_rerank_toggle.py tests/ -k "rag or graph or eval" -q --basetemp=.\.pytmp`
Expected: PASS
```bash
git add backend/app/rag/reorder_service.py backend/eval/config.py backend/tests/test_rerank_toggle.py
git commit -m "feat: reranker 增加 RERANK_ENABLE 开关并接入消融"
```

---

## 自检（计划 vs 目标）

- 换 reranker 后阈值重标 → Task 1（健康检查）+ Task 2（max_score）+ Task 3（扫描）+ Task 4（定标）。✅
- rerank 增益量化 → Task 5（RERANK_ENABLE，可选）。✅

**类型一致性**：`parse_events` 新增 `max_score` → `threshold_sweep.main` 消费 `r["max_score"]`；`sweep` 输出 `{precision,recall,f1}` → `render`/`best_threshold` 消费；复用 `gap_precision_recall`（P1）与 `ROUTING`（P1 runner 常量）。

**改 app 范围**：Task 1-4 不改 app；Task 5（可选）改 `reorder_service.py` 一处。

**关键判读**：Task 1 健康检查是前置闸门——若 `qwen3-rerank` 无效，先解决再继续，否则后续全部数据失真。
