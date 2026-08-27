# RAG 评估闭环自动化 P2 实现计划（阈值扫描 + 回归门禁）

> **For agentic workers（codex 执行）：** 逐任务实现，`- [ ]` 跟踪。**执行者：codex；验收：Claude。**
>
> **关联**：`RAG_GAP_THRESHOLD` 默认 0.75 是当初肉眼拍的。本计划①扫该阈值出"缺口精确/召回 vs 阈值"曲线，把"凭感觉拍"变"用数据选"；②把关键指标基线写成可重复跑的回归断言，防后续改动悄悄退步。这是从"会评估"到"评估驱动迭代"（LLMOps）的升级。
>
> **版本控制说明**：实现产物需 commit；`docs/plans/`、报告产物不 commit；`git add <具体文件>`。

**Goal:** 给 RAG 评估加两件事——阈值扫描脚本（数据驱动选 `RAG_GAP_THRESHOLD`）+ 关键指标回归断言（改动后一键自查不退步）。

**Architecture:** 阈值扫描复用现有子进程注入机制（`RAG_GAP_THRESHOLD` 是 import-time 常量,子进程设 env 后 import 即读到）+ `system_under_test.run_dataset` + `metrics.gap_precision_recall`,只跑 `routing.jsonl`（缺口指标只需它,比全集快）。回归门禁让 `runner` 额外吐结构化 `metrics.json`,新增 pytest 读 `baselines.json` 断言不退步。

**Tech Stack:** Python、子进程 env 注入、`schema`/`metrics`/`system_under_test`（稳定接口）、pytest。需 DeepSeek+千问 key、seed 已灌。

---

## 文件结构

```
backend/eval/sweep_gap_threshold.py    # 新增：阈值扫描
backend/eval/runner.py                 # 改：额外输出结构化 metrics.json
backend/eval/baselines.json            # 新增：关键指标基线（回归门禁基准）
backend/tests/test_eval_regression.py  # 新增：回归断言
```

---

### Task 1: 阈值扫描脚本

**Files:** Create `backend/eval/sweep_gap_threshold.py`

- [ ] **Step 1: 写脚本**

```python
# backend/eval/sweep_gap_threshold.py
"""阈值扫描：扫 RAG_GAP_THRESHOLD，看缺口精确/召回随阈值变化，把"肉眼拍 0.75"
变成"数据选阈值"。只跑 routing.jsonl（缺口指标只需它）。

用法（backend 目录，需 DeepSeek+千问 key、seed 已灌）：
    .\.venv\Scripts\python.exe -m eval.sweep_gap_threshold
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.metrics import gap_precision_recall
from eval.schema import load_cases

EVAL_DIR = Path(__file__).resolve().parent
ROUTING = EVAL_DIR / "datasets" / "routing.jsonl"
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


async def _worker(out_path: str):
    from eval.system_under_test import run_dataset
    raw = await run_dataset(load_cases(ROUTING))
    Path(out_path).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def _run_threshold(thr: float, tmp: Path) -> list:
    out = tmp / f"thr_{thr}.json"
    env = {**os.environ, "AGENT_ENGINE": "graph", "AGENT_CRITIC_ENABLE": "true",
           "RAG_HYDE_ENABLE": "false", "RAG_GAP_THRESHOLD": str(thr)}
    proc = subprocess.run(
        [sys.executable, "-m", "eval.sweep_gap_threshold", "--worker", "--out", str(out)],
        env=env, cwd=str(EVAL_DIR.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"阈值 {thr} 子进程失败 returncode={proc.returncode}")
    return json.loads(out.read_text(encoding="utf-8"))


def main():
    cases = load_cases(ROUTING)
    tmp = EVAL_DIR / "reports" / "_sweep"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    print("| RAG_GAP_THRESHOLD | 缺口精确率 | 缺口召回率 |")
    print("|---|---|---|")
    for thr in THRESHOLDS:
        raw = _run_threshold(thr, tmp)
        by_id = {r["id"]: r for r in raw}
        pairs = [(by_id.get(c.id, {}).get("gap_triggered", False), c.expect_gap_triggered)
                 for c in cases]
        gp, gr = gap_precision_recall(pairs)
        fp = "—" if gp is None else round(gp, 3)
        fr = "—" if gr is None else round(gr, 3)
        print(f"| {thr} | {fp} | {fr} |")
        rows.append({"threshold": thr, "gap_precision": gp, "gap_recall": gr})
    (tmp / "sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[sweep] 明细写入 {tmp / 'sweep.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.worker:
        asyncio.run(_worker(args.out))
    else:
        main()
```

- [ ] **Step 2: 跑扫描**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.sweep_gap_threshold`
Expected: 打印 6 行阈值 × (缺口精确率, 缺口召回率) 表。**观察曲线**：阈值越高越严 → 精确率通常升、召回率降；找精确/召回平衡点。

- [ ] **Step 3: 据曲线确定阈值并记录决策**

在 `docs/`（不 commit）记一句结论，如"阈值 0.75 时精确 0.87/召回 0.72 为最佳平衡；0.80 精确升但召回掉到 X，故维持 0.75"——这就是"用数据定参"的证据。如曲线显示更优值，更新 `rag_service.py` 的 `RAG_GAP_THRESHOLD` 默认或 `.env`。

- [ ] **Step 4: 提交**

```bash
git add backend/eval/sweep_gap_threshold.py
git commit -m "feat: 新增 RAG_GAP_THRESHOLD 阈值扫描（数据驱动选阈值）"
```

---

### Task 2: runner 输出结构化 metrics.json（供回归门禁读）

现在 runner 只出 `summary.md`（markdown,不好机器断言）。加一份结构化 `metrics.json`。

**Files:** Modify `backend/eval/runner.py`

- [ ] **Step 1: 在 runner 写报告处追加 metrics.json**

定位 runner `main()` 里写 `summary.md` 的地方（`(out_dir / "summary.md").write_text(...)` 附近），在其后追加：

```python
        # 结构化指标，供回归门禁机器读取（每配置的 mean）
        metrics_json = {
            name: {k: (v.get("mean") if isinstance(v, dict) else v)
                   for k, v in m.items()}
            for name, m in metrics_matrix.items()
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")
```

> codex 注意：`metrics_matrix` 的确切变量名/结构以 runner 现有代码为准（它是"配置名 → 指标 dict（含 mean/std）"）。若结构不同,据实调整提取 mean 的方式,保证 `metrics.json` 是 `{配置: {指标: 数值}}`。

- [ ] **Step 2: 跑一次确认 metrics.json 生成**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.runner`
Expected: `reports/<timestamp>/` 下新增 `metrics.json`,内容是各配置的关键指标数值。

- [ ] **Step 3: 提交**

```bash
git add backend/eval/runner.py
git commit -m "feat: 评估 runner 额外输出结构化 metrics.json"
```

---

### Task 3: 关键指标回归门禁

把当前 +critic 配置的关键指标作为基线,断言后续不退步（容差 0.05）。

**Files:** Create `backend/eval/baselines.json` + `backend/tests/test_eval_regression.py`

- [ ] **Step 1: 用最近一次评估的 +critic 指标填基线**

`backend/eval/baselines.json`（数值取自最近 `summary.md` 的 +critic 行,下面是当前值,codex 据最新报告核对）：

```json
{
  "+critic": {
    "recall@1": 0.881,
    "assert_pass_rate": 0.99,
    "route_accuracy": 0.719,
    "gap_precision": 0.866,
    "coverage": 0.693,
    "faithfulness": 0.648
  }
}
```

- [ ] **Step 2: 写回归断言**

```python
# backend/tests/test_eval_regression.py
"""评估回归门禁：最新一次评估的 +critic 关键指标不得比基线退步超过容差。
手动在改动后跑：先 `python -m eval.runner` 生成新报告，再跑本测试。
（不在 CI 自动跑——评估依赖真实 LLM/中间件,CI 跑不起；本测试是本地自查门禁。）
"""
import json
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parent.parent / "eval"
TOL = 0.05  # 容差：允许小幅波动,超过即视为退步


def _latest_metrics():
    reports = EVAL / "reports"
    dirs = sorted([d for d in reports.iterdir() if (d / "metrics.json").exists()],
                  key=lambda d: d.name)
    if not dirs:
        pytest.skip("无 metrics.json 报告,先跑 eval.runner")
    return json.loads((dirs[-1] / "metrics.json").read_text(encoding="utf-8"))


def test_critic_metrics_no_regression():
    baseline = json.loads((EVAL / "baselines.json").read_text(encoding="utf-8"))["+critic"]
    latest = _latest_metrics().get("+critic", {})
    regressed = []
    for key, base in baseline.items():
        cur = latest.get(key)
        if cur is None:
            continue
        if cur < base - TOL:
            regressed.append(f"{key}: {cur:.3f} < 基线 {base:.3f} - {TOL}")
    assert not regressed, "指标退步：\n" + "\n".join(regressed)
```

- [ ] **Step 3: 跑门禁（需先有一次 runner 报告）**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_regression.py -v`
Expected: PASS（最新报告指标 ≥ 基线 - 容差）；若某指标退步,FAIL 并列出。

- [ ] **Step 4: 提交**

```bash
git add backend/eval/baselines.json backend/tests/test_eval_regression.py
git commit -m "feat: 新增评估关键指标回归门禁（基线断言不退步）"
```

---

## 自检

- 阈值扫描 → Task 1（复用子进程注入 + run_dataset + gap_precision_recall）。✅
- 数据驱动选阈值 → Task 1 Step 3 记录决策。✅
- 回归门禁 → Task 2（结构化 metrics.json）+ Task 3（基线断言）。✅
- 诚实边界：CI 自动跑真实评估不现实（key/慢/中间件）,门禁定位为"本地改动后自查",计划已注明。✅

**Claude 验收要点**：①实跑 `sweep_gap_threshold` 看是否真出 6 行阈值曲线、缺口精确/召回随阈值单调变化合理；②确认 `metrics.json` 生成且结构正确；③故意把某指标基线调高、跑门禁应 FAIL（验证门禁有效）。**面试价值**："RAG_GAP_THRESHOLD 不是拍的,是扫出来的"——这句话能直接回击 HR 的"阈值肉眼拍"批判。
