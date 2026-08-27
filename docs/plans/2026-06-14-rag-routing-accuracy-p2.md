# RAG 路由准确率优化 P2 实现计划

> **For agentic workers（codex 执行）：** 逐任务实现，步骤用 `- [ ]` 复选框跟踪。**执行者：codex；验收：Claude。**
>
> **关联**：评估报告 `backend/eval/reports/<latest>/summary.md` 显示 coordinator **路由准确率仅 0.719**（三配置一致，与 critic/hyde 无关），是编排层唯一明显偏低的硬指标。本计划用「难例分析 → 针对性改 prompt → 评估验证」闭环把它提上去——这本身就是"评估驱动迭代"的完整故事。
>
> **版本控制说明**：实现产物（`analyze_routing.py`、`coordinator.py` 改动）需 commit（中文 conventional commit）。`docs/plans/` 不 commit。`git add <具体文件>`，禁止 `git add -A`。报告产物不 commit。

**Goal:** 把 RAG coordinator 的路由准确率从 0.72 提升（目标 ≥0.85），方法是先用难例分析定位易混类型，再给 coordinator prompt 加 few-shot + 边界澄清，最后用 routing 评估验证提升。

**Architecture:** 不改图结构,只优化 `coordinator_node` 的 prompt。新增 `eval/analyze_routing.py`：对 `routing.jsonl` 逐题只调 coordinator（不跑全 graph,快）分类,输出准确率 + 混淆矩阵 + 错误案例,据此定位薄弱类型。改 `_COORDINATOR_PROMPT` 后重跑分析验证。

**Tech Stack:** Python、coordinator（DeepSeek flash via `get_chat_model("coordinator")`，结构化输出 `CoordinatorPlan`）、`routing.jsonl`（32 条,5 类路由）。需 DeepSeek key 在位。

---

## 文件结构

```
backend/eval/analyze_routing.py        # 新增：路由难例分析（混淆矩阵 + 错误案例）
backend/app/agent/graph/nodes/coordinator.py   # 改：_COORDINATOR_PROMPT 加 few-shot + 边界
```

---

### Task 1: 路由难例分析脚本

只调 coordinator 对 `routing.jsonl` 逐题分类（不起全 graph,省时省钱），输出薄弱类型。

**Files:** Create `backend/eval/analyze_routing.py`

- [ ] **Step 1: 写 analyze_routing.py**

```python
# backend/eval/analyze_routing.py
"""路由难例分析：对 routing.jsonl 逐题跑 coordinator 分类，输出准确率 + 混淆矩阵
+ 错误案例，定位 coordinator 路由薄弱类型，指导 prompt 优化。

用法（backend 目录，需 DeepSeek key 在位）：
    .\.venv\Scripts\python.exe -m eval.analyze_routing
"""
import asyncio
from collections import Counter, defaultdict
from pathlib import Path

from eval.schema import load_cases
from app.agent.graph.nodes.coordinator import (
    _COORDINATOR_PROMPT, CoordinatorPlan, _plan_to_dict)
from app.utils.factory import get_chat_model

ROUTING = Path(__file__).resolve().parent / "datasets" / "routing.jsonl"


async def classify(model, question: str) -> str:
    structured = model.with_structured_output(CoordinatorPlan, include_raw=True)
    result = await structured.ainvoke([
        {"role": "system", "content": _COORDINATOR_PROMPT},
        {"role": "user", "content": f"当前问题：{question}"},
    ])
    if result.get("parsing_error"):
        return "unknown"
    return _plan_to_dict(result["parsed"])["task_type"]


async def main():
    cases = load_cases(ROUTING)
    model = get_chat_model("coordinator")
    errors, confusion = [], defaultdict(Counter)
    correct = 0
    for c in cases:
        pred = await classify(model, c.question)
        confusion[c.expected_route][pred] += 1
        if pred == c.expected_route:
            correct += 1
        else:
            errors.append({"id": c.id, "question": c.question,
                           "expected": c.expected_route, "predicted": pred})
    print(f"路由准确率：{correct}/{len(cases)} = {correct/len(cases):.3f}\n")
    print("=== 混淆矩阵（expected -> predicted 计数）===")
    for exp, preds in confusion.items():
        print(f"  {exp}: {dict(preds)}")
    print(f"\n=== 错误案例（{len(errors)} 条）===")
    for e in errors:
        print(f"  [{e['id']}] expected={e['expected']} predicted={e['predicted']}")
        print(f"       Q: {e['question']}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 跑分析,记录基线与薄弱类型**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.analyze_routing`
Expected: 打印基线准确率（应 ≈0.72）、混淆矩阵、错误案例。**记录哪些 expected 类型最常被错判成什么**（典型猜测：document_compare ↔ report_generation ↔ document_generation 互混；knowledge_qa ↔ knowledge_gap 边界不清）。

- [ ] **Step 3: 提交分析脚本**

```bash
git add backend/eval/analyze_routing.py
git commit -m "feat: 新增路由难例分析脚本（混淆矩阵+错误案例）"
```

---

### Task 2: 优化 coordinator prompt（few-shot + 边界澄清）

据 Task 1 的混淆矩阵针对性改 prompt。下面是改后模板（已覆盖最常见易混点）；codex 按 Task 1 实际错误案例**补充对应的 few-shot 示例**（错得最多的类型多给 1-2 个例子）。

**Files:** Modify `backend/app/agent/graph/nodes/coordinator.py`（仅 `_COORDINATOR_PROMPT` 字符串）

- [ ] **Step 1: 替换 `_COORDINATOR_PROMPT`**

把 `coordinator.py` 中的 `_COORDINATOR_PROMPT = """..."""` 整体替换为：

```python
_COORDINATOR_PROMPT = """你是企业知识库 Agent 的任务协调器。判断用户问题属于哪种任务类型，以及是否需要检索知识库。

只输出一个 JSON 对象，不要任何额外解释，格式：
{"task_type": "<类型>", "need_retrieval": <true|false>, "reason": "<简短中文理由>"}

task_type 取值与判定边界：
- knowledge_qa：就某个事实/政策直接提问（如"住宿上限多少""年假几天"）。
- document_compare：要求对比两份/新旧版文档的差异（出现"对比""差异""比…有什么变化"）。
- report_generation：要求生成结构化报告或要点汇总（"生成一份…报告""整理一份…对照"）。
- document_generation：要求起草可提交的申请/说明正文（"帮我写一份…申请""起草…"）。
- knowledge_gap：问题主题明显超出企业知识库范围、需记录缺口。
- unknown：完全无法识别。

关键区分（按"动作"而非"主题"判，同一主题可对应不同动作）：
- 「对比/差异」→ document_compare；「生成报告/汇总要点」→ report_generation；「起草申请/正文」→ document_generation。
- 是否 knowledge_gap：若是常见公司制度（薪酬/绩效/考勤/差旅报销/福利保险/股权激励/生育陪产假）→ 不是 gap，按上面动作归类；若明显是公司不管的外部事项（个税计算、外部签证、食堂菜单等）→ knowledge_gap。

示例：
- "一线城市住宿上限多少" → knowledge_qa
- "对比2023和2025版差旅报销的差异" → document_compare
- "整理一份公司各类假期的对照报告" → report_generation
- "帮我起草一份病假申请" → document_generation
- "差旅报销款需要缴纳个人所得税吗" → knowledge_gap

need_retrieval：除非是与企业知识完全无关的闲聊，否则一律为 true。"""
```

> codex 注意：保留 `CoordinatorPlan` 模型、`_plan_to_dict`、`coordinator_node` 不变,只动 prompt 字符串。若 Task 1 显示某类型仍频繁错判,在"示例"段再补 1-2 条该类型的真实难例（取自 Task 1 错误案例的 question）。

- [ ] **Step 2: 重跑难例分析验证提升**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.analyze_routing`
Expected: 准确率较 Task 1 基线**明显上升**（目标 ≥0.85）；混淆矩阵里原薄弱类型的错判减少。若提升不足,据新错误案例再补 few-shot 重试（最多迭代 2-3 轮）。

- [ ] **Step 3: 提交 prompt 优化**

```bash
git add backend/app/agent/graph/nodes/coordinator.py
git commit -m "feat: coordinator prompt 加 few-shot+边界澄清提升路由准确率"
```

---

### Task 3: 全量评估回归（确认路由提升、其它指标不退）

**Files:** 无新代码,端到端真实跑（需 seed 已灌、DeepSeek+千问 key、judge=qwen3.7-max）。

- [ ] **Step 1: 重跑完整评估**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.runner`
Expected: 新报告 `reports/<timestamp>/summary.md` 里 `路由准确率` 较 0.719 明显上升；`recall@1`、事实断言、缺口精确/召回、rubric/忠实度等**不出现明显退步**。

- [ ] **Step 2: 评估器单测回归**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_*.py -q`
Expected: 全绿。

---

## 自检

- 难例定位 → Task 1 混淆矩阵 + 错误案例。✅
- 针对性改进 → Task 2 few-shot + 边界澄清（动作维度区分 + gap 边界）。✅
- 验证提升 → Task 2 Step 2（快速）+ Task 3（全量,确认不退其它指标）。✅
- 不改图结构 → 只动 `_COORDINATOR_PROMPT`。✅

**Claude 验收要点**：①实跑 `analyze_routing.py` 看准确率是否真从 ≈0.72 升到目标区间；②实读新 `summary.md` 确认路由准确率上升且其它指标不退；③抽查改后 prompt 的 few-shot 是否覆盖了 Task 1 暴露的薄弱类型。**面试价值**：这份工作能讲成"用评估定位路由是弱环 → 难例分析 → 针对性优化 → 数据验证提升"的完整评估驱动闭环。
