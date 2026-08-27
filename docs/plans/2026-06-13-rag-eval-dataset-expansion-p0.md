# RAG 评估「扩样本 + 扩语料」P0 实现计划

> **For agentic workers（codex 执行）：** 逐任务实现，步骤用 `- [ ]` 复选框跟踪；每个 Task 测试通过后再做下一个。**执行者：codex；验收：Claude。**
>
> **关联**：本计划回应「招 agent HR 评审」中最致命的一条——评估被当核心亮点，却建在 tasks n=4、语料仅 6 篇、recall@3 恒=1.0 的玩具级样本上，一问即崩。本计划把数据集与语料扩到「指示性可信级」，并用质量校验测试守住红线。评估管线代码（`schema.py` / `runner.py` / `judges/` / `config.py`）**已就绪、本计划不改**，只扩数据与语料 + 加红线测试 + 重跑。
>
> **版本控制说明**：实现产物（语料 md、数据集 jsonl、新增测试）**需要 commit**（中文 conventional commit，如 `test:` / `feat:` / `docs:`）。但 `docs/plans/`、`docs/design/` 下的设计/计划文档**不要 commit**——保持 untracked。每个 Task 用 `git add <具体文件>` 精确添加，**禁止 `git add -A` / `git add .`**（否则会把 `docs/` 设计文档误提交）。
>
> ⚠️ **语料/标注一致性是本计划的生命线**：每道题的 `answer_assertions` / `rubric_points` / `expected_doc(s)` 必须与语料正文的事实**逐字对齐**。新增语料先定「关键事实点」，题目标注严格锚定这些事实点，否则评估会把"对的回答判成错"。

**Goal:** 把 RAG 评估数据集从 retrieval 18 / routing 16 / tasks 4、语料 6 篇，扩到 retrieval ≥40 / routing ≥32 / tasks ≥21、语料 ≥10 篇，让三层指标有区分度、经得起"开放任务几个样本""语料多大"的追问；并新增数据集质量校验测试守住红线。

**Architecture:** 三步走——①新增 4 篇企业制度语料（覆盖现 routing.jsonl 中股权激励/商业医保/陪产假等"知识缺口"主题，把它们从"无答案负样本"转成"有答案正样本"，一举提升语料量与样本量）；②按相同 schema 扩三个 jsonl；③新增 `test_eval_dataset_quality.py` 强制数量/字段/语料引用一致性达标。最后重新 `seed_corpus` 灌库 + 重跑 `runner` 出新报告。

**Tech Stack:** Python 3.12、pytest、jsonl 数据集、Markdown 语料。被测真实模型栈：chat=DeepSeek V4（graph 引擎分角色 flash/pro）、embedding/rerank=千问（重跑评估需真实依赖在跑）。

---

## 文件结构

```
docs/test-corpus/                              # 新增 4 篇语料（延续 01-06 命名风格）
├── 07-薪酬与绩效管理制度.md                    # 新增
├── 08-员工福利与商业保险管理办法.md             # 新增（覆盖补充商业医疗保险）
├── 09-生育与陪产假管理规定.md                  # 新增（覆盖陪产假/育儿假/产假）
└── 10-员工持股与股权激励计划.md                # 新增（覆盖股权激励）

backend/eval/datasets/
├── retrieval.jsonl    # 18 → ≥40（新语料事实题 + 更多拒答负样本）
├── routing.jsonl      # 16 → ≥32（gap 题按新语料改判 + 补新 gap 负样本）
└── tasks.jsonl        # 4  → ≥21（每类 document_compare/report_generation/document_generation ≥7）

backend/tests/
└── test_eval_dataset_quality.py               # 新增：数据集红线校验
```

每文件单一职责：语料（纯内容）/ 数据集（标注）/ 质量测试（红线）解耦。质量测试是硬约束——数量、字段完整性、`expected_doc(s)` 必须指向真实存在的语料文件、id 全局唯一。

---

### Task 1: 数据集质量校验测试（红线先行）

先写"扩充达标"的红线测试。此时数据集尚未扩充，测试**应当失败**——这定义了 Task 2~5 的验收目标。

**Files:**
- Create: `backend/tests/test_eval_dataset_quality.py`

- [ ] **Step 1: 写红线测试**

```python
# backend/tests/test_eval_dataset_quality.py
"""数据集质量红线：数量、字段完整性、语料引用一致性、id 唯一。

这是数据集扩充（Task 2~5）的验收门。codex 扩到达标后本测试全绿。
直接读 jsonl raw（不经 load_cases），以便校验 expected_docs 等字段。
"""
import json
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parent.parent / "eval" / "datasets"
CORPUS = Path(__file__).resolve().parents[2] / "docs" / "test-corpus"

RETRIEVAL = EVAL / "retrieval.jsonl"
ROUTING = EVAL / "routing.jsonl"
TASKS = EVAL / "tasks.jsonl"

ROUTE_TYPES = {"knowledge_qa", "document_compare", "report_generation",
               "document_generation", "knowledge_gap"}
TASK_TYPES = {"document_compare", "report_generation", "document_generation"}


def _load(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _corpus_files():
    return {p.name for p in CORPUS.glob("*.md") if p.name != "README.md"}


def test_corpus_has_at_least_10_docs():
    assert len(_corpus_files()) >= 10, "语料需 ≥10 篇，让 recall@k 有区分度"


def test_retrieval_size_and_fields():
    rows = _load(RETRIEVAL)
    assert len(rows) >= 40, f"retrieval 需 ≥40 条，当前 {len(rows)}"
    files = _corpus_files()
    refuse = 0
    for r in rows:
        assert r["id"] and r["question"] and r["type"]
        if r.get("should_refuse"):
            refuse += 1
            continue
        # 非拒答题：expected_doc 必须指向真实语料文件，且有事实断言
        assert r.get("expected_doc") in files, f"{r['id']} expected_doc 不在语料：{r.get('expected_doc')}"
        ai = r.get("answer_assertions") or {}
        assert ai.get("must_include"), f"{r['id']} 非拒答题须有 must_include 断言"
    assert refuse >= 6, f"拒答负样本需 ≥6，当前 {refuse}"


def test_tasks_size_and_type_balance():
    rows = _load(TASKS)
    assert len(rows) >= 21, f"tasks 需 ≥21 条（解 n=4 炸点），当前 {len(rows)}"
    files = _corpus_files()
    by_type = {t: 0 for t in TASK_TYPES}
    for r in rows:
        assert r["type"] in TASK_TYPES, f"{r['id']} type 非法：{r['type']}"
        by_type[r["type"]] += 1
        pts = r.get("rubric_points") or []
        assert len(pts) >= 3, f"{r['id']} rubric_points 需 ≥3 点"
        for d in (r.get("expected_docs") or []):
            assert d in files, f"{r['id']} expected_docs 不在语料：{d}"
    for t, n in by_type.items():
        assert n >= 7, f"开放式类型 {t} 需 ≥7 条，当前 {n}"


def test_routing_size_and_gap_balance():
    rows = _load(ROUTING)
    assert len(rows) >= 32, f"routing 需 ≥32 条，当前 {len(rows)}"
    gap = 0
    for r in rows:
        assert r["expected_route"] in ROUTE_TYPES, f"{r['id']} route 非法：{r['expected_route']}"
        if r["expected_route"] == "knowledge_gap":
            gap += 1
            assert r.get("expect_gap_triggered") is True, f"{r['id']} gap 题须 expect_gap_triggered=true"
    assert gap >= 8, f"knowledge_gap 负样本需 ≥8（缺口指标分母），当前 {gap}"


def test_ids_globally_unique():
    all_ids = [r["id"] for path in (RETRIEVAL, ROUTING, TASKS) for r in _load(path)]
    dup = {i for i in all_ids if all_ids.count(i) > 1}
    assert not dup, f"存在重复 id：{dup}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_dataset_quality.py -v`
Expected: 多条 FAIL（语料 <10、retrieval <40、tasks <21、routing gap 计数等）。这是预期，定义了扩充目标。

- [ ] **Step 3: 提交红线测试**

```bash
git add backend/tests/test_eval_dataset_quality.py
git commit -m "test: 新增 RAG 评估数据集质量红线校验"
```

---

### Task 2: 新增 4 篇语料（覆盖现有 knowledge_gap 主题）

新增语料先**钉死关键事实点**，后续题目标注全部锚定这些点。codex 按下列事实点写成与现有 01-06 同风格的制度正文（标题 + 章节 + 条款，每篇 600~1200 字）。**事实点中的数字必须原样落进正文**，因为 retrieval/tasks 的标注靠它们。

**Files:**
- Create: `docs/test-corpus/07-薪酬与绩效管理制度.md`
- Create: `docs/test-corpus/08-员工福利与商业保险管理办法.md`
- Create: `docs/test-corpus/09-生育与陪产假管理规定.md`
- Create: `docs/test-corpus/10-员工持股与股权激励计划.md`

- [ ] **Step 1: 写 `07-薪酬与绩效管理制度.md`，正文必须包含以下事实点**
  - 月薪结构：基本工资 + 绩效工资 + 岗位补贴三部分，其中**绩效工资占月薪 30%**。
  - 绩效考核分 **A/B/C/D 四档**，A 档绩效系数 1.2、B 档 1.0、C 档 0.8、D 档 0.6。
  - 年度调薪窗口为**每年 4 月**，依据上一年度绩效。
  - 全年发放 **13 薪**，第 13 薪于次年 1 月随工资发放。
  - 试用期工资为转正后基本工资的 **80%**。

- [ ] **Step 2: 写 `08-员工福利与商业保险管理办法.md`，正文必须包含**
  - 五险一金按国家规定足额缴纳。
  - 公司额外提供**补充商业医疗保险，保额 50 万元/年**，住院费用在社保报销后由商业保险**承担剩余部分的 80%**。
  - 每年提供 **1 次免费体检**。
  - 节日福利：法定节日每次 **500 元**福利。
  - 补充保险覆盖**员工本人**，可自费为配偶子女加保。

- [ ] **Step 3: 写 `09-生育与陪产假管理规定.md`，正文必须包含**
  - **产假 158 天**（含国家规定基础产假 + 地方奖励假）。
  - **陪产假 15 天**，需在配偶分娩后 3 个月内休完。
  - **育儿假每年 10 天**，子女**满 3 周岁前**夫妻双方各享。
  - 产检假：孕期每月可享 1 天带薪产检假。
  - 难产或多胞胎按规定增加产假天数。

- [ ] **Step 4: 写 `10-员工持股与股权激励计划.md`，正文必须包含**
  - 激励对象条件：**司龄满 2 年**且最近一年绩效 **B 档及以上**。
  - 激励方式为**股票期权**，按授予日公允价设定行权价。
  - 归属安排：**4 年归属，每年归属 25%**。
  - 授予后设 **1 年锁定期**，锁定期内不得行权。
  - 离职时未归属部分作废，已归属部分按计划回购。

- [ ] **Step 5: 校验语料能被灌库脚本识别**

Run: `cd backend; .\.venv\Scripts\python.exe -c "from pathlib import Path; print(sorted(p.name for p in (Path('..')/'docs'/'test-corpus').glob('*.md')))"`
Expected: 列出含 07/08/09/10 共 10 个 md（不含 README）。

- [ ] **Step 6: 提交语料**

```bash
git add "docs/test-corpus/07-薪酬与绩效管理制度.md" "docs/test-corpus/08-员工福利与商业保险管理办法.md" "docs/test-corpus/09-生育与陪产假管理规定.md" "docs/test-corpus/10-员工持股与股权激励计划.md"
git commit -m "feat: 新增 4 篇评估语料覆盖薪酬/福利/生育/股权主题"
```

> 注意：`docs/test-corpus/` 是语料、**需要 commit**（被测系统的一部分），与 `docs/plans` / `docs/design`（untracked）不同。

---

### Task 3: 扩 retrieval.jsonl 到 ≥40

每行一题，字段：`id` / `question` / `type:"knowledge_qa"` / `expected_doc` / `answer_assertions{must_include[, must_not_include]}` / `should_refuse`。

下列为**已核对语料事实的必落条目**（含旧 18 条之外的新增）。codex 直接追加这些，并按相同规则**每篇语料至少挖 4 道事实题**补足到 ≥40、拒答负样本 ≥6。`must_include` 用语料中的**具体数字/关键词**，确保程序化断言可命中。

- [ ] **Step 1: 向 `backend/eval/datasets/retrieval.jsonl` 追加新条目**

```jsonl
{"id":"qa-101","question":"绩效工资占月薪的比例是多少？","type":"knowledge_qa","expected_doc":"07-薪酬与绩效管理制度.md","answer_assertions":{"must_include":["30%"]},"should_refuse":false}
{"id":"qa-102","question":"绩效考核A档的绩效系数是多少？","type":"knowledge_qa","expected_doc":"07-薪酬与绩效管理制度.md","answer_assertions":{"must_include":["1.2"]},"should_refuse":false}
{"id":"qa-103","question":"公司每年的调薪窗口在几月？","type":"knowledge_qa","expected_doc":"07-薪酬与绩效管理制度.md","answer_assertions":{"must_include":["4"]},"should_refuse":false}
{"id":"qa-104","question":"试用期工资是转正后基本工资的百分之多少？","type":"knowledge_qa","expected_doc":"07-薪酬与绩效管理制度.md","answer_assertions":{"must_include":["80%"]},"should_refuse":false}
{"id":"qa-105","question":"补充商业医疗保险的保额是多少？","type":"knowledge_qa","expected_doc":"08-员工福利与商业保险管理办法.md","answer_assertions":{"must_include":["50"]},"should_refuse":false}
{"id":"qa-106","question":"社保报销后商业保险承担住院费用剩余部分的比例是多少？","type":"knowledge_qa","expected_doc":"08-员工福利与商业保险管理办法.md","answer_assertions":{"must_include":["80%"]},"should_refuse":false}
{"id":"qa-107","question":"公司每年提供几次免费体检？","type":"knowledge_qa","expected_doc":"08-员工福利与商业保险管理办法.md","answer_assertions":{"must_include":["1"]},"should_refuse":false}
{"id":"qa-108","question":"陪产假有多少天？","type":"knowledge_qa","expected_doc":"09-生育与陪产假管理规定.md","answer_assertions":{"must_include":["15"]},"should_refuse":false}
{"id":"qa-109","question":"育儿假每年多少天，子女多大之前可以休？","type":"knowledge_qa","expected_doc":"09-生育与陪产假管理规定.md","answer_assertions":{"must_include":["10","3"]},"should_refuse":false}
{"id":"qa-110","question":"产假一共有多少天？","type":"knowledge_qa","expected_doc":"09-生育与陪产假管理规定.md","answer_assertions":{"must_include":["158"]},"should_refuse":false}
{"id":"qa-111","question":"股权激励的归属安排是怎样的？","type":"knowledge_qa","expected_doc":"10-员工持股与股权激励计划.md","answer_assertions":{"must_include":["25%"]},"should_refuse":false}
{"id":"qa-112","question":"获得股权激励需要满足的司龄和绩效条件是什么？","type":"knowledge_qa","expected_doc":"10-员工持股与股权激励计划.md","answer_assertions":{"must_include":["2","B"]},"should_refuse":false}
{"id":"qa-113","question":"股权激励的锁定期是多久？","type":"knowledge_qa","expected_doc":"10-员工持股与股权激励计划.md","answer_assertions":{"must_include":["1"]},"should_refuse":false}
{"id":"qa-150","question":"公司报销差旅时使用哪家旅行社？","type":"knowledge_qa","expected_doc":null,"answer_assertions":{},"should_refuse":true}
{"id":"qa-151","question":"员工食堂每周的菜单是怎么安排的？","type":"knowledge_qa","expected_doc":null,"answer_assertions":{},"should_refuse":true}
{"id":"qa-152","question":"公司年会在每年几月举办？","type":"knowledge_qa","expected_doc":null,"answer_assertions":{},"should_refuse":true}
```

> 补充规则（codex 据此补到 ≥40，拒答 ≥6）：每篇 01-10 语料至少 4 道事实题；`must_include` 取语料里出现的**确切数字或唯一关键词**；对比型旧版/新版数字可用 `must_not_include` 排除旧值（如 2025 版住宿题 `must_not_include:["450"]`）；拒答题主题须**任何语料都不覆盖**（如旅行社、食堂菜单、年会时间）。

- [ ] **Step 2: 暂不跑质量测试（待 Task 5 一起转绿）**，先确认 jsonl 合法

Run: `cd backend; .\.venv\Scripts\python.exe -c "import json,pathlib; [json.loads(l) for l in pathlib.Path('eval/datasets/retrieval.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; print('retrieval jsonl OK')"`
Expected: 打印 `retrieval jsonl OK`（无 JSON 解析异常）。

- [ ] **Step 3: 提交**

```bash
git add backend/eval/datasets/retrieval.jsonl
git commit -m "feat: 评估 retrieval 数据集扩到 40+ 含新语料事实题与拒答负样本"
```

---

### Task 4: 扩 tasks.jsonl 到 ≥21（每类 ≥7）—— 解 n=4 炸点的核心

字段：`id` / `question` / `type` / `rubric_points[]`（≥3 点，每点是回答应覆盖的**具体事实**）/ `expected_docs[]`。`rubric_points` 必须能在 `expected_docs` 语料里找到依据。

- [ ] **Step 1: 向 `backend/eval/datasets/tasks.jsonl` 追加新条目**

```jsonl
{"id":"cmp-101","question":"对比2023版和2025版差旅报销在市内交通和打款时效上的差异","type":"document_compare","rubric_points":["市内交通80→120","打款5→3工作日"],"expected_docs":["01-差旅与报销管理办法-2023版.md","02-差旅与报销管理办法-2025版.md"]}
{"id":"cmp-102","question":"对比试用期员工和转正员工在年假与薪资上的差异","type":"document_compare","rubric_points":["试用期工资80%","试用期无带薪年假或需转正后享有","转正后按工龄计年假"],"expected_docs":["05-员工入职与转正管理办法.md","07-薪酬与绩效管理制度.md","03-考勤与请假管理制度.md"]}
{"id":"cmp-103","question":"对比绩效A档和C档员工在绩效系数上的差异","type":"document_compare","rubric_points":["A档系数1.2","C档系数0.8","绩效工资占月薪30%"],"expected_docs":["07-薪酬与绩效管理制度.md"]}
{"id":"rpt-101","question":"根据生育与陪产假规定，生成一份产假、陪产假、育儿假的要点报告","type":"report_generation","rubric_points":["产假158天","陪产假15天","育儿假每年10天","子女满3周岁前","产检假每月1天"],"expected_docs":["09-生育与陪产假管理规定.md"]}
{"id":"rpt-102","question":"整理一份员工福利与商业保险的要点报告","type":"report_generation","rubric_points":["五险一金","补充商业医疗保险保额50万","社保后报销80%","每年1次体检","节日福利500元"],"expected_docs":["08-员工福利与商业保险管理办法.md"]}
{"id":"rpt-103","question":"根据股权激励计划，生成一份激励对象、方式与归属安排的要点报告","type":"report_generation","rubric_points":["司龄满2年","绩效B及以上","股票期权","4年归属每年25%","1年锁定期"],"expected_docs":["10-员工持股与股权激励计划.md"]}
{"id":"rpt-104","question":"整理一份公司薪酬结构与调薪机制的要点报告","type":"report_generation","rubric_points":["基本工资+绩效工资+岗位补贴","绩效工资占30%","每年4月调薪","13薪","绩效ABCD四档"],"expected_docs":["07-薪酬与绩效管理制度.md"]}
{"id":"gen-101","question":"帮我写一份陪产假申请，配偶下周一分娩，预计休15天","type":"document_generation","rubric_points":["申请人/事由明确","陪产假15天","起止时间合理（配偶分娩后3个月内）","格式为可提交的申请正文"],"expected_docs":["09-生育与陪产假管理规定.md"]}
{"id":"gen-102","question":"帮我起草一份育儿假申请，孩子2岁，计划休5天","type":"document_generation","rubric_points":["申请人/事由明确","育儿假（子女满3周岁前）","天数在每年10天额度内","格式为可提交的申请正文"],"expected_docs":["09-生育与陪产假管理规定.md"]}
{"id":"gen-103","question":"帮我写一份免费体检预约申请说明","type":"document_generation","rubric_points":["申请人明确","引用每年1次免费体检福利","时间安排","格式为可提交的申请/说明正文"],"expected_docs":["08-员工福利与商业保险管理办法.md"]}
{"id":"gen-104","question":"帮我起草一份调薪申请，依据是上一年度绩效A档","type":"document_generation","rubric_points":["申请人明确","引用绩效A档系数1.2","对应4月调薪窗口","格式为可提交的申请正文"],"expected_docs":["07-薪酬与绩效管理制度.md"]}
```

> 补充规则（codex 补到每类 ≥7）：`document_compare` 需 ≥7（含旧 cmp-001/002 + 上面 3 条 + 再补 2 条，可对比 2023/2025 差旅其余条款、或薪酬档位）；`report_generation` ≥7；`document_generation` ≥7。每条 `rubric_points` 的事实必须在 `expected_docs` 里**真实可查**，不得臆造。

- [ ] **Step 2: 确认 jsonl 合法**

Run: `cd backend; .\.venv\Scripts\python.exe -c "import json,pathlib; rows=[json.loads(l) for l in pathlib.Path('eval/datasets/tasks.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; from collections import Counter; print(Counter(r['type'] for r in rows))"`
Expected: 打印三类计数，每类 ≥7。

- [ ] **Step 3: 提交**

```bash
git add backend/eval/datasets/tasks.jsonl
git commit -m "feat: 评估 tasks 开放式数据集扩到 21+ 每类≥7 解小样本"
```

---

### Task 5: 扩 routing.jsonl 到 ≥32（改判 gap 题 + 补 gap 负样本）

新增语料后，部分原 `knowledge_gap` 题已"有答案"，须改判为对应正样本路由；同时补足新的、语料确实不覆盖的 gap 负样本，维持 gap ≥8。

- [ ] **Step 1: 改判已被新语料覆盖的旧 gap 题**

在 `backend/eval/datasets/routing.jsonl` 中，把下列三条改判（主题已被新语料覆盖，不再是缺口）：
- `rt-009`（股权激励）→ `expected_route:"knowledge_qa"`、`expect_gap_triggered:false`、`type:"knowledge_qa"`
- `rt-011`（补充商业医疗保险）→ `expected_route:"knowledge_qa"`、`expect_gap_triggered:false`、`type:"knowledge_qa"`
- `rt-012`（陪产假/育儿假）→ `expected_route:"knowledge_qa"`、`expect_gap_triggered:false`、`type:"knowledge_qa"`

改判后这三条形如：
```jsonl
{"id":"rt-009","question":"公司的员工持股/股权激励计划细则是怎样的？","type":"knowledge_qa","expected_route":"knowledge_qa","expect_gap_triggered":false}
{"id":"rt-011","question":"公司有没有补充商业医疗保险？保额多少？","type":"knowledge_qa","expected_route":"knowledge_qa","expect_gap_triggered":false}
{"id":"rt-012","question":"陪产假和育儿假各有几天？","type":"knowledge_qa","expected_route":"knowledge_qa","expect_gap_triggered":false}
```

- [ ] **Step 2: 追加新 gap 负样本（主题任何语料都不覆盖）与各类型正样本，补到 ≥32**

```jsonl
{"id":"rt-101","question":"公司团建/团队建设活动的预算标准是多少？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
{"id":"rt-102","question":"加班餐补和打车费的报销标准是多少？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
{"id":"rt-103","question":"公司是否提供租房补贴或住房公积金以外的住房福利？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
{"id":"rt-104","question":"员工推荐入职有没有伯乐奖？奖金多少？","type":"knowledge_gap","expected_route":"knowledge_gap","expect_gap_triggered":true}
{"id":"rt-110","question":"绩效工资占月薪的比例是多少？","type":"knowledge_qa","expected_route":"knowledge_qa","expect_gap_triggered":false}
{"id":"rt-111","question":"对比2023版和2025版差旅报销在打款时效上的差异。","type":"document_compare","expected_route":"document_compare","expect_gap_triggered":false}
{"id":"rt-112","question":"根据股权激励计划生成一份归属安排要点报告。","type":"report_generation","expected_route":"report_generation","expect_gap_triggered":false}
{"id":"rt-113","question":"帮我写一份陪产假申请，配偶下周分娩。","type":"document_generation","expected_route":"document_generation","expect_gap_triggered":false}
```

> 补充规则（codex 补到 ≥32、gap ≥8）：保留旧 rt-010/013/014/015/016 的 gap 题（共 5 条）+ 上面 rt-101~104（4 条 gap）= 9 条 gap ≥8 ✅；各非 gap 路由类型（knowledge_qa / document_compare / report_generation / document_generation）每类 ≥4 条。

- [ ] **Step 3: 数据集质量测试转绿（Task 1~5 的总验收门）**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_dataset_quality.py -v`
Expected: **全部 PASS**（语料≥10、retrieval≥40 拒答≥6、tasks≥21 每类≥7、routing≥32 gap≥8、id 全局唯一）。若失败，按报错补数据直到全绿。

- [ ] **Step 4: 提交**

```bash
git add backend/eval/datasets/routing.jsonl
git commit -m "feat: 评估 routing 扩到 32+ 改判已覆盖 gap 题并补新缺口负样本"
```

---

### Task 6: 重新灌库 + 重跑评估 + 验收新报告

**Files:** 无新代码，端到端真实跑（需 MySQL/Redis/向量库/embedding/DeepSeek+千问 key 在位）。

- [ ] **Step 1: 重新幂等灌库（含新增 4 篇语料）**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.seed_corpus`
Expected: 打印新增 4 篇 `processed`，旧 6 篇 `duplicates`（去重跳过）。

- [ ] **Step 2: 重跑评估全矩阵**

Run: `cd backend; .\.venv\Scripts\python.exe -m eval.runner`
Expected: 控制台打印 baseline / +critic / +hyde 三配置对比表与成本表，`backend/eval/reports/<新 timestamp>/` 生成 `summary.md` / `cost.md` / `details.json`。

- [ ] **Step 3: 验收新报告关键变化（codex 自检 + 交 Claude 复核）**

打开最新 `reports/<timestamp>/summary.md`，确认：
  - `n` 显著增大（≈ retrieval+routing+tasks 合计，应 ≥90）。
  - **recall@3 不再恒等于 1.000**（语料变多后出现 <1 的值，说明检索有区分度了）——这是本计划成败的关键信号。
  - `rubric覆盖率`、`忠实度` 现基于 tasks ≥21（不再是 n=4），数值更可信。
  - 缺口精确率/召回率基于改判后的 gap 集（≥8）。

- [ ] **Step 4: 评估器单测回归（确认扩数据未破坏管线）**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest tests/test_eval_*.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: 不提交报告产物**

`backend/eval/reports/` 是否纳入 git 看项目既有 `.gitignore` 约定；若既有报告未被追踪，则本次新报告同样**保持 untracked**，不 `git add`。仅在控制台/文件留存供 Claude 验收与简历取数。

---

## 自检（计划 vs 目标）

- **解"评估样本太小"** → Task 4 tasks 4→≥21（每类≥7）+ Task 3 retrieval 18→≥40 + Task 5 routing 16→≥32。✅
- **解"语料 6 篇 recall@3 恒 1.0"** → Task 2 新增 4 篇语料 + Task 6 Step 3 验收 recall@3 出现 <1。✅
- **守红线不靠人肉** → Task 1 质量测试强制数量/字段/语料引用一致/id 唯一。✅
- **标注与语料一致** → Task 2 先钉事实点，Task 3/4 标注锚定具体数字；质量测试校验 `expected_doc(s)` 文件真实存在。✅
- **不改评估管线** → 全程零改 `schema/runner/judges/config`，仅扩数据 + 加测试。✅
- **gap 指标不被破坏** → Task 5 改判已覆盖 gap 题、补新 gap 负样本维持 ≥8。✅

**类型一致性核对**：retrieval 用 `expected_doc`(单)/`answer_assertions`/`should_refuse`；tasks 用 `rubric_points`/`expected_docs`(复)；routing 用 `expected_route`/`expect_gap_triggered`——均与 `backend/eval/schema.py` 现有字段及 `load_cases` 解析一致；质量测试直接读 jsonl raw 以覆盖 `expected_docs`（`load_cases` 未解析该字段，不影响校验）。✅

**Claude 验收要点（codex 跑完后）**：①实跑 `pytest tests/test_eval_dataset_quality.py` 必须全绿（不信"已达标"自述）；②实读最新 `summary.md` 确认 recall@3 出现 <1.000、tasks 相关指标 n≥21；③抽查 3 条新题的 `answer_assertions`/`rubric_points` 与语料正文逐字对得上。
