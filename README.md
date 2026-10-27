# Insurance AI

人身险理赔智能判定系统 —— **重疾认定**(本次确诊是否属于重大疾病)。

对一组理赔申请材料(病理报告、诊断证明、出院小结、检查报告等),通过
**OCR → 聚合 → LLM 抽取 → 规则 + LLM 投票判定**,输出三分类结论。

---

## 结论三分类

|结论|含义|触发|
|-|-|-|
|**是**|本次确诊属于重疾|排除硬规则全过 + LLM 投票 confidence ≥ 0.70|
|**否**|本次确诊不构成重疾|criteria 未过 / exclusion 命中 / 既往病史 / confidence < 0.70 / 疾病不在 100 种白名单|
|**无法认定**|关键材料 / 关键指标 / 解析失败|硬规则触发(6 种): 缺材料 / 缺诊断 / 缺指标 / 疾病不在白名单 / 兜底失败|

---

## 端到端流水线

```
                案例目录(图片/PDF)
                      │
                      ▼
            ┌─────────────────────┐
            │  1_ocr          PaddleOCR 3.3 
            │  每份文件 →     OCRResult 抽文本层
            └─────────────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  2_aggregate        │ MergedCase
            │  多份 OCR → merged_ │ 4 类齐全性(病理/诊断/出院/检查)
            │  text + per_doc     │ doc_type: 文件名 → 标题嗅探(500字)
            └─────────────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  3_llm_extract      │ 
            │  OCR 文本           │ response_format=json_object
            │  CaseExtraction     │ DocType Literal 强校验 + 归一化
            │                     │ 网络/SSL 异常 → 指数退避重试
            └─────────────────────┘
                      │
                      ▼
            ┌──────────────────────────────────────────┐
            │  4_judge(3 层 + is_historical)         
            │                                          
            │  Layer 1  硬规则 gate                     
            │   ├─ < 2 类诊断材料 → 无法认定             
            │   └─ 缺诊断名        → 无法认定            
            │                                         
            │  Layer 2  算法 + LLM 双重规则匹配         
            │   ├─ ICD 范围 / 疾病名 → candidates       
            │   └─ 0 候选 → LLM 二次比对                
            │         ├─ 在 100 种 → 用 rule_id       
            │         └─ 失败/不在 → 否/无法认定        
            │                                          
            │  Layer 2.5  关键指标检查                  
            │   └─ 规则要求但 LLM 没抽到                
            │         → 无法认定 - 缺指标               
            │                                          
            │  Layer 3  LLM 投票 3 次(并发) + 后置      
            │   ├─ is_historical=true → 否            
            │   ├─ exclusion 命中   → 否               
            │   ├─ criteria 部分缺 → WARNING(不直接判否)
            │   ├─ confidence ≥ 0.70 → 是              
            │   ├─ confidence < 0.70 → 否              
            │   └─ 否 + information_insufficient=True 
            │      → 无法认定(INSUFFICIENT_INFO 升级)  
            └──────────────────────────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  5_evaluate 评析    │ LLM 综合解读
            │  整合前 4 阶段结果    │ 1 段话 + 关键要点
            │  → 关键依据/风险/建议 │ 失败兜底,不阻塞主流程
            └──────────────────────┘
                      │
                      ▼
            Result
            (结论 + 置信度 + 规则 + 风险码 + 评析)
```

---

## 项目结构

```
insurance\_ai/
├── src/
│   ├── ocr/                 # 多模态文档解析
│   │   ├── schemas.py       # OCRResult / OCRPage / OCRBlock
│   │   └── ocr_runner.py    # PaddleOCR (lru_cache 单例) + pdfplumber
│   ├── aggregation/         # 多源文本归并
│   │   ├── schemas.py       # MergedCase / MergedDocument
│   │   └── aggregator.py    # detect_doc_type (文件名/标题嗅探 500字) / aggregate
│   ├── extraction/          # LLM 抽取
│   │   ├── schemas.py       # CaseExtraction
│   │   ├── prompt_builder.py
│   │   └── extractor.py     # SSL/超时指数退避 + Pydantic Literal 校验 + 归一化
│   ├── judge/               # 判定器
│   │   ├── engine.py        # 4 层主流程 + Layer 1 降级
│   │   ├── llm/
│   │   │   ├── prompt_builder.py  # SELF_ASSESS / FALLBACK_MATCH / 含既往口径
│   │   │   └── assessor.py        # 3 轮并发投票 + 二次比对
│   │   ├── rules/
│   │   │   └── matcher.py         # ICD 范围 / 疾病名
│   │   ├── stages/
│   │   │   ├── precheck.py        # Layer 1 硬规则(2 类材料 + 1 类核心)
│   │   │   └── postcheck.py       # Layer 3 后置 + is_historical 判否 + 信息不充分升级
│   │   ├── evaluator.py     # Stage 5 评析(LLM 综合解读)
│   │   └── schema/
│   │       └── output_schema.py   # JudgeOutput
│   ├── llm/                 # LLM 客户端(OpenAI 兼容 / requests)
│   │   └── client.py        # response_format=json_object + 60s timeout + 指数退避
│   ├── executor.py          # 端到端编排:5 stage + StepLogger + OCR 准确度
│   ├── webui/               # Streamlit(5 步 + 评析 + OCR 准确度)
│   │   ├── app.py
│   │   └── demo_cases.py    # 预置 demo
│   └── utils/
│       ├── step_logger.py   # 阶段输入/输出日志 + 落盘 logs/
│       └── retention.py
├── data/
│   ├── knowledge_base/
│   │   ├── judgment_rules.json    # 100 条 R 规则的 criteria / exclusions
│   │   ├── diseases.json          # 100 条 R ↔ ICD-10 映射(171 子类)
│   │   └── icd_official.json      # 国临版 ICD-10 字典(35587 条)
│   └── output/              # 输出结果
├── logs/                    # 步骤日志(case_<id>.log.json)
├── scripts/
│   ├── batch_run.py         # CLI:单 case / 批跑
├── docs/
│   ├── rules_intro.md       # 规则介绍
│   └── diseases_intro.md    # 重疾介绍
└── .env / requirements.txt
```

---

## 快速开始

```bash
# 1. 配环境(Python 3.13)
conda activate insurance_ai

# 2. 装依赖
pip install -r requirements.txt
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleocr

# 3. 配 .env
cp .env.example .env
# 填 INSURANCE_LLM_BASE_URL / INSURANCE_LLM_API_KEY / INSURANCE_LLM_MODEL

# 4. 跑单案例
python -m scripts.batch_run single \
  --case <CASE_ID> \
  --dir <路径/到/案例目录>

# 5. 批跑一个目录下的所有案例
python -m scripts.batch_run batch \
  --batch-dir <路径/到/案例根目录> \
  --limit 5
```

---

启动 WebUI

```bash
# 启动 streamlit
streamlit run src/webui/app.py \
  --server.headless true \
  --server.port 8502 \
  --server.address 0.0.0.0 \
  --server.fileWatcherType poll \
  --browser.gatherUsageStats false
```

启动后访问：
http://localhost:8502

WebUI 功能：
左栏:案例 ID 输入、上传图片(可多选)、预置 Demo 下拉
右栏:5 步流水线状态(OCR/聚合/LLM 抽取/判定/评析) + 4 列指标(置信度/命中规则/总耗时/OCR 平均准确度)
跑完自动清理临时图片



## 关键设计

### 1\. 置信度阈值(0.70)

贴近业务口径:经验值 0.70 让"ICD 范围命中 + 关键 evidence 齐备"的 case 判"是"。

|区间|结论|
|-|-|
|≥ 0.70 + 未命中 exclusion/既往病史|是|
|< 0.70|否(永远不是"无法认定")|

### 2\. 既往病史判否(is\_historical)

抽取 stage 标记诊断是"本次新发/复发"还是"既往(>1 年前 + 无治疗记录)"。判定 stage 看到 `is_historical=true` → 直接判否 + 风险码 HISTORICAL_DIAGNOSIS,绕过 confidence 阈值。

### 3\. 信息不充分升级路径

抽取 stage 暴露"信息不充分"信号(`information_insufficient` 字段 + `extraction_notes` 关键词),判定 stage 走完正常路径:

* 若结论="否" + 信息不充分信号=true → 升级为"无法认定"
* 若结论="是" → 维持"是"(信息不充分但 LLM 仍给出高置信度)

### 4\. 投票 3 次(并发)

```python
# 3 轮投票并发
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = \[pool.submit(_do_single_vote, ...) for _ in range(3)]
# 聚合:方差 < 0.01 → 取平均;否则 → 取中位数
```

### 5\. 硬规则永远先于 LLM(优先级降序)

1. `is_historical=true` → 否 + HISTORICAL_DIAGNOSIS
2. exclusion 命中任一 → 否
3. criteria 部分缺失 → WARNING(不直接判否,让 confidence 主导)
4. confidence ≥ 0.70 → 是
5. confidence < 0.70 → 否
6. 否 + information_insufficient=True → 升级为无法认定(INSUFFICIENT_INFO)
7. 缺材料 / 缺诊断 / 缺指标 → 无法认定

### 6\. LLM 调用稳定性

* `response_format={"type": "json_object"}` 强制 LLM 返 JSON(避免自然语言长响应超时)
* 网络/SSL/Timeout 错误 → 指数退避重试(1s → 2s → 4s)
* Pydantic Literal 强校验 + 后处理归一化(`诊断书` → `诊断证明`)
* 单次 timeout 60s
* 输出剥离 `<think>...</think>

### 7\. 5 步流水线 + StepLogger

每个 stage 都有结构化输入/输出日志,落盘到 `logs/case_<id>.log.json`。
WebUI 实时显示 5 步状态 + 详细输入输出。
**Result JSON 顶层暴露 `ocr_avg_confidence` / `ocr_min_confidence`**。

### 8\. 评析模块(Stage 5)

* 输入:前 4 阶段结果(抽取的疾病/材料/指标 + 判定结论/规则/风险码)
* 输出:`{summary: 一段话 80-200 字, 关键依据/风险点/后续建议; risk_callouts: \[关键要点]}`
* 失败兜底:LLM 调用失败时返回默认文本,不阻塞主流程
* WebUI 右栏"💬 评析"区域直接显示



\---

\---

## 相关文档

* `docs/rules_intro.md` — 判定规则介绍(分级 / 类型 / 阈值 / 投票)
* `docs/diseases_intro.md` — 100 种重疾逐条简介 + 排除条款 + 边界
* `data/knowledge_base/judgment_rules.json` — 规则数据源
* `data/knowledge_base/diseases.json` — 重疾 ↔ ICD 映射
* `logs/case_<id>.log.json` — 每次 case 跑批的 5 步结构化日志

