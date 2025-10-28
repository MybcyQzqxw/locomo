# Recall@K 分析功能说明

## 🎯 什么是 Recall@K？

**Recall@K** 是 RAG（检索增强生成）方法的核心评估指标，用于衡量**检索质量**。

### 定义
```
Recall@K = 包含正确证据的检索结果数 / 总证据数
```

### 通俗解释
假设问题的答案需要 3 条对话作为证据，RAG 系统检索出了 Top-10 条对话：
- 如果 Top-10 中包含了全部 3 条证据 → Recall@10 = 3/3 = 1.0 ✅
- 如果 Top-10 中包含了 2 条证据 → Recall@10 = 2/3 = 0.67 ⚠️
- 如果 Top-10 中包含了 0 条证据 → Recall@10 = 0/3 = 0.0 ❌

---

## 📊 Recall@K vs F1 Score

| 指标 | 衡量对象 | 计算方式 | 适用场景 |
|------|---------|---------|---------|
| **Recall@K** | **检索质量** | 检索到的证据比例 | 评估 RAG 的检索效果 |
| **F1 Score** | **答案质量** | 生成答案与标准答案的匹配度 | 评估最终答案的准确性 |

### 关系
```
好的 Recall@K → 为模型提供正确的上下文
好的上下文 → 模型更容易生成正确答案
正确答案 → 高 F1 Score

所以：Recall@K 是 F1 Score 的"先行指标"
```

---

## 🚀 使用方法

### 1. 基础查看（自动显示 Recall@K）
```bash
python3 scripts/analyze_f1.py --table-format fancy_grid
```

**输出示例：**
```
================================================================================
  RAG_HF_LLM - mistral-instruct-7b-32k-v2_observation_top_5
================================================================================
  Model: mistral-instruct-7b-32k-v2_observation_top_5
  Total Questions: 1986
  Average F1 Score: 0.2702
  Average Recall@K: 0.8543
  📊 Note: Recall@K measures whether evidence dialogues are in Top-K retrieved contexts

  📋 Category Breakdown:
╔═════════════╤═════════════╤═══════════╤═══════════╗
║ Category    │   Questions │ Avg F1    │ Recall@K  ║
╠═════════════╪═════════════╪═══════════╪═══════════╣
║ Single Hop  │         282 │ 0.1905    │ 0.9234    ║
║ Temporal    │         321 │ 0.1682    │ 0.8567    ║
║ Multi Hop   │          96 │ 0.0794    │ 0.7123    ║
║ Open Domain │         841 │ 0.2179    │ 0.8891    ║
║ Adversarial │         446 │ 0.5336    │ 0.8234    ║
╚═════════════╧═════════════╧═══════════╧═══════════╝
```

### 2. 模型对比（包含 Recall@K）
```bash
python3 scripts/analyze_f1.py --category-comparison --table-format fancy_grid
```

**输出示例：**
```
╔═════╤══════════════════════════════╤════════╤═══════════╤═════════════╤═══════════╗
║ Rank│ Model                        │ Method │ Avg F1    │ Questions   │ Recall@K  ║
╠═════╪══════════════════════════════╪════════╪═══════════╪═════════════╪═══════════╣
║ 🥇  │ observation_top_5            │ RAG    │ 0.2702    │ 1986        │ 0.8543    ║
║ 🥈  │ observation_top_10           │ RAG    │ 0.2652    │ 1986        │ 0.9012    ║
║ 🥉  │ dialog_top_5                 │ RAG    │ 0.2499    │ 1986        │ 0.7834    ║
║ 4.  │ mistral-instruct-7b-32k-v2   │ LLM    │ 0.1944    │ 1986        │ N/A       ║
╚═════╧══════════════════════════════╧════════╧═══════════╧═════════════╧═══════════╝
```

### 3. 导出 CSV（包含 Recall@K）
```bash
python3 scripts/analyze_f1.py --export-csv results_with_recall.csv
```

**CSV 格式：**
```csv
Model,Total Questions,Average F1,Category,Category Questions,Category F1,Category Recall@K
observation_top_5,1986,0.2702,Overall,1986,0.2702,0.8543
observation_top_5,1986,0.2702,Category 1,282,0.1905,0.9234
...
```

---

## 📈 典型的 Recall@K 值

### 按 K 值
```
Recall@5  = 0.75-0.85  → Top-5 检索较精准但覆盖有限
Recall@10 = 0.85-0.92  → Top-10 是常见的平衡点
Recall@25 = 0.90-0.95  → Top-25 覆盖广但可能引入噪音
Recall@50 = 0.92-0.97  → Top-50 覆盖很全但噪音更多
```

### 按 RAG 模式
```
Dialog (原始对话):        Recall@10 ≈ 0.78-0.85
Observation (观察总结):   Recall@10 ≈ 0.85-0.90  ← 通常最好
Summary (会话摘要):       Recall@10 ≈ 0.70-0.80
```

### 按问题类别
```
Single Hop (单跳):      Recall ≈ 0.90-0.95  ← 最容易检索
Temporal (时间):        Recall ≈ 0.85-0.90
Open Domain (开放域):   Recall ≈ 0.85-0.92
Multi Hop (多跳):       Recall ≈ 0.70-0.85  ← 最难检索
Adversarial (对抗):     Recall ≈ 0.80-0.85
```

---

## 🤔 为什么 Recall 高但 F1 低？

### 常见原因

#### 1. 检索到了但模型理解不了
```
Recall@10 = 0.90  ✅ 找到了 90% 的证据
F1 = 0.25         ❌ 但模型只答对了 25%

原因：
- 检索到的上下文太多（如 Top-50），模型被干扰
- 检索到的片段顺序混乱，缺乏时间连贯性
- 模型本身理解能力有限
```

#### 2. 检索模式不匹配问题类型
```
Multi Hop 问题:
Recall@5 = 0.70   ← 只找到部分证据
F1 = 0.08         ← 答案需要整合多条证据，缺一不可

原因：
- K 值太小（5），无法覆盖所有相关对话
- 应该使用更大的 K 值（如 25）
```

#### 3. 证据噪音问题
```
Recall@50 = 0.95  ← 找到了几乎所有证据
F1 = 0.24         ← 但答案质量反而下降

原因：
- Top-50 中包含太多不相关的对话
- 模型被噪音干扰，生成了错误答案
```

---

## 💡 优化建议

### 1. 根据 Recall@K 调整 K 值
```bash
# 如果 Recall@5 < 0.70，说明 K 太小
→ 尝试增加到 Top-10 或 Top-25

# 如果 Recall@50 很高但 F1 低，说明噪音太多
→ 尝试减少到 Top-10 或 Top-25
```

### 2. 根据问题类别选择策略
```
Single Hop: K=5 即可（单条证据）
Multi Hop:  K=25 更好（需要多条证据）
Temporal:   K=10-25（需要时间跨度）
```

### 3. 尝试不同 RAG 模式
```
Observation 模式: 通常 Recall 最高
Dialog 模式:      保留原文细节
Summary 模式:     高度概括但可能丢失信息
```

---

## 📊 完整分析命令

```bash
# 1. 快速查看所有结果（包括 Recall@K）
python3 scripts/analyze_f1.py --table-format fancy_grid

# 2. 对比所有 RAG 配置
python3 scripts/analyze_f1.py --method rag_hf_llm --category-comparison

# 3. 只看某个模式（如 observation）
python3 scripts/analyze_f1.py --method rag_hf_llm | grep observation

# 4. 导出完整报告
python3 scripts/analyze_f1.py \
    --export-csv full_report_with_recall.csv \
    --category-comparison \
    --table-format github > report.md
```

---

## 🎯 Recall@K 的重要性

**Recall@K 是 RAG 系统的"体检指标"：**

1. **诊断检索质量**
   - Recall 低 → 检索模型有问题（Dragon 编码器、向量化等）
   - Recall 高但 F1 低 → 生成模型有问题（LLM 理解能力）

2. **指导超参数调整**
   - K 值、RAG 模式、检索器类型

3. **理解模型瓶颈**
   - 是检索不准？还是生成不行？

**只看 F1 Score 无法区分这些问题，Recall@K 提供了关键洞察！** 🎯
