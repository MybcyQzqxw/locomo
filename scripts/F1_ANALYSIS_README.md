# F1 分数分析工具使用说明

本目录包含用于分析 LoCoMo 评估结果的 Python 脚本。

## 🔬 analyze_f1.py - F1 分数分析工具

**功能**：
- 详细的 F1 分数分析
- 按问题类别统计
- 模型对比
- 导出 CSV 报告
- 自动扫描所有输出结果

**使用方法**：

### 基础用法 - 展示所有结果
```bash
python3 scripts/analyze_f1.py
```

### 导出 CSV 报告
```bash
python3 scripts/analyze_f1.py --export-csv results.csv
```

### 只分析纯 LLM 结果
```bash
python3 scripts/analyze_f1.py --method hf_llm
```

### 只分析 RAG 结果
```bash
python3 scripts/analyze_f1.py --method rag_hf_llm
```

### 分析特定模型
```bash
python3 scripts/analyze_f1.py --model mistral
```

### 组合使用
```bash
python3 scripts/analyze_f1.py --method rag_hf_llm --export-csv rag_results.csv
```

**输出示例**：
```
🔍 Scanning output directory...

📊 Found 1 result(s)

============================================================
  HF_LLM - mistral-instruct-7b-32k-v2
============================================================
  Model: mistral-instruct-7b-32k-v2
  Total Questions: 199
  Average F1 Score: 0.2145

  Category Breakdown:
  Category     Questions    Avg F1      
  ----------------------------------------
  Cat 1        32           0.1501      
  Cat 2        37           0.1342      
  Cat 3        13           0.0713      
  Cat 4        70           0.2428      
  Cat 5        47           0.3191      

✅ Results exported to: results.csv
```

---

## 📁 输出文件位置

评估脚本会自动在以下位置生成结果：

```
outputs/
├── hf_llm/                          # 纯 LLM 结果
│   ├── locomo1_qa.json              # 详细问答结果（含每题F1）
│   └── locomo1_qa_stats.json        # 统计信息（含分类F1）
│
└── rag_hf_llm/                      # RAG 结果
    ├── locomo1_qa.json              # 详细问答结果
    ├── locomo1_qa_stats.json        # 统计信息
    └── embeddings/                   # 向量缓存
        └── *.pkl
```

---

## 📋 问题类别说明

- **Category 1**: 简单事实问题（需要记忆单个事实）
- **Category 2**: 时间推理问题（需要理解时间关系）
- **Category 3**: 跨会话推理问题（需要整合多个会话信息）
- **Category 4**: 多步推理问题（需要多步逻辑推理）
- **Category 5**: 对抗性问题（可能无法从对话中找到答案）

---

## 🎯 F1 分数说明

F1 分数计算公式：
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **F1 = 1.0**: 完美匹配
- **F1 = 0.5**: 部分匹配
- **F1 = 0.0**: 完全不匹配

---

## 💡 使用建议

1. **快速查看**：直接运行 `python3 scripts/analyze_f1.py` 查看所有结果
2. **详细分析**：查看按类别统计的详细 F1 分数
3. **对比实验**：运行不同配置后，使用 `--export-csv` 导出结果对比
4. **Excel 分析**：将导出的 CSV 文件导入 Excel 进行进一步分析

---

## 🚀 完整工作流程示例

```bash
# 1. 运行纯 LLM 评估
./scripts/evaluate_hf_llm_test.sh

# 2. 运行 RAG 评估
./scripts/evaluate_rag_hf_llm_test.sh

# 3. 查看所有结果
python3 scripts/analyze_f1.py

# 4. 导出 CSV 报告用于深入分析
python3 scripts/analyze_f1.py --export-csv comparison.csv

# 5. 在 Excel 中打开 comparison.csv 进行可视化分析
```

---

## 📖 相关文件

- `task_eval/evaluate_qa.py` - 主评估脚本
- `task_eval/evaluation_stats.py` - 统计计算模块
- `task_eval/evaluation.py` - F1 分数计算模块
