# analyze_f1.py 美观表格功能使用指南

## 🎨 新增功能

1. **美观的表格输出**（使用 tabulate 库）
2. **多种表格格式**（grid, fancy_grid, github, markdown 等）
3. **按类别对比模型**
4. **排名显示**（🥇🥈🥉）

---

## 📦 安装依赖（可选，但推荐）

```bash
pip install tabulate
```

**注意：** 如果不安装 tabulate，脚本会自动降级到简单文本格式，不会报错。

---

## 🚀 使用示例

### 1. 基础用法（默认 grid 格式）

```bash
python3 scripts/analyze_f1.py
```

### 2. 使用 fancy_grid 格式（最美观）

```bash
python3 scripts/analyze_f1.py --table-format fancy_grid
```

### 3. GitHub Markdown 格式（适合文档）

```bash
python3 scripts/analyze_f1.py --table-format github
```

### 4. 导出 CSV + 显示类别对比

```bash
python3 scripts/analyze_f1.py --export-csv results.csv --category-comparison
```

### 5. 只分析特定模型

```bash
python3 scripts/analyze_f1.py --model mistral --table-format fancy_grid
```

### 6. 只分析 RAG 方法

```bash
python3 scripts/analyze_f1.py --method rag_hf_llm --category-comparison
```

---

## 🎨 支持的表格格式

| 格式 | 描述 | 适用场景 |
|------|------|---------|
| `plain` | 纯文本，无边框 | 简单输出 |
| `simple` | 简单边框 | 终端查看 |
| `grid` | 网格边框（**默认**） | 平衡美观和兼容性 |
| `fancy_grid` | 双线网格（**最美观**） | 终端演示 |
| `pipe` | 管道符分隔 | 轻量级表格 |
| `github` | GitHub Markdown | 文档、README |
| `rst` | reStructuredText | Sphinx 文档 |
| `html` | HTML 表格 | 网页展示 |
| `latex` | LaTeX 表格 | 学术论文 |

---

## 📊 输出示例

### Grid 格式（默认）
```
╒════════════╤═════════════╤═══════════╕
│ Category   │   Questions │ Avg F1    │
╞════════════╪═════════════╪═══════════╡
│ Cat 1      │          50 │ 0.8542    │
├────────────┼─────────────┼───────────┤
│ Cat 2      │          30 │ 0.7234    │
╘════════════╧═════════════╧═══════════╛
```

### Fancy Grid 格式（最美观）
```
╔════════════╤═════════════╤═══════════╗
║ Category   │   Questions │ Avg F1    ║
╠════════════╪═════════════╪═══════════╣
║ Cat 1      │          50 │ 0.8542    ║
╟────────────┼─────────────┼───────────╢
║ Cat 2      │          30 │ 0.7234    ║
╚════════════╧═════════════╧═══════════╝
```

### GitHub Markdown 格式
```
| Category   |   Questions | Avg F1   |
|------------|-------------|----------|
| Cat 1      |          50 | 0.8542   |
| Cat 2      |          30 | 0.7234   |
```

---

## 🆕 新功能说明

### 1. 排名显示
模型对比时会自动显示排名：
- 🥇 第一名
- 🥈 第二名  
- 🥉 第三名
- 4., 5., ... 后续排名

### 2. 类别对比矩阵（--category-comparison）
横向对比所有模型在各个类别上的表现：

```
╔════════════╤══════════════╤══════════════╗
║ Category   │ Model A      │ Model B      ║
╠════════════╪══════════════╪══════════════╣
║ Cat 1      │ 0.8542       │ 0.8231       ║
║ Cat 2      │ 0.7234       │ 0.7556       ║
║ Overall    │ 0.7888       │ 0.7893       ║
╚════════════╧══════════════╧══════════════╝
```

### 3. 自动降级
如果没有安装 tabulate，脚本会：
- ⚠️ 显示友好提示
- ✅ 自动使用简单文本格式
- ✅ 所有功能正常工作

---

## 💡 最佳实践

### 日常使用（快速查看）
```bash
python3 scripts/analyze_f1.py --table-format fancy_grid
```

### 生成报告（文档）
```bash
python3 scripts/analyze_f1.py \
    --table-format github \
    --category-comparison \
    --export-csv full_report.csv \
    > report.md
```

### 对比 RAG vs 纯 LLM
```bash
# 查看所有结果
python3 scripts/analyze_f1.py --category-comparison --table-format fancy_grid

# 只看纯 LLM
python3 scripts/analyze_f1.py --method hf_llm --table-format grid

# 只看 RAG
python3 scripts/analyze_f1.py --method rag_hf_llm --table-format grid
```

---

## 🐛 故障排除

### 问题：表格显示乱码
**解决方案：** 确保终端支持 UTF-8 编码

### 问题：提示 "tabulate library not found"
**解决方案：** 
```bash
pip install tabulate
```
或者忽略提示，脚本会使用简单格式

### 问题：表格太宽，超出终端
**解决方案：** 使用 `simple` 或 `plain` 格式
```bash
python3 scripts/analyze_f1.py --table-format simple
```

---

## 📝 完整参数列表

```bash
python3 scripts/analyze_f1.py \
    --output-dir ./outputs \              # 输出目录
    --model mistral \                     # 过滤特定模型
    --method all \                        # all/hf_llm/rag_hf_llm
    --table-format fancy_grid \           # 表格格式
    --category-comparison \               # 显示类别对比
    --export-csv results.csv              # 导出 CSV
```
