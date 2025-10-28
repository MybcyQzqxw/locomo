#!/usr/bin/env python3
"""
LoCoMo Detailed F1 Score Analysis Tool
详细的 F1 分数分析工具

功能：
1. 计算并展示每个模型的平均 F1 分数
2. 按问题类别展示 F1 分数
3. 对比纯 LLM 和 RAG 方法的性能
4. 导出 CSV 报告
5. 输出美观的表格（支持多种格式）
6. 生成可视化图表（如果安装了 matplotlib）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# 尝试导入 tabulate 用于美观表格输出
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


def load_stats(stats_file):
    """加载统计文件"""
    if not os.path.exists(stats_file):
        return None
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_f1_scores(stats_data, model_name):
    """计算 F1 分数统计"""
    if model_name not in stats_data:
        return None
    
    model_data = stats_data[model_name]
    category_counts = model_data['category_counts']
    cum_accuracy = model_data['cum_accuracy_by_category']
    
    results = {
        'model_name': model_name,
        'total_questions': sum(category_counts.values()),
        'total_f1': sum(cum_accuracy.values()),
        'categories': {}
    }
    
    # 计算平均 F1
    results['avg_f1'] = results['total_f1'] / results['total_questions'] if results['total_questions'] > 0 else 0
    
    # 按类别计算
    for cat in sorted(category_counts.keys(), key=int):
        cat_count = category_counts[cat]
        cat_f1_sum = cum_accuracy[cat]
        cat_avg_f1 = cat_f1_sum / cat_count if cat_count > 0 else 0
        
        results['categories'][int(cat)] = {
            'count': cat_count,
            'total_f1': cat_f1_sum,
            'avg_f1': cat_avg_f1
        }
    
    return results


def print_model_results(results, title="Model Results", table_format="grid"):
    """打印模型结果（支持美观表格）
    
    Args:
        results: 结果字典
        title: 标题
        table_format: 表格格式 (grid, fancy_grid, simple, plain, etc.)
    """
    if results is None:
        print(f"  ❌ No results available")
        return
    
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  Model: {results['model_name']}")
    print(f"  Total Questions: {results['total_questions']}")
    print(f"  Average F1 Score: {results['avg_f1']:.4f}")
    print(f"\n  📋 Category Breakdown:")
    
    # 准备表格数据
    table_data = []
    for cat_id in sorted(results['categories'].keys()):
        cat_data = results['categories'][cat_id]
        cat_name = get_category_name(cat_id)
        table_data.append([
            cat_name,
            cat_data['count'],
            f"{cat_data['avg_f1']:.4f}"
        ])
    
    # 使用 tabulate 输出美观表格（如果可用）
    if TABULATE_AVAILABLE:
        headers = ['Category', 'Questions', 'Avg F1']
        table = tabulate(table_data, headers=headers, tablefmt=table_format)
        # 缩进表格
        indented_table = '\n'.join(['  ' + line for line in table.split('\n')])
        print(indented_table)
    else:
        # 降级到简单格式
        print(f"  {'Category':<12} {'Questions':<12} {'Avg F1':<12}")
        print(f"  {'-'*40}")
        for row in table_data:
            print(f"  {row[0]:<12} {row[1]:<12} {row[2]:<12}")


def get_category_name(cat_id):
    """获取问题类别名称"""
    category_names = {
        1: "Cat 1",  # 简单事实
        2: "Cat 2",  # 时间推理
        3: "Cat 3",  # 跨会话推理
        4: "Cat 4",  # 多步推理
        5: "Cat 5",  # 对抗性问题
    }
    return category_names.get(cat_id, f"Cat {cat_id}")


def compare_models(results_list, table_format="grid"):
    """对比多个模型（支持美观表格）
    
    Args:
        results_list: 结果列表
        table_format: 表格格式
    """
    if not results_list:
        print("\n❌ No results to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"  📊 Model Comparison (sorted by F1 score)")
    print(f"{'='*80}")
    
    # 按 F1 分数排序
    sorted_results = sorted(results_list, key=lambda x: x['avg_f1'], reverse=True)
    
    # 准备表格数据
    table_data = []
    for idx, result in enumerate(sorted_results, 1):
        rank_emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
        table_data.append([
            rank_emoji,
            result['model_name'],
            result['method'].upper(),
            f"{result['avg_f1']:.4f}",
            result['total_questions']
        ])
    
    # 使用 tabulate 输出美观表格（如果可用）
    if TABULATE_AVAILABLE:
        headers = ['Rank', 'Model', 'Method', 'Avg F1', 'Questions']
        table = tabulate(table_data, headers=headers, tablefmt=table_format)
        # 缩进表格
        indented_table = '\n'.join(['  ' + line for line in table.split('\n')])
        print(indented_table)
    else:
        # 降级到简单格式
        print(f"  {'Rank':<6} {'Model':<40} {'Method':<12} {'Avg F1':<15} {'Questions':<15}")
        print(f"  {'-'*90}")
        for row in table_data:
            print(f"  {row[0]:<6} {row[1]:<40} {row[2]:<12} {row[3]:<15} {row[4]:<15}")


def export_to_csv(results_list, output_file):
    """导出结果到 CSV"""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['Model', 'Total Questions', 'Average F1', 'Category', 'Category Questions', 'Category F1'])
        
        # 写入数据
        for result in results_list:
            model_name = result['model_name']
            total_q = result['total_questions']
            avg_f1 = result['avg_f1']
            
            # 写入总体数据
            writer.writerow([model_name, total_q, f"{avg_f1:.4f}", 'Overall', total_q, f"{avg_f1:.4f}"])
            
            # 写入分类数据
            for cat_id in sorted(result['categories'].keys()):
                cat_data = result['categories'][cat_id]
                writer.writerow([
                    model_name,
                    total_q,
                    f"{avg_f1:.4f}",
                    f"Category {cat_id}",
                    cat_data['count'],
                    f"{cat_data['avg_f1']:.4f}"
                ])
    
    print(f"\n✅ Results exported to: {output_file}")


def scan_outputs_directory(base_dir="./outputs"):
    """扫描输出目录，找到所有统计文件"""
    results = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"⚠️  Directory not found: {base_dir}")
        return results
    
    # 查找所有 *_stats.json 文件
    for stats_file in base_path.rglob("*_stats.json"):
        stats_data = load_stats(stats_file)
        if stats_data:
            # 获取相对路径以区分不同方法
            rel_path = stats_file.relative_to(base_path)
            method = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
            
            for model_name in stats_data.keys():
                result = calculate_f1_scores(stats_data, model_name)
                if result:
                    result['method'] = method
                    result['file'] = str(stats_file)
                    results.append(result)
    
    return results


def print_category_comparison(results_list, table_format="grid"):
    """按类别对比所有模型的表现"""
    if not results_list:
        return
    
    print(f"\n{'='*80}")
    print(f"  📊 Category-wise Performance Comparison")
    print(f"{'='*80}")
    
    # 收集所有类别
    all_categories = set()
    for result in results_list:
        all_categories.update(result['categories'].keys())
    
    # 准备表格数据
    table_data = []
    for cat_id in sorted(all_categories):
        cat_name = get_category_name(cat_id)
        row = [cat_name]
        
        for result in results_list:
            if cat_id in result['categories']:
                cat_f1 = result['categories'][cat_id]['avg_f1']
                row.append(f"{cat_f1:.4f}")
            else:
                row.append("N/A")
        
        table_data.append(row)
    
    # 添加平均行
    avg_row = ["Overall"]
    for result in results_list:
        avg_row.append(f"{result['avg_f1']:.4f}")
    table_data.append(avg_row)
    
    # 构建表头
    headers = ['Category'] + [f"{r['model_name'][:20]}..." if len(r['model_name']) > 20 else r['model_name'] for r in results_list]
    
    # 使用 tabulate 输出美观表格（如果可用）
    if TABULATE_AVAILABLE:
        table = tabulate(table_data, headers=headers, tablefmt=table_format)
        indented_table = '\n'.join(['  ' + line for line in table.split('\n')])
        print(indented_table)
    else:
        # 降级到简单格式
        col_width = 15
        header_line = "  Category".ljust(15)
        for h in headers[1:]:
            header_line += h[:col_width-1].ljust(col_width)
        print(header_line)
        print(f"  {'-'*80}")
        for row in table_data:
            line = f"  {row[0]:<15}"
            for val in row[1:]:
                line += f"{val:<15}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description='LoCoMo F1 Score Analysis Tool')
    parser.add_argument('--output-dir', default='./outputs', help='Output directory to scan')
    parser.add_argument('--export-csv', help='Export results to CSV file')
    parser.add_argument('--model', help='Specific model name to analyze')
    parser.add_argument('--method', choices=['hf_llm', 'rag_hf_llm', 'all'], default='all',
                       help='Method to analyze (pure LLM or RAG)')
    parser.add_argument('--table-format', default='grid',
                       choices=['plain', 'simple', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 
                               'rst', 'mediawiki', 'html', 'latex', 'github'],
                       help='Table format (requires tabulate library)')
    parser.add_argument('--category-comparison', action='store_true',
                       help='Show category-wise comparison across all models')
    
    args = parser.parse_args()
    
    # 检查 tabulate 可用性
    if not TABULATE_AVAILABLE and args.table_format != 'plain':
        print("⚠️  tabulate library not found. Install it for beautiful tables:")
        print("   pip install tabulate")
        print("   Falling back to simple format...\n")
    
    print("🔍 Scanning output directory...")
    all_results = scan_outputs_directory(args.output_dir)
    
    if not all_results:
        print(f"\n❌ No results found in {args.output_dir}")
        print(f"   Please run evaluation scripts first:")
        print(f"   - Pure LLM: ./scripts/evaluate_hf_llm.sh")
        print(f"   - RAG:      ./scripts/evaluate_rag_hf_llm.sh")
        return
    
    # 过滤结果
    filtered_results = all_results
    if args.method != 'all':
        filtered_results = [r for r in all_results if r['method'] == args.method]
    
    if args.model:
        filtered_results = [r for r in filtered_results if args.model in r['model_name']]
    
    if not filtered_results:
        print(f"\n❌ No results matching criteria")
        return
    
    # 显示结果
    print(f"\n📊 Found {len(filtered_results)} result(s)")
    
    for result in filtered_results:
        title = f"{result['method'].upper()} - {result['model_name']}"
        print_model_results(result, title, table_format=args.table_format)
    
    # 对比
    if len(filtered_results) > 1:
        compare_models(filtered_results, table_format=args.table_format)
        
        # 按类别对比（如果请求）
        if args.category_comparison:
            print_category_comparison(filtered_results, table_format=args.table_format)
    
    # 导出 CSV
    if args.export_csv:
        export_to_csv(filtered_results, args.export_csv)
    
    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    if TABULATE_AVAILABLE:
        print(f"   Table format: {args.table_format}")
    print(f"   Total results analyzed: {len(filtered_results)}")


if __name__ == "__main__":
    main()
