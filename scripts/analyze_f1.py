#!/usr/bin/env python3
"""
LoCoMo Detailed F1 Score Analysis Tool
详细的 F1 分数分析工具

功能：
1. 计算并展示每个模型的平均 F1 分数
2. 按问题类别展示 F1 分数
3. 对比纯 LLM 和 RAG 方法的性能
4. 导出 CSV 报告
5. 生成可视化图表（如果安装了 matplotlib）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


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


def print_model_results(results, title="Model Results"):
    """打印模型结果"""
    if results is None:
        print(f"  ❌ No results available")
        return
    
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Model: {results['model_name']}")
    print(f"  Total Questions: {results['total_questions']}")
    print(f"  Average F1 Score: {results['avg_f1']:.4f}")
    print(f"\n  Category Breakdown:")
    print(f"  {'Category':<12} {'Questions':<12} {'Avg F1':<12}")
    print(f"  {'-'*40}")
    
    for cat_id in sorted(results['categories'].keys()):
        cat_data = results['categories'][cat_id]
        cat_name = get_category_name(cat_id)
        print(f"  {cat_name:<12} {cat_data['count']:<12} {cat_data['avg_f1']:<12.4f}")


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


def compare_models(results_list):
    """对比多个模型"""
    if not results_list:
        print("\n❌ No results to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"  📊 Model Comparison")
    print(f"{'='*80}")
    print(f"  {'Model':<40} {'Avg F1':<15} {'Questions':<15}")
    print(f"  {'-'*70}")
    
    # 按 F1 分数排序
    sorted_results = sorted(results_list, key=lambda x: x['avg_f1'], reverse=True)
    
    for result in sorted_results:
        model_display = result['model_name'][:38]
        print(f"  {model_display:<40} {result['avg_f1']:<15.4f} {result['total_questions']:<15}")


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


def main():
    parser = argparse.ArgumentParser(description='LoCoMo F1 Score Analysis Tool')
    parser.add_argument('--output-dir', default='./outputs', help='Output directory to scan')
    parser.add_argument('--export-csv', help='Export results to CSV file')
    parser.add_argument('--model', help='Specific model name to analyze')
    parser.add_argument('--method', choices=['hf_llm', 'rag_hf_llm', 'all'], default='all',
                       help='Method to analyze (pure LLM or RAG)')
    
    args = parser.parse_args()
    
    print("🔍 Scanning output directory...")
    all_results = scan_outputs_directory(args.output_dir)
    
    if not all_results:
        print(f"\n❌ No results found in {args.output_dir}")
        print(f"   Please run evaluation scripts first:")
        print(f"   - Pure LLM: ./scripts/evaluate_hf_llm_test.sh")
        print(f"   - RAG:      ./scripts/evaluate_rag_hf_llm_test.sh")
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
        print_model_results(result, title)
    
    # 对比
    if len(filtered_results) > 1:
        compare_models(filtered_results)
    
    # 导出 CSV
    if args.export_csv:
        export_to_csv(filtered_results, args.export_csv)
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
