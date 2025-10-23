"""
LoCoMo 问答任务评估脚本

该脚本用于评估各种大语言模型（LLM）在 LoCoMo 数据集的问答任务上的表现。
支持的模型包括：
- OpenAI GPT 系列（gpt-3.5-turbo, gpt-4等）
- Anthropic Claude 系列
- Google Gemini 系列
- Hugging Face 开源模型（Gemma, LLaMA, Mistral等）

可选功能：
- 检索增强生成（RAG）模式
- 4-bit 量化加载（用于大型模型）
"""

import sys
from pathlib import Path
# 将项目根目录添加到系统路径，以便导入其他模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm  # 进度条显示
import argparse
# 导入API密钥设置函数
from global_methods import set_openai_key, set_anthropic_key, set_gemini_key
# 导入评估和统计分析函数
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
# 导入不同模型的答案生成函数
from task_eval.gpt_utils import get_gpt_answers
from task_eval.claude_utils import get_claude_answers
from task_eval.gemini_utils import get_gemini_answers
from task_eval.hf_llm_utils import init_hf_model, get_hf_answers

import numpy as np
import google.generativeai as genai

def parse_args():
    """
    解析命令行参数
    
    返回:
        args: 包含所有命令行参数的对象
        
    参数说明:
        --out-file: 输出结果文件路径（必需）
        --model: 要评估的模型名称（必需）
        --data-file: 输入数据文件路径（必需）
        --use-rag: 是否使用检索增强生成（RAG）模式
        --use-4bit: 是否使用4-bit量化加载模型（节省内存）
        --batch-size: 批处理大小，默认为1
        --rag-mode: RAG模式类型（如使用对话、观察或摘要作为数据库）
        --emb-dir: 嵌入向量目录路径（RAG模式下使用）
        --top-k: RAG检索时返回的top-k个最相关文档，默认为5
        --retriever: 检索器类型，默认为"contriever"
        --overwrite: 是否覆盖已有的预测结果
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()
    return args


def main():
    """
    主函数：执行问答任务评估的完整流程
    
    流程:
    1. 解析命令行参数
    2. 根据模型类型初始化相应的API或加载模型
    3. 加载数据集
    4. 对每个样本生成答案
    5. 评估答案质量（F1分数等）
    6. 保存结果并生成统计信息
    """
    # 获取命令行参数
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    # 根据模型类型设置相应的API密钥或初始化模型
    if 'gpt' in args.model:
        # OpenAI GPT系列模型：设置OpenAI API密钥
        set_openai_key()

    elif 'claude' in args.model:
        # Anthropic Claude系列模型：设置Anthropic API密钥
        set_anthropic_key()

    elif 'gemini' in args.model:
        # Google Gemini系列模型：设置Gemini API密钥
        set_gemini_key()
        if args.model == "gemini-pro-1.0":
            model_name = "models/gemini-1.0-pro-latest"

        # 初始化Gemini生成模型
        gemini_model = genai.GenerativeModel(model_name)
    
    elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
        # Hugging Face开源模型（Gemma、LLaMA、Mistral等）：初始化模型管道
        hf_pipeline, hf_model_name = init_hf_model(args)

    else:
        # 不支持的模型类型
        raise NotImplementedError


    # 加载对话数据集
    samples = json.load(open(args.data_file))
    
    # 构造预测结果的键名
    # 如果不使用RAG：键名格式为 "模型名_prediction"
    # 如果使用RAG：键名格式为 "模型名_RAG模式_top_K值_prediction"
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    
    # 构造模型标识键名（用于保存F1分数等指标）
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    
    # 如果输出文件已存在，加载已有的预测结果（用于增量更新或断点续传）
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}


    # 遍历数据集中的每个样本
    for data in samples:

        # 准备输出数据结构
        out_data = {'sample_id': data['sample_id']}
        
        # 如果该样本已有预测结果，则复用；否则使用原始QA数据
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()

        # 根据模型类型调用相应的答案生成函数
        if 'gpt' in args.model:
            # 使用GPT系列模型生成答案
            answers = get_gpt_answers(data, out_data, prediction_key, args)
        elif 'claude' in args.model:
            # 使用Claude系列模型生成答案
            answers = get_claude_answers(data, out_data, prediction_key, args)
        elif 'gemini' in args.model:
            # 使用Gemini系列模型生成答案
            answers = get_gemini_answers(gemini_model, data, out_data, prediction_key, args)
        elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
            # 使用Hugging Face开源模型生成答案
            answers = get_hf_answers(data, out_data, args, hf_pipeline, hf_model_name)
        else:
            raise NotImplementedError

        # 评估问答质量：计算F1分数、答案长度和召回率（RAG模式）
        # exact_matches: 每个问题的F1分数列表
        # lengths: 答案长度列表
        # recall: RAG模式下的召回率列表
        exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
        # 将评估指标添加到每个问答样本中
        for i in range(0, len(answers['qa'])):
            # 保存F1分数（保留3位小数）
            answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
            # 如果使用RAG模式且有召回率数据，则保存召回率
            if args.use_rag and len(recall) > 0:
                answers['qa'][i][model_key + '_recall'] = round(recall[i], 3)

        # 将处理后的样本添加到输出字典中
        out_samples[data['sample_id']] = answers


    # 将所有结果保存到输出文件（JSON格式，带缩进以便阅读）
    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)

    # 分析并保存聚合统计信息
    # 生成包含平均准确率、不同类别表现等的统计文件
    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace('.json', '_stats.json'),
                model_key, model_key + '_f1', rag=args.use_rag)
    # encoder=tiktoken.encoding_for_model(args.model))  # 注释掉的代码：可用于token计数


# 程序入口点
main()

