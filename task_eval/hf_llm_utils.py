"""
HuggingFace 大语言模型工具模块
==============================

本模块提供了与 HuggingFace 模型进行交互的核心功能，包括：
- 模型初始化和配置
- 问答任务的提示词构建
- 模型推理和答案生成
- 上下文窗口管理

支持的模型系列：
- LLaMA 系列（2/3代，7B/70B，基础版和Chat版）
- Mistral 系列（7B，基础版和Instruct版，不同上下文长度）
- Gemma 系列（7B-IT）

主要功能：
1. init_hf_model(): 初始化指定的 HuggingFace 模型
2. get_hf_answers(): 批量生成问答任务的答案
3. get_input_context(): 构建符合模型上下文窗口的输入
4. run_mistral/run_gemma/run_llama(): 特定模型的执行函数
"""

# 添加父目录到系统路径，以便导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 标准库导入
import random
import os, json
from tqdm import tqdm

# HuggingFace 相关导入
from transformers import AutoTokenizer
import transformers
import torch
import huggingface_hub

from transformers import (
    AutoTokenizer,           # 用于加载模型的分词器
    AutoModelForCausalLM,    # 用于加载因果语言模型
    BitsAndBytesConfig,      # 用于配置量化参数（4-bit/8-bit）
)
import torch
import huggingface_hub
from task_eval.rag_utils import prepare_for_rag, get_rag_context  # 复用RAG准备与上下文检索


# 各模型的最大上下文窗口长度（单位：tokens）
# 用于在构建输入时确保不超过模型的处理能力
MAX_LENGTH={'llama2': 4096,                      # LLaMA 2 基础版 7B
            'llama2-70b': 4096,                  # LLaMA 2 基础版 70B
            'llama2-chat': 4096,                 # LLaMA 2 Chat 版 7B
            'llama2-chat-70b': 4096,             # LLaMA 2 Chat 版 70B
            'llama3-chat-70b': 4096,             # LLaMA 3 Chat 版 70B
            'gpt-3.5-turbo-16k': 16000,          # GPT-3.5 Turbo 16k 上下文版本
            'gpt-3.5-turbo': 4096,               # GPT-3.5 Turbo 标准版
            'gemma-7b-it': 8000,                 # Google Gemma 7B Instruct
            'mistral-7b-128k': 128000,           # Mistral 7B 超长上下文版
            'mistral-7b-4k': 4096,               # Mistral 7B 4k 上下文
            'mistral-7b-8k': 8000,               # Mistral 7B 8k 上下文
            'mistral-instruct-7b-4k': 4096,      # Mistral Instruct 4k 上下文
            'mistral-instruct-7b-8k': 8000,      # Mistral Instruct 8k 上下文
            'mistral-instruct-7b-32k-v2': 8000,  # Mistral Instruct v0.2（实际使用8k）
            'mistral-instruct-7b-8k-new': 8000,  # Mistral Instruct 新版 8k
            'mistral-instruct-7b-32k': 32000,    # Mistral Instruct 32k 上下文
            'mistral-instruct-7b-128k': 128000}  # Mistral Instruct 超长上下文版


# 单个问题的提示词模板
# 要求模型基于对话历史，用简短的词语回答问题
# 强调尽可能使用对话中的原文
QA_PROMPT = """
Based on the above conversations, write a short answer for the following question in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

Question: {}
"""

# 批量问题的提示词模板（已注释，未使用）
# 原本设计为返回字符串列表的 JSON 格式
# QA_PROMPT_BATCH = """
# Based on the above conversations, answer the following questions in a few words. Write the answers as a list of strings in the json format. Start and end with a square bracket.

# """

# 批量问题的提示词模板（当前使用版本）
# 要求模型以 JSON 字典格式返回答案
# 格式：{"1": "answer1", "2": "answer2", ...}
QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the question number as 'key' and the short answer as value. Answer with exact words from the conversations whenever possible.

"""

# LLaMA 2 Chat 模型的系统提示词模板
# 使用 LLaMA 2 特定的对话格式：<s>[INST] <<SYS>> ... <</SYS>> ... [/INST]
# 定义了助手的角色和行为准则
LLAMA2_CHAT_SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation.
If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST]
"""


# LLaMA 3 Chat 模型的系统提示词模板
# 格式与 LLaMA 2 相同，但适配 LLaMA 3 的行为特性
LLAMA3_CHAT_SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation.
If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST]
"""


# Mistral Instruct 模型的系统提示词模板
# 使用 Mistral 特定的简洁格式：<s>[INST] ... [/INST]
MISTRAL_INSTRUCT_SYSTEM_PROMPT = """
<s>[INST] {} [/INST]
"""

# Google Gemma Instruct 模型的提示词模板
# 使用 Gemma 特定的对话标记：<bos><start_of_turn>user ... <end_of_turn>
GEMMA_INSTRUCT_PROMPT = """
<bos><start_of_turn>user
{}<end_of_turn>
"""

# 对话开始的提示词，用于介绍对话的背景信息
# 说明对话双方的名字和对话跨越多天的情况
CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

# 每个问题预计的答案 token 数量
# 用于计算批量处理时的 max_new_tokens 参数
ANS_TOKENS_PER_QUES = 50


def run_mistral(pipeline, question, data, tokenizer, args):
    """
    运行 Mistral 模型生成答案
    
    参数：
        pipeline: HuggingFace 的文本生成 pipeline
        question: 要回答的问题（字符串）
        data: 包含对话历史的数据字典，需要有 'conversation' 键
        tokenizer: 模型对应的分词器
        args: 命令行参数对象，包含 batch_size 等配置
    
    返回：
        生成的答案文本（字符串）
    
    工作流程：
        1. 使用 QA_PROMPT 格式化问题
        2. 获取适配上下文窗口的对话历史
        3. 应用 Mistral 的聊天模板格式
        4. 使用 pipeline 生成答案
    """
    # 格式化问题提示词
    question_prompt =  QA_PROMPT.format(question)
    
    # 获取符合上下文窗口限制的对话历史
    query_conv = get_input_context(data['conversation'], MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(question_prompt), tokenizer, args)

    # 旧方法：不使用 chat_template（已注释）
    # query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
    
    # 新方法：使用 tokenizer 的 chat_template 功能
    # 自动应用模型特定的对话格式
    query = tokenizer.apply_chat_template([{"role": "user", "content": query_conv + '\n\n' + question_prompt}], tokenize=False, add_generation_prompt=True)

    # 使用 pipeline 生成文本
    sequences = pipeline(
                        query,
                        # max_length=8000,  # 已注释，使用 max_new_tokens 代替
                        max_new_tokens=args.batch_size*ANS_TOKENS_PER_QUES,  # 根据批次大小计算最大新 token 数
                        pad_token_id=tokenizer.pad_token_id,                  # 填充 token ID
                        eos_token_id=tokenizer.eos_token_id,                  # 结束 token ID
                        do_sample=True,                                       # 启用采样（而非贪婪解码）
                        top_k=10,                                             # Top-K 采样参数
                        temperature=0.4,                                      # 温度参数（较低值使输出更确定）
                        top_p=0.9,                                            # Nucleus 采样参数
                        return_full_text=False,                               # 只返回生成的新文本
                        num_return_sequences=1,                               # 返回 1 个序列
                        )
    return sequences[0]['generated_text']


def run_gemma(pipeline, question, data, tokenizer, args):
    """
    运行 Google Gemma 模型生成答案
    
    参数：
        pipeline: HuggingFace 的文本生成 pipeline
        question: 要回答的问题（字符串）
        data: 包含对话历史的数据字典，需要有 'conversation' 键
        tokenizer: 模型对应的分词器
        args: 命令行参数对象，包含 batch_size 等配置
    
    返回：
        生成的答案文本（字符串）
    
    注意：
        与 run_mistral 类似，但使用 Gemma 特定的提示词格式
    """
    # 格式化问题提示词
    question_prompt =  QA_PROMPT.format(question)
    
    # 获取符合上下文窗口限制的对话历史
    query_conv = get_input_context(data['conversation'], GEMMA_INSTRUCT_PROMPT.format(question_prompt), tokenizer, args)

    # 旧方法：不使用 chat_template（已注释）
    # query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
    
    # 新方法：使用 Gemma 的 chat_template
    query = tokenizer.apply_chat_template([{"role": "user", "content": query_conv + '\n\n' + question_prompt}], tokenize=False, add_generation_prompt=True)

    # 生成答案，参数设置与 Mistral 相同
    sequences = pipeline(
                        query,
                        # max_length=8000,
                        max_new_tokens=args.batch_size*ANS_TOKENS_PER_QUES,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=10,
                        temperature=0.4,
                        top_p=0.9,
                        return_full_text=False,
                        num_return_sequences=1,
                        )
    return sequences[0]['generated_text']


def run_llama(pipeline, question, data, tokenizer, args):
    """
    运行 LLaMA 模型（2/3代）生成答案
    
    参数：
        pipeline: HuggingFace 的文本生成 pipeline
        question: 要回答的问题（字符串）
        data: 包含对话历史的数据字典，需要有 'conversation' 键
        tokenizer: 模型对应的分词器
        args: 命令行参数对象，包含 batch_size 等配置
    
    返回：
        生成的答案文本（字符串）
    
    注意：
        LLaMA 模型使用两条消息的对话格式：
        1. system 角色：定义助手的行为
        2. user 角色：提供对话历史和问题
    """
    # 格式化问题提示词
    question_prompt =  QA_PROMPT.format(question)
    
    # 获取符合上下文窗口限制的对话历史
    query_conv = get_input_context(data['conversation'], LLAMA3_CHAT_SYSTEM_PROMPT.format(question_prompt), tokenizer, args)

    # 旧方法：不使用 chat_template（已注释）
    # query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
    
    # 新方法：使用 LLaMA 的 chat_template，包含 system 和 user 两个角色
    query = tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation. If you don't know the answer to a question, please don't share false information."},
                                           {"role": "user", "content": query_conv + '\n\n' + question_prompt}], tokenize=False, add_generation_prompt=True)

    # 生成答案，参数设置与其他模型相同
    sequences = pipeline(
                        query,
                        # max_length=8000,
                        max_new_tokens=args.batch_size*ANS_TOKENS_PER_QUES,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=10,
                        temperature=0.4,
                        top_p=0.9,
                        return_full_text=False,
                        num_return_sequences=1,
                        )
    return sequences[0]['generated_text']


def get_chatgpt_summaries(ann_file):
    """
    获取 ChatGPT 生成的对话摘要（功能未完整实现）
    
    参数：
        ann_file: 标注文件的路径
    
    注意：
        此函数似乎是为了提取对话历史的辅助函数，
        但当前实现不完整（没有返回值）
    """
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    conv = ''
    for i in range(1,20):
        if 'session_%s' % i in data:
            conv = conv + data['session_%s_date_time' % i] + '\n'
            for dialog in data['session_%s' % i]:
                conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'


def get_input_context(data, question_prompt, encoding, args):
    """
    构建符合模型上下文窗口限制的输入文本
    
    参数：
        data: 包含多个对话 session 的数据字典
              格式：{'session_1': [...], 'session_1_date_time': '...', ...}
        question_prompt: 已格式化的问题提示词
        encoding: 模型的分词器（用于计算 token 数量）
        args: 命令行参数对象，包含 model 等配置
    
    返回：
        query_conv: 裁剪后的对话历史文本
        min_session: 包含的最早 session 编号（从1开始）
    
    工作原理：
        1. 计算问题提示词的 token 数
        2. 计算开始提示词的 token 数
        3. 从最新的 session 开始，逐个添加对话历史
        4. 当总 token 数接近模型最大长度时停止
        5. 确保为答案生成预留足够空间
    """
    # 计算问题提示词的 token 数量
    question_tokens = len(encoding.encode(question_prompt))

    # 构建并计算开始提示词的 token 数量
    # 提取对话双方的名字
    speakers_names = list(set([d['speaker'] for d in data['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = len(encoding.encode(start_prompt))

    # 初始化变量
    query_conv = ''           # 最终的对话历史文本
    total_tokens = 0          # 已累积的 token 数量
    min_session = -1          # 记录包含的最早 session
    stop = False              # 标记是否已达到上下文限制
    
    # 提取所有 session 的编号
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    
    # 从最早的 session 开始遍历（实际按时间顺序）
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in data:
            # 逆序遍历当前 session 的对话（从最新到最旧）
            # 这样可以优先保留最新的对话内容
            for dialog in data['session_%s' % i][::-1]:
                # 构建单轮对话的文本
                turn = ''
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                # 如果有分享图片，添加图片描述
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'

                # 估算添加这一轮对话后的总 token 数
                # 包括日期、对话标签等格式化文本
                new_tokens = len(encoding.encode('DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + turn))
                
                # 检查是否还有足够空间容纳这一轮对话
                # 需要确保：开始提示词 + 新对话 + 已有对话 + 问题 + 答案空间 < 最大长度
                if (start_tokens + new_tokens + total_tokens + question_tokens) < (MAX_LENGTH[args.model]-(ANS_TOKENS_PER_QUES*args.batch_size)):
                    # 将新对话添加到开头（因为是逆序遍历）
                    query_conv = turn + query_conv
                    total_tokens += len(encoding.encode(turn))
                else:
                    # 达到上下文限制，记录最早 session 并停止
                    min_session = i
                    stop = True
                    break

            # 为当前 session 添加日期和标签
            query_conv = '\nDATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        
        # 如果已达到上下文限制，停止添加更多 session
        if stop:
            break
    
    # 在开头添加对话介绍提示词
    query_conv = start_prompt + query_conv

    return query_conv


def get_hf_answers(in_data, out_data, args, pipeline, model_name):
    """
    使用 HuggingFace 模型批量生成问答任务的答案
    
    参数：
        in_data: 输入数据字典，包含 'qa' 键（问答列表）和 'conversation' 键（对话历史）
        out_data: 输出数据字典，用于存储预测结果
        args: 命令行参数对象，包含 model、batch_size、overwrite 等配置
        pipeline: HuggingFace 的文本生成 pipeline
        model_name: 模型的完整名称（用于加载分词器）
    
    返回：
        out_data: 更新后的输出数据，包含模型的预测答案
    
    工作流程：
        1. 按批次处理问题（batch_size 个问题一组）
        2. 对特定类别的问题进行预处理（时间类、对抗类）
        3. 根据模型类型选择相应的执行函数
        4. 解析模型输出并存储预测结果
        5. 跳过已有预测的问题（除非 overwrite=True）
    """
    # 计算与 evaluate_qa.py 一致的预测键名，确保评估阶段能找到对应字段
    prediction_key = (
        f"{args.model}_prediction" if not args.use_rag else f"{args.model}_{args.rag_mode}_top_{args.top_k}_prediction"
    )

    # 加载模型对应的分词器
    # 注意：这里 if-else 的两个分支目前是相同的
    if 'mistral' in model_name:
        encoding = AutoTokenizer.from_pretrained(model_name)
    else:
        encoding = AutoTokenizer.from_pretrained(model_name)

    # 如果启用 RAG，则准备检索库与问题向量（与 GPT 流程保持一致）
    if args.use_rag:
        assert args.batch_size == 1, "RAG 模式下当前仅支持 batch_size=1"
        context_database, query_vectors = prepare_for_rag(args, in_data)
    else:
        context_database, query_vectors = None, None

    # 按批次遍历所有问题
    for batch_start_idx in range(0, len(in_data['qa']) + args.batch_size, args.batch_size):

        # 初始化批次相关的变量
        questions = []          # 当前批次的问题列表
        include_idxs = []       # 需要预测的问题索引
        cat_5_idxs = []        # 对抗类问题（category 5）的索引
        cat_5_answers = []     # 对抗类问题的答案选项
        
        # 收集当前批次的问题
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            # 如果所有问题都已处理完毕，结束循环
            if i>=len(in_data['qa']):
                break
            qa = in_data['qa'][i]
            
            # 跳过已有预测的问题（除非设置了 overwrite）
            if '%s_prediction' % args.model not in qa or args.overwrite:
                include_idxs.append(i)
            else:
                print("Skipping -->", qa['question'])
                continue

            # 对特定类别的问题进行预处理
            # Category 2: 时间相关问题 - 提示模型使用对话日期
            if qa['category'] == 2:
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')
            
            # Category 5: 对抗性问题 - 将问题转换为选择题格式
            # 随机打乱正确答案和 "No information available" 的顺序
            # 注意：对抗性问题使用 'adversarial_answer' 字段
            elif qa['category'] == 5:
                question = qa['question'] + " (a) {} (b) {}. Select the correct answer by writing (a) or (b)."
                if random.random() < 0.5:
                    # 正确答案在选项 (b)
                    question = question.format('No information available', qa['adversarial_answer'])
                    answer = {'a': 'No information available', 'b': qa['adversarial_answer']}
                else:
                    # 正确答案在选项 (a)
                    question = question.format(qa['adversarial_answer'], 'No information available')
                    answer = {'b': 'No information available', 'a': qa['adversarial_answer']}
                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
                # 旧方法（已注释）：直接询问是否可回答
                # questions.append(qa['question'] + " Write NOT ANSWERABLE if the question cannot be answered.")
            
            # 其他类别：直接使用原问题
            else:
                questions.append(qa['question'])

        # 如果当前批次没有问题需要处理，跳过
        if questions == []:
            continue


        # 单问题处理模式（batch_size = 1）
        if args.batch_size == 1:

            # RAG 模式：使用检索到的上下文直接构建 query；非 RAG 模式：调用各模型专用函数
            if args.use_rag:
                # 获取检索上下文
                query_conv, context_ids = get_rag_context(context_database, query_vectors[include_idxs][0], args)

                # 使用当前模型的 chat_template 构造输入
                question_prompt = QA_PROMPT.format(questions[0])
                tokenizer = encoding

                if 'mistral' in model_name:
                    query = tokenizer.apply_chat_template(
                        [{"role": "user", "content": query_conv + '\n\n' + question_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                elif 'gemma' in model_name:
                    query = tokenizer.apply_chat_template(
                        [{"role": "user", "content": query_conv + '\n\n' + question_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                elif 'llama' in model_name:
                    query = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation. If you don't know the answer to a question, please don't share false information."},
                            {"role": "user", "content": query_conv + '\n\n' + question_prompt},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    raise NotImplementedError

                sequences = pipeline(
                    query,
                    max_new_tokens=args.batch_size * ANS_TOKENS_PER_QUES,
                    pad_token_id=encoding.pad_token_id,
                    eos_token_id=encoding.eos_token_id,
                    do_sample=True,
                    top_k=10,
                    temperature=0.4,
                    top_p=0.9,
                    return_full_text=False,
                    num_return_sequences=1,
                )
                answer = sequences[0]["generated_text"]
            else:
                # 根据模型类型选择相应的执行函数
                if 'mistral' in model_name:
                    answer = run_mistral(pipeline, questions[0], in_data, encoding, args)
                elif 'llama' in model_name:
                    answer = run_llama(pipeline, questions[0], in_data, encoding, args)
                elif 'gemma' in model_name:
                    answer = run_gemma(pipeline, questions[0], in_data, encoding, args)
                else:
                    raise NotImplementedError

            # 打印问题和答案（用于调试）
            print(questions[0], answer)

            # 后处理答案
            # 1. 清理转义字符和空格
            answer = answer.replace('\\"', "'").strip()
            # 2. 提取第一行非空内容（模型可能生成多行）
            answer = [w.strip() for w in answer.split('\n') if not w.strip().isspace()][0]
            
            # 3. 特殊处理对抗性问题（category 5）
            if len(cat_5_idxs) > 0:
                answer = answer.lower().strip()
                # 根据模型选择的选项 (a) 或 (b) 提取实际答案
                if '(a)' in answer:
                    answer = cat_5_answers[0]['a']
                else:
                    answer = cat_5_answers[0]['b']
            
            # 4. 其他问题：移除选项标记和前缀
            else:
                answer = answer.lower().replace('(a)', '').replace('(b)', '').replace('a)', '').replace('b)', '').replace('answer:', '').strip()
            
            # 将预测答案存储到输出数据中（键名需与评估阶段一致）
            out_data['qa'][batch_start_idx][prediction_key] = answer
            # 若启用 RAG，同时记录检索到的上下文 ID，便于召回率评估
            if args.use_rag:
                out_data['qa'][batch_start_idx][prediction_key + '_context'] = context_ids

        # 批量处理模式（batch_size > 1）
        # 目前未实现，会抛出错误
        else:            
            raise NotImplementedError

    return out_data


def init_hf_model(args):
    """
    初始化 HuggingFace 模型并返回 pipeline
    
    参数：
        args: 命令行参数对象，需要包含以下属性：
            - model: 模型简称（如 'mistral-7b-8k', 'llama2-chat' 等）
            - use_4bit: 是否使用 4-bit 量化推理
    
    返回：
        pipeline: HuggingFace 文本生成 pipeline
        model_name: 模型的完整 HuggingFace Hub 名称
    
    支持的模型：
        - LLaMA 2 系列：7B/70B，基础版和 Chat 版
        - LLaMA 3 系列：70B Chat 版
        - Mistral 系列：7B，基础版和 Instruct 版（多种上下文长度配置）
        - Gemma 系列：7B Instruct 版
    
    工作流程：
        1. 根据模型简称映射到完整的 HuggingFace Hub 名称
        2. 尝试使用 HF_TOKEN 登录（如果环境变量中存在）
        3. 根据 use_4bit 参数选择加载方式（量化或全精度）
        4. 创建并返回文本生成 pipeline
    """
    # 根据模型简称映射到 HuggingFace Hub 上的完整模型名称
    if args.model == 'llama2':
        model_name = "meta-llama/Llama-2-7b-hf"

    elif args.model == 'llama2-70b':
        model_name = "meta-llama/Llama-2-70b-hf"

    elif args.model == 'llama2-chat':
        model_name = "meta-llama/Llama-2-7b-chat-hf"

    elif args.model == 'llama2-chat-70b':
        model_name = "meta-llama/Llama-2-70b-chat-hf"

    elif args.model == 'llama3-chat-70b':
        model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

    # Mistral 基础版（不同上下文长度配置使用相同的模型）
    elif args.model in ['mistral-7b-128k', 'mistral-7b-4k', 'mistral-7b-8k']:
        model_name = "mistralai/Mistral-7B-v0.1"

    # Mistral Instruct v0.1 版本
    elif args.model in ['mistral-instruct-7b-128k', 'mistral-instruct-7b-8k', 'mistral-instruct-7b-12k']:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    elif args.model in ['mistral-instruct-7b-8k-new']:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Mistral Instruct v0.2 版本
    elif args.model in ['mistral-instruct-7b-32k-v2']:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Google Gemma Instruct 版本
    elif args.model in ['gemma-7b-it']:
        model_name = 'google/gemma-7b-it'

    # 通用 Mistral 模型（通过名称匹配）
    elif 'mistral' in args.model.lower():
        model_name = 'mistralai/' + args.model

    # 不支持的模型
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # 获取 HF Token（如果存在）
    # 对于开放模型（如 Mistral），token 是可选的
    # 对于受限模型（如 LLaMA、Gemma），需要通过申请获取访问权限和 token
    hf_token = os.environ.get('HF_TOKEN', None)
    
    # 只有在提供了 token 时才登录
    if hf_token:
        huggingface_hub.login(hf_token)
        print("Logged in to HuggingFace Hub with token")
    else:
        print("No HF_TOKEN provided, proceeding without authentication")

    # 根据量化选项加载模型
    if args.use_4bit:
        # 使用 4-bit 量化推理（节省显存，适合大模型）
        print("Using 4-bit inference")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        # 设置填充 token 为结束 token（用于开放式生成）
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # 配置量化参数
        if 'gemma' in args.model:
            # Gemma 模型使用简单的 4-bit 量化配置
            bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)
        else:
            # 其他模型使用更复杂的量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                      # 启用 4-bit 量化
                bnb_4bit_quant_type="nf4",              # 使用 NormalFloat4 量化类型
                bnb_4bit_compute_dtype=torch.float16,   # 计算使用 FP16
                bnb_4bit_use_double_quant=True,         # 启用双重量化（进一步节省显存）
            )

        # 加载模型（根据模型类型选择不同的配置）
        if 'mistralai' in model_name:
            if 'v0.1' in model_name:
                # Mistral v0.1 使用 Flash Attention 2 加速
                model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                            torch_dtype=torch.float16, 
                                                            attn_implementation="flash_attention_2",
                                                            quantization_config=bnb_config,
                                                            device_map="auto",           # 自动分配设备
                                                            trust_remote_code=True,)      # 信任远程代码
            else:
                # Mistral 其他版本使用标准配置
                model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            quantization_config=bnb_config,
                                                            device_map="auto",
                                                            trust_remote_code=True)
        
        else:
            # 其他模型（LLaMA、Gemma 等）使用标准配置
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            torch_dtype=torch.float16,
                                            quantization_config=bnb_config,
                                            device_map="auto",
                                            trust_remote_code=True,)

        # 创建文本生成 pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",    # 自动查找并使用 GPU
        )
    
    else:
        # 不使用量化，直接加载 FP16 模型
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # pipeline = None  # 已注释的旧代码
    
    print("Loaded model")
    return pipeline, model_name

