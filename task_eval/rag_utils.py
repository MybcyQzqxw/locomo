import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import os, json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from global_methods import get_openai_embedding, set_openai_key, run_chatgpt_with_examples



def save_eval(data_file, accs, key='exact_match'):

    
    if os.path.exists(data_file.replace('.json', '_scores.json')):
        with open(data_file.replace('.json', '_scores.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    assert len(data['qa']) == len(accs), (len(data['qa']), len(accs), accs)
    for i in range(0, len(data['qa'])):
        data['qa'][i][key] = accs[i]
    
    with open(data_file.replace('.json', '_scores.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def init_context_model(retriever):

    if retriever == 'dpr':
        from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
        context_model.eval()
        return context_tokenizer, context_model

    elif retriever == 'contriever':

        from transformers import AutoTokenizer, AutoModel
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        context_model = AutoModel.from_pretrained('facebook/contriever').cuda()
        context_model.eval()
        return context_tokenizer, context_model

    elif retriever == 'dragon':

        from transformers import AutoTokenizer, AutoModel
        # Dragon 的上下文侧应使用 context-encoder 对应的 tokenizer
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-context-encoder')
        # 使用 use_safetensors=True 强制加载 safetensors 格式，避免 torch.load 安全问题
        context_model = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder', use_safetensors=True).cuda()
        return context_tokenizer, context_model

    elif retriever == 'openai':

        set_openai_key()
        return None, None
    
    else:
        raise ValueError
    
def init_query_model(retriever):

    if retriever == 'dpr':
        from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()
        question_model.eval()
        return question_tokenizer, question_model

    elif retriever == 'contriever':

        from transformers import AutoTokenizer, AutoModel
        # 使用与上下文相同的模型，但需要在本作用域内显式初始化 tokenizer
        question_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        question_model = AutoModel.from_pretrained('facebook/contriever').cuda()
        question_model.eval()
        return question_tokenizer, question_model

    elif retriever == 'dragon':

        from transformers import AutoTokenizer, AutoModel
        # Dragon 使用不同的 query/context 编码器，这里初始化查询侧的 tokenizer/model
        question_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        # 使用 use_safetensors=True 强制加载 safetensors 格式
        question_model = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder', use_safetensors=True).cuda()
        return question_tokenizer, question_model

    elif retriever == 'openai':

        set_openai_key()
        return None, None
    
    else:
        raise ValueError


def get_embeddings(retriever, inputs, mode='context'):

    if mode == 'context':
        tokenizer, encoder = init_context_model(retriever)
    else:
        tokenizer, encoder = init_query_model(retriever)
    
    all_embeddings = []
    batch_size = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            # print(input_ids.shape)
            if retriever == 'dpr':
                input_ids = tokenizer(inputs[i:(i+batch_size)], return_tensors="pt", padding=True)["input_ids"].cuda()
                embeddings = encoder(input_ids).pooler_output.detach()
                # print(embeddings.shape)
                all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
            elif retriever == 'contriever':
                # Compute token embeddings
                ctx_input = tokenizer(inputs[i:(i+batch_size)], padding=True, truncation=True, return_tensors='pt')
                # move to device of encoder
                ctx_input = {k: v.to(device) for k, v in ctx_input.items()}
                outputs = encoder(**ctx_input)
                embeddings = mean_pooling(outputs[0], ctx_input['attention_mask'])
                all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
            elif retriever == 'dragon':
                ctx_input = tokenizer(inputs[i:(i+batch_size)], padding=True, truncation=True, return_tensors='pt').to(device)
                embeddings = encoder(**ctx_input).last_hidden_state[:, 0, :]
                # all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                all_embeddings.append(embeddings)
            elif retriever == 'openai':
                all_embeddings.append(torch.tensor(get_openai_embedding(inputs)))
            else:
                raise ValueError

    return torch.cat(all_embeddings, dim=0).cpu().numpy()


def get_context_embeddings(retriever, data, context_tokenizer, context_encoder, captions=None):

    context_embeddings = []
    context_ids = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for i in tqdm(range(1,20), desc="Getting context encodings"):
        contexts = []
        if 'session_%s' % i in data:
            date_time_string = data['session_%s_date_time' % i]
            for dialog in data['session_%s' % i]:

                turn = ''
                # conv = conv + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                try:
                    turn = dialog['speaker'] + ' said, \"' + dialog['compressed_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['compressed_text'] + '\n'
                except KeyError:
                    turn = dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'
                if "img_file" in dialog and len(dialog["img_file"]) > 0:
                    turn += '[shares %s]\n' % dialog["blip_caption"]
                contexts.append('(' + date_time_string + ') ' + turn)

                context_ids.append(dialog["dia_id"])
            with torch.no_grad():
                # print(input_ids.shape)
                if retriever == 'dpr':
                    input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                    embeddings = context_encoder(input_ids).pooler_output.detach()
                    # print(embeddings.shape)
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif retriever == 'contriever':
                    # Compute token embeddings
                    inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
                    # move to device of encoder
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    # input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                    outputs = context_encoder(**inputs)
                    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif retriever == 'dragon':
                    ctx_input = context_tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
                    embeddings = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif retriever == 'openai':
                    context_embeddings.append(torch.tensor(get_openai_embedding(contexts)))
                else:
                    raise ValueError

    # print(context_embeddings[0].shape[0])
    context_embeddings = torch.cat(context_embeddings, dim=0)
    # print(context_embeddings.shape[0])

    return context_ids, context_embeddings


def prepare_for_rag(args, data):
    """
    准备 RAG 所需的检索库与查询向量（与生成模型无关，可复用于 GPT/HF 等任意模型）。

    返回：
        database: dict，包含 embeddings、date_time、dia_id、context
        question_embeddings: np.ndarray，问题文本的向量
    """
    dataset_prefix = os.path.splitext(os.path.split(args.data_file)[-1])[0]

    if args.rag_mode == 'dialog':
        pkl_path = os.path.join(args.emb_dir, f"{dataset_prefix}_dialog_{data['sample_id']}.pkl")
        if not os.path.exists(pkl_path):
            dialogs = []
            date_times = []
            context_ids = []
            session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if 'session' in k and 'date_time' not in k]
            for i in range(min(session_nums), max(session_nums) + 1):
                date_time = data['conversation'][f'session_{i}_date_time']
                for dialog in data['conversation'][f'session_{i}']:
                    context_ids.append(dialog['dia_id'])
                    date_times.append(date_time)
                    if 'blip_caption' in dialog:
                        dialogs.append(dialog['speaker'] + ' said, "' + dialog['text'] + '"' + ' and shared ' + dialog['blip_caption'])
                    else:
                        dialogs.append(dialog['speaker'] + ' said, "' + dialog['text'] + '"')

            print(f"Getting embeddings for {len(dialogs)} dialogs")
            embeddings = get_embeddings(args.retriever, dialogs, 'context')
            assert embeddings.shape[0] == len(dialogs), "Lengths of embeddings and dialogs do not match"
            database = {
                'embeddings': embeddings,
                'date_time': date_times,
                'dia_id': context_ids,
                'context': dialogs,
            }

            # 确保输出目录存在
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump(database, f)
        else:
            database = pickle.load(open(pkl_path, 'rb'))

    elif args.rag_mode == "summary":
        pkl_path = os.path.join(args.emb_dir, f"{dataset_prefix}_session_summary_{data['sample_id']}.pkl")
        if not os.path.exists(pkl_path):
            # 从 data['session_summary'] 中提取摘要并生成嵌入
            summaries = []
            date_times = []
            context_ids = []
            
            session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if 'session' in k and 'date_time' not in k]
            for i in range(min(session_nums), max(session_nums) + 1):
                # 从 session_summary 字典中提取已有的摘要
                summary_key = f'session_{i}_summary'
                if 'session_summary' in data and summary_key in data['session_summary']:
                    summary = data['session_summary'][summary_key]
                else:
                    raise ValueError(f"Missing {summary_key} in data['session_summary'] for {data['sample_id']}")
                
                # 提取日期时间
                date_time = data['conversation'][f'session_{i}_date_time']
                
                # 添加到数组
                summaries.append(summary)
                date_times.append(date_time)
                context_ids.append(f'S{i}')
            
            # 检查数组长度是否一致
            if not (len(summaries) == len(date_times) == len(context_ids)):
                raise ValueError(
                    f"Data length mismatch for {data['sample_id']}: "
                    f"summaries={len(summaries)}, date_times={len(date_times)}, "
                    f"context_ids={len(context_ids)}. Some session data may be missing."
                )
            
            print(f"Getting embeddings for {len(summaries)} summaries")
            embeddings = get_embeddings(args.retriever, summaries, 'context')
            assert embeddings.shape[0] == len(summaries), "Lengths of embeddings and summaries do not match"
            
            database = {
                'embeddings': embeddings,
                'date_time': date_times,
                'dia_id': context_ids,
                'context': summaries,
            }
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump(database, f)
        else:
            database = pickle.load(open(pkl_path, 'rb'))

    elif args.rag_mode == 'observation':
        pkl_path = os.path.join(args.emb_dir, f"{dataset_prefix}_observation_{data['sample_id']}.pkl")
        if not os.path.exists(pkl_path):
            # 从 data['observation'] 中提取观察并生成嵌入
            observations = []
            date_times = []
            context_ids = []
            
            # 遍历所有 session 的 observation
            for session_key in sorted(data['observation'].keys()):
                # 从 session_key 提取对应的时间
                # 例如: 'session_1_observation' -> 'session_1_date_time'
                session_time_key = session_key.replace(
                    '_observation', '_date_time'
                )
                date_time = data['conversation'].get(
                    session_time_key, "Unknown"
                )
                
                # 遍历该 session 中每个人物的观察
                session_data = data['observation'][session_key]
                for person_name, obs_list in session_data.items():
                    # obs_list 是 [[observation_text, dialog_id], ...]
                    for obs_text, dia_id in obs_list:
                        observations.append(obs_text)
                        context_ids.append(dia_id)
                        date_times.append(date_time)
            
            # 检查数组长度是否一致
            if not (len(observations) == len(date_times) ==
                    len(context_ids)):
                raise ValueError(
                    f"Data length mismatch for {data['sample_id']}: "
                    f"observations={len(observations)}, "
                    f"date_times={len(date_times)}, "
                    f"context_ids={len(context_ids)}. "
                    f"Some observation data may be missing."
                )
            
            print(f"Getting embeddings for {len(observations)} "
                  f"observations")
            embeddings = get_embeddings(
                args.retriever, observations, 'context'
            )
            assert embeddings.shape[0] == len(observations), \
                "Lengths of embeddings and observations do not match"
            
            database = {
                'embeddings': embeddings,
                'date_time': date_times,
                'dia_id': context_ids,
                'context': observations,
            }
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump(database, f)
        else:
            database = pickle.load(open(pkl_path, 'rb'))

    else:
        raise ValueError

    print(f"Getting embeddings for {len(data['qa'])} questions")
    question_embeddings = get_embeddings(args.retriever, [q['question'] for q in data['qa']], 'query')
    return database, question_embeddings


def get_rag_context(context_database, query_vector, args):
    """
    基于向量相似度检索 Top-K 上下文，并返回可直接拼接到提示中的文本与对应证据 ID。
    """
    output = np.dot(query_vector, context_database['embeddings'].T)
    sorted_outputs = np.argsort(output)[::-1]
    sorted_context = [context_database['context'][idx] for idx in sorted_outputs[:args.top_k]]

    sorted_context_ids = []
    for idx in sorted_outputs[:args.top_k]:
        context_id = context_database['dia_id'][idx]
        if isinstance(context_id, str) and ',' in context_id:
            context_id = [s.strip() for s in context_id.split(',')]
        if isinstance(context_id, list):
            sorted_context_ids.extend(context_id)
        else:
            sorted_context_ids.append(context_id)

    sorted_date_times = [context_database['date_time'][idx] for idx in sorted_outputs[:args.top_k]]
    if args.rag_mode in ['dialog', 'observation']:
        query_context = '\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])
    else:
        query_context = '\n\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])

    return query_context, sorted_context_ids
