# **ACL 2024** 论文 "**Evaluating Very Long-Term Conversational Memory of LLM Agents**" 的数据与代码

**作者**: [Adyasha Maharana](https://adymaharana.github.io/), [Dong-Ho Lee](https://www.danny-lee.info/), [Sergey Tulyakov](https://stulyakov.com/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), [Francesco Barbieri](https://fvancesco.github.io/) 和 [Yuwei Fang](https://yuwfan.github.io/)

**论文**: [pdf](https://github.com/snap-research/locomo/tree/main/static/paper/locomo.pdf)

[English](README.MD) | 简体中文

## 目录

- [数据](#数据)
- [代码](#代码)
  - [生成长期对话](#生成长期对话)
  - [评估LLM](#评估llm)
  - [生成观察与会话摘要](#生成观察与会话摘要)
  - [评估RAG模型](#评估rag模型)
- [引用](#引用)

## 数据

我们发布了 LoCoMo，这是一个由*超长期*对话数据组成的高质量评估基准。该基准包含10个对话。每个对话都针对**问答**和**事件摘要**任务进行了标注。此外，每个对话中的对话内容还可用于**多模态对话生成**任务。请参见下表中的数据集统计信息。

![image](./static/images/locomo_example_stats.png)

数据集可在本仓库的 `./data/locomo10.json` 文件中找到。每个样本代表一个完整的对话及其相应的标注：

* `sample_id`: 样本标识符
* `conversation`: 
    * 会话列表（`session_<num>`）及其时间戳（`session_<num>_date_time`）。数字 `<num>` 表示会话的时间顺序。
    * 还包括两位说话者的名字，即 `speaker_a` 和 `speaker_b`。
    * 每个*会话*中的一个*轮次*包含说话者的名字 `speaker`、对话id `dia_id` 以及对话内容 `text`。
    * 如果轮次包含图像，还会包括图像链接 `img_url`、由 [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) 模型为图像生成的标题 `blip_caption`，以及第三方模块 [icrawler](https://icrawler.readthedocs.io/en/latest/) 用于检索图像的搜索查询。
* `observation`（生成的）：`conversation` 中每个会话的观察记录（`session_<num>_observation`）。有关重新生成观察的代码，请参见下文。这些观察被用作我们论文中评估检索增强生成（RAG）模型的数据库之一。
* `session_summary`（生成的）：`conversation` 中每个会话的会话级摘要（`session_<num>_summary`）。有关重新生成会话级摘要的代码，请参见下文。这些摘要也被用作我们论文中评估RAG模型的数据库之一。
* `event_summary`（标注的）：`conversation` 中每个会话内每位说话者的重要事件列表（`events_session_<num>`）。这些是LoCoMo数据集中事件摘要任务的真实标注。
* `qa`（标注的）：LoCoMo数据集中问答任务的问答标注。每个样本包含 `question`（问题）、`answer`（答案）、`category`（类别）标签以及包含答案的对话id列表 `evidence`（如果可用）。

**注意 1**：此次发布是我们在2024年3月第一个Arxiv版本中发布的对话的子集。最初的发布包含50个对话。我们对数据进行了采样，保留了具有高质量标注的最长对话，以实现对闭源LLM的经济高效评估。

**注意 2**：我们不发布图像本身。但是，数据集中包含图像的网址、标题和搜索查询。

## 代码

API密钥、输出目录等配置变量在 `scripts/env.sh` 中设置，并在所有其他脚本开始时运行。

### 生成长期对话

使用我们基于LLM的生成框架，在预先分配个性的两个LLM代理之间生成*超长期*对话。

生成对话的代码位于 `scripts/generate_conversations.sh`，可按如下方式运行：

```bash
bash scripts/generate_conversations.sh
```

此代码可在两种设置下运行：

1. **生成具有自定义角色的代理之间的对话**。要启用此设置，请将 `--out-dir` 指向包含 `agent_a.json` 和 `agent_b.json` 文件的目录。这些文件应包含代理所代表的说话者的 `name` 和 `persona_summary`。参见 `data/multimodal_dialog/example` 中的示例。

```json
{
  "name": "Angela",
  "persona_summary": "Angela is a 31 year old woman who works as the manager of a gift shop in Chapel Hill. She curates interesting pieces from local artists and has maintained a beautiful gallery in the form of the gift shop. She also makes her own art sometimes, in the form of oil paintings."
}
```

2. **使用MSC数据集的提示创建个性**。要启用此设置，请将 `--out-dir` 指向一个空目录。这将使脚本从 `data/msc_personas_all.json` 中采样一对个性。

有关可为生成对话调整的各种参数的详细信息，请参见 `scripts/generate_conversations.py`。例如，可以更改 `--num-days` 以指定对话的时间跨度。

### 评估LLM

以（截断的）对话作为上下文，在LoCoMo问答任务上评估开源和闭源LLM。

* **评估OpenAI模型**
```bash
bash scripts/evaluate_gpts.sh
```

* **评估Anthropic模型**
```bash
bash scripts/evaluate_claude.sh
```

* **评估Gemini模型**
```bash
bash scripts/evaluate_gemini.sh
```

* **评估Huggingface上可用的模型**
```bash
bash scripts/evaluate_hf_llm.sh
```

### 生成观察与会话摘要

使用 `gpt-3.5-turbo` 从LoCoMo对话生成观察和会话摘要，用于评估基于RAG的模型。

我们在LoCoMo数据集的发布中提供了观察和摘要。按照以下说明重新生成相同内容或为不同的对话集生成。

* **从所有会话生成观察**：
```bash
bash scripts/generate_observations.sh
```

* **生成每个会话的摘要**：
```bash
bash scripts/generate_session_summaries.sh
```

**注意 3**：会话摘要不同于事件摘要任务的事件摘要。前者仅摘要单个会话，而事件摘要特定于每个说话者，并包含跨会话的因果、时间连接。

### 评估RAG模型

使用（a）对话、（b）观察和（c）会话摘要作为数据库，在LoCoMo问答任务上评估检索增强的 `gpt-3.5-turbo`。

* **使用基于检索的增强评估 `gpt-3.5-turbo`**
```bash
bash scripts/evaluate_rag_gpts.sh
```

### 评估事件摘要任务的模型

即将推出！

### 在多模态对话生成任务上训练和评估 `MiniGPT-5` 模型

即将推出！

## 引用

如果您在研究中使用LoCoMo，请引用我们的论文：

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```

## 许可证

请参阅 [LICENSE.txt](LICENSE.txt) 文件。

## 联系方式

如有任何问题或建议，请通过GitHub Issues联系我们。
