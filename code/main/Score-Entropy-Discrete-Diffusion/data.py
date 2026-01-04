import re
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch
import os
import urllib.request
import zipfile
import requests
import json
from datasets import Dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "/root/autodl-tmp/sedd-models/gpt2/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    # pad_token用eos_token（GPT2原生设计），严格控制EOS仅在文本末尾
    tokenizer.pad_token = tokenizer.eos_token
    EOS_TOKEN_ID = tokenizer.eos_token_id

    name_normalized = name.strip().lower()
    if name_normalized == "s1k-1.1":
        mode = "valid" if mode == "validation" else mode
        # 1. 加载清洗后的JSONL文件
        cleaned_data_path = os.path.join(cache_dir, "s1k_cleaned_final.jsonl")
        if not os.path.exists(cleaned_data_path):
            raise FileNotFoundError(f"清洗后的数据集文件不存在：{cleaned_data_path}")

        full_dataset = load_dataset(
            "json",
            data_files={"train": cleaned_data_path},
            split="train"
        )

        # 2. 拆分训练/验证集
        dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)
        dataset = dataset_split["train"] if mode == "train" else dataset_split["test"]

        # 3. 仅做文本合并+末尾加EOS（无需LaTeX清洗）
        def format_text(example):
            # 直接合并question和solution，文本已清洗过
            text = f"Question: {example['question']}\nSolution: {example['solution']}"
            # 只在文本末尾添加1个EOS（避免重复添加）
            return {"text": text.strip() + tokenizer.eos_token}

        dataset = dataset.map(format_text)
        # 删除冗余字段
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

        # 4. 分词函数（核心：确保EOS只在末尾，无重复）
        def tokenize_function(example):
            tokenized = tokenizer(
                example["text"],
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_attention_mask=False,
                return_overflowing_tokens=False,
            )
            # 仅保留文本末尾的EOS，其他位置的EOS替换为pad
            input_ids = []
            for seq in tokenized["input_ids"]:
                non_pad_indices = [i for i, idx in enumerate(seq) if idx != tokenizer.pad_token_id]
                if non_pad_indices:
                    last_non_pad = non_pad_indices[-1]
                    new_seq = [
                        idx if (idx != EOS_TOKEN_ID or i == last_non_pad) else tokenizer.pad_token_id
                        for i, idx in enumerate(seq)
                    ]
                else:
                    new_seq = seq
                input_ids.append(new_seq)
            tokenized["input_ids"] = input_ids
            return tokenized

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names
        )
        dataset.set_format(type="torch", columns=["input_ids"])
        return dataset
    # name_normalized = name.strip().lower()
    # if name_normalized == "s1k-1.1":
    #     mode = "valid" if mode == "validation" else mode
    #     # 1. 拼接Parquet文件路径（确保路径正确）
    #     parquet_path = os.path.join(cache_dir, "s1K-1.1/data/train-00000-of-00001.parquet")
    #     # 检查文件是否存在，避免加载失败
    #     if not os.path.exists(parquet_path):
    #         raise FileNotFoundError(
    #             f"Parquet文件不存在：{parquet_path}\n请确认cache_dir配置正确，或替换为从HuggingFace加载：full_dataset = load_dataset('simplescaling/s1K-1.1', split='train')")
    #
    #     # 2. 加载原始Parquet文件（全部1000样本）
    #     full_dataset = load_dataset(
    #         "parquet",
    #         data_files={"train": parquet_path},
    #         split="train"  # 加载数据集作者的train拆分（全部样本）
    #     )
    #     # 3. 手动拆分训练/验证集（10%验证集，seed保证可复现）
    #     dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    #     # 4. 根据mode参数返回对应子集
    #     if mode == "train":
    #         dataset = dataset_split["train"]  # 900样本（训练用）
    #     elif mode == "valid":
    #         dataset = dataset_split["test"]  # 100样本（验证用）
    #     else:
    #         raise ValueError(f"mode {mode} not supported for s1K-1.1, only 'train'/'valid'")
    #
    #     # 5. 格式化文本：合并question+solution为text字段
    #     def format_text(example):
    #         return {"text": f"Question: {example['question']}\nSolution: {example['solution']}"}
    #
    #     dataset = dataset.map(format_text)
    #
    #     # 6. 删除冗余字段，仅保留text
    #     dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    #
    #     # ========== 修复2：Tokenizer 核心配置（彻底解决<|endoftext|>问题） ==========
    #     tokenizer = GPT2TokenizerFast.from_pretrained(
    #         "/root/autodl-tmp/sedd-models/gpt2/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
    #     # 关键：不用eos当pad，改用unk_token（避免模型学生成<|endoftext|>）
    #     tokenizer.pad_token = tokenizer.unk_token
    #     EOS_TOKEN_ID = tokenizer.encode(tokenizer.eos_token)[0]  # 单独存eos的ID
    #
    #     # 7. 分词函数：仅文本结尾加eos，padding用unk，过滤多余eos
    #     def tokenize_function(example):
    #         # 先给每个文本末尾加真正的eos（结束符）
    #         texts_with_eos = [t.strip() + tokenizer.eos_token for t in example["text"]]
    #         # 分词：截断到block_size，padding用unk_token
    #         tokenized = tokenizer(
    #             texts_with_eos,
    #             truncation=True,
    #             max_length=block_size,
    #             padding="max_length",
    #             return_overflowing_tokens=False,
    #         )
    #         # 过滤：只保留文本末尾的eos，其他位置的eos替换为unk
    #         input_ids = []
    #         for seq in tokenized["input_ids"]:
    #             # 找到最后一个非pad_token的位置
    #             non_pad_indices = [i for i, idx in enumerate(seq) if idx != tokenizer.pad_token_id]
    #             if non_pad_indices:
    #                 last_non_pad = non_pad_indices[-1]
    #                 # 非末尾的eos全部替换为unk
    #                 new_seq = [
    #                     idx if (idx != EOS_TOKEN_ID or i == last_non_pad) else tokenizer.pad_token_id
    #                     for i, idx in enumerate(seq)
    #                 ]
    #             else:
    #                 new_seq = seq
    #             input_ids.append(new_seq)
    #         tokenized["input_ids"] = input_ids
    #         return tokenized
    #
    #     # 执行分词：替换原有字段，生成input_ids
    #     dataset = dataset.map(
    #         tokenize_function,
    #         batched=True,
    #         num_proc=num_proc,
    #         remove_columns=dataset.column_names
    #     )
    #
    #     # 转换为PyTorch张量
    #     dataset.set_format(type="torch", columns=["input_ids"])
    #
    #     # 必须return，避免走到后面的错误逻辑
    #     return dataset

        #其他数据集的加载逻辑
    elif name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada":
        data = dataset
    else:
        data = dataset[mode]


    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained('/root/autodl-tmp/sedd-models/gpt2/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    ###数据预处理###
    def preprocess_and_tokenize(example):

        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]


        # print(list(example.keys()))
        # exit()
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader

