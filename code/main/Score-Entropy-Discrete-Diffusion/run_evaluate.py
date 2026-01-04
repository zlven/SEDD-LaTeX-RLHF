import torch
import argparse
import os
from collections import defaultdict


from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling



def calculate_perplexity(samples, tokenizer, device):
    """
    计算生成样本的困惑度（PPL）：值越低，生成文本的语言流畅度越高
    """
    # 加载GPT2-Large作为评估模型（SEDD原仓库用此模型评估PPL）
    eval_model = GPT2LMHeadModel.from_pretrained(
        "/root/autodl-tmp/sedd-models/gpt2-large/models--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device).eval()

    total_perplexity = 0.0
    batch_size = samples.shape[0]

    with torch.no_grad():
        # 计算每个样本的PPL（忽略padding，仅计算有效token）
        loss = eval_model(samples, labels=samples).loss  # (batch_size,)
        perplexity = torch.exp(loss).mean().item()  # 批次平均PPL

    # 释放显存
    del eval_model
    torch.cuda.empty_cache()

    return perplexity


def calculate_ngram_repetition(texts, n=2):
    """纯Python实现n-gram重复率，无需NLTK"""
    total_ngrams = 0
    repeated_ngrams = 0

    for text in texts:
        tokens = text.split()  # 按空格分词
        if len(tokens) < n:
            continue

        # 手动生成n-gram（替代nltk.ngrams）
        ng = []
        for i in range(len(tokens) - n + 1):
            ng.append(tuple(tokens[i:i + n]))  # tuple保证可哈希

        # 统计重复
        ng_count = defaultdict(int)
        for g in ng:
            ng_count[g] += 1
            total_ngrams += 1
            if ng_count[g] > 1:
                repeated_ngrams += 1

    if total_ngrams == 0:
        return 0.0
    return repeated_ngrams / total_ngrams



def main():
    parser = argparse.ArgumentParser(description="Generate samples and evaluate them")
    parser.add_argument("--model_path", default="/root/autodl-tmp/sedd-models/converted_sedd-small", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 加载模型、图结构、噪声调度
    print("加载模型...")
    model, graph, noise = load_model(args.model_path, device)
    model.eval()  # 推理模式

    # 加载分词器
    tokenizer = GPT2TokenizerFast.from_pretrained(
        '/root/autodl-tmp/sedd-models/gpt2/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
    )

    # 生成样本
    print("开始采样...")
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )
    with torch.no_grad():
        samples = sampling_fn(model)  # (batch_size, 1024)

    # 解码为文本
    print("解码生成文本...")
    text_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

    # 打印生成的文本
    print("\n===== 生成的文本样本 =====")
    for idx, text in enumerate(text_samples):
        print(f"\n【样本 {idx + 1}】")
        print(text)
        print("=" * 80)

    # 评估生成结果
    print("\n===== 评估结果 =====")
    # 1. 困惑度（PPL）
    ppl = calculate_perplexity(samples, tokenizer, device)
    print(f"1. 困惑度（PPL）：{ppl:.4f}（越低越好，预训练模型通常在15-25之间）")

    # 2. 2-gram重复率
    rep_rate = calculate_ngram_repetition(text_samples, n=2)
    print(f"2. 2-gram重复率：{rep_rate:.4f}（越低越好，<0.2为优质）")

    # 汇总指标到文件（可选，便于后续对比）
    eval_result = f"""
    模型路径：{args.model_path}
    采样步数：{args.steps}
    批次大小：{args.batch_size}
    困惑度（PPL）：{ppl:.4f}
    2-gram重复率：{rep_rate:.4f}
 
    """
    with open("pretrain_evaluation_small.txt", "w", encoding="utf-8") as f:
        f.write(eval_result)
    print("\n评估结果已保存到 pretrain_evaluation_small.txt")


if __name__ == "__main__":
    main()