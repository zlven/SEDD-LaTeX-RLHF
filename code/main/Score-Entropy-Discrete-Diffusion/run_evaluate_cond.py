import torch
import argparse
import os
import warnings
from collections import defaultdict

# 关闭tokenizers并行警告（可选）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import sampling


def calculate_perplexity(samples, tokenizer, device):
    """
    计算条件生成样本的困惑度（PPL）：值越低，生成文本流畅度越高
    """
    # 加载GPT2-Large评估模型（SEDD原仓库标准）
    eval_model = GPT2LMHeadModel.from_pretrained(
        "/root/autodl-tmp/sedd-models/gpt2-large/models--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device).eval()

    with torch.no_grad():
        # 计算交叉熵损失并转换为PPL
        loss = eval_model(samples, labels=samples).loss
        perplexity = torch.exp(loss).mean().item()

    # 释放显存
    del eval_model
    torch.cuda.empty_cache()

    return perplexity


def calculate_ngram_repetition(texts, n=2):
    """
    纯Python实现2-gram重复率（无需nltk）：值越低，生成文本多样性越好
    """
    total_ngrams = 0
    repeated_ngrams = 0

    for text in texts:
        # 分词（按空格拆分，兼容GPT2分词结果）
        tokens = text.split()
        if len(tokens) < n:
            continue

        # 手动生成n-gram
        ng = []
        for i in range(len(tokens) - n + 1):
            ng.append(tuple(tokens[i:i + n]))

        # 统计重复n-gram
        ng_count = defaultdict(int)
        for g in ng:
            ng_count[g] += 1
            total_ngrams += 1
            if ng_count[g] > 1:
                repeated_ngrams += 1

    if total_ngrams == 0:
        return 0.0
    return repeated_ngrams / total_ngrams


def check_constraint_accuracy(text_samples, prefix, suffix):
    """
    验证条件生成的约束有效性（核心指标）：前缀/后缀是否准确保留
    返回：(前缀匹配率, 后缀匹配率)
    """
    prefix_match = 0
    suffix_match = 0
    total_samples = len(text_samples)

    for text in text_samples:
        # 检查前缀是否匹配（忽略首尾空格）
        if text.strip().startswith(prefix.strip()):
            prefix_match += 1
        # 检查后缀是否匹配（忽略首尾空格）
        if text.strip().endswith(suffix.strip()):
            suffix_match += 1

    prefix_acc = prefix_match / total_samples if total_samples > 0 else 0.0
    suffix_acc = suffix_match / total_samples if total_samples > 0 else 0.0
    return prefix_acc, suffix_acc


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Evaluate conditional generation of SEDD model")
    parser.add_argument("--model_path", default="/root/autodl-tmp/sedd-models/converted_sedd-small", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--prefix", type=str, default="Hi, my name is")
    parser.add_argument("--suffix", type=str, default=" and that's why I'm late.")
    args = parser.parse_args()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"===== 评估配置 =====")
    print(f"设备：{device}")
    print(f"模型路径：{args.model_path}")
    print(f"前缀：{args.prefix}")
    print(f"后缀：{args.suffix}")
    print(f"采样步数：{args.steps} | 批次大小：{args.batch_size}\n")

    # 加载分词器（使用用户指定的本地路径）
    print("加载分词器...")
    tokenizer = GPT2TokenizerFast.from_pretrained(
        '/root/autodl-tmp/sedd-models/gpt2/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
    )

    # 构建条件约束（prefix + suffix）
    print("构建条件约束...")
    prefix_ids = tokenizer(args.prefix).input_ids
    suffix_ids = tokenizer(args.suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(1024 - len(suffix_ids), 1024))
    input_ids = torch.tensor(input_ids, device=device)[None].repeat(args.batch_size, 1)

    # 定义投影函数（强制保留前缀/后缀）
    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x

    # 加载模型、图结构、噪声调度
    print("加载模型...")
    model, graph, noise = load_model(args.model_path, device)
    model.eval()  # 推理模式

    # 生成条件样本
    print("开始条件采样...")
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device, proj_fun=proj_fun
    )
    with torch.no_grad():
        samples = proj_fun(sampling_fn(model))  # (batch_size, 1024)

    # 解码为可读文本
    print("解码生成文本...\n")
    text_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

    # 打印生成的文本样本
    print("===== 条件生成文本样本 =====")
    for idx, text in enumerate(text_samples):
        print(f"\n【样本 {idx + 1}】")
        print(text)
        print("=" * 80)

    # 计算评估指标
    print("\n===== 条件生成评估结果 =====")
    # 1. 约束有效性（条件生成核心）
    prefix_acc, suffix_acc = check_constraint_accuracy(text_samples, args.prefix, args.suffix)
    print(f"1. 约束有效性：")
    print(f"   - 前缀匹配率：{prefix_acc:.4f}（1.0表示全部匹配）")
    print(f"   - 后缀匹配率：{suffix_acc:.4f}（1.0表示全部匹配）")

    # 2. 困惑度（PPL）
    ppl = calculate_perplexity(samples, tokenizer, device)
    print(f"2. 困惑度（PPL）：{ppl:.4f}（越低越好，预训练模型通常15-25）")

    # 3. 2-gram重复率（多样性）
    rep_rate = calculate_ngram_repetition(text_samples, n=2)
    print(f"3. 2-gram重复率：{rep_rate:.4f}（越低越好，<0.2为优质）")

    # 保存评估结果到文件
    eval_result = f"""
===== 条件生成评估报告 =====
模型路径：{args.model_path}
采样参数：步数={args.steps} | 批次大小={args.batch_size}
条件约束：前缀="{args.prefix}" | 后缀="{args.suffix}"
评估指标：
1. 前缀匹配率：{prefix_acc:.4f}
2. 后缀匹配率：{suffix_acc:.4f}
3. 困惑度（PPL）：{ppl:.4f}
4. 2-gram重复率：{rep_rate:.4f}
生成样本数：{len(text_samples)}
"""
    with open("cond_generation_evaluation_small.txt", "w", encoding="utf-8") as f:
        f.write(eval_result)
    print("\n评估报告已保存到 cond_generation_evaluation_small.txt")


if __name__ == "__main__":
    main()