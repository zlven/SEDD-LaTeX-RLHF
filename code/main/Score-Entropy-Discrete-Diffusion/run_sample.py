import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
#######受模型本身参数量影响，模型理解效果一遍，故sft让模型学会数学格式，RL让模型学习latex语法#######
##########latex版本sft效果最好：      /root/autodl-tmp/exp_local/model-small-latex0.0
##########非latex公式效果最好medium： /root/autodl-tmp/exp_local/model-medium
##########rl后效果较好模型：          /root/autodl-tmp/exp_rl/model-rl-latex1
##########非latex公式效果最好small    /root/autodl-tmp/exp_local/model-small6.0
def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="/root/autodl-tmp/exp_local/model-small6.0", type=str)
    parser.add_argument("--dataset", default="s1K-1.1", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained(
        '/root/autodl-tmp/sedd-models/gpt2/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')

    # 获取EOS token ID（<|endoftext|>对应的ID）
    eos_token_id = tokenizer.eos_token_id

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )

    samples = sampling_fn(model)

    # ========== 核心修改：截断+清理<|endoftext|> ==========
    text_samples = []
    for seq in samples:
        # 1. 找到第一个<|endoftext|>的位置，截断序列
        eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # 只保留第一个EOS之前的内容
            seq = seq[:eos_positions[0]]
        # 2. 解码，跳过所有特殊token（包括EOS/BOS等）
        text = tokenizer.decode(
            seq,
            skip_special_tokens=True,  # 跳过特殊token
            clean_up_tokenization_spaces=True  # 清理多余空格
        )
        # 3.过滤残留的<|endoftext|>（防止解码不彻底）
        text = text.replace("<|endoftext|>", "").strip()
        text_samples.append(text)

    # 打印结果
    for i in text_samples:
        print(i)
        print("=================================================")


if __name__ == "__main__":
    main()