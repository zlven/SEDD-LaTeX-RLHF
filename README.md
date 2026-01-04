
# SEDD-LaTeX-RLHF

**SEDD + PPO 实现离散扩散模型的 RLHF 对齐，用于生成高质量 LaTeX 数学公式**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange)](https://pytorch.org/)

## 项目简介

本项目基于论文 [**Score Entropy Discrete Diffusion Models**](https://arxiv.org/abs/2310.16834)（SEDD）的官方实现，完成了以下工作：

1. 在 HuggingFace 预训练模型（sedd-small / sedd-medium）基础上成功复现采样与条件生成  
2. 使用 S1K-1.1 数学数据集进行监督微调（SFT）  
3. 将 PPO（Proximal Policy Optimization）与 SEDD 结合，实现离散扩散模型的 RLHF 对齐  
4. 设计针对 LaTeX 的多维度奖励函数（语法 + 数学内容 + 长度 + 探索 bonus）  
5. 针对大词汇表（GPT-2 vocab=50257）进行深度工程优化（稀疏 scatter_add_ + 逐位置采样），彻底解决 OOM 问题，使单卡训练成为可能  

最终效果：相比纯 SFT，RL 对齐后的模型生成的 LaTeX 公式在语法正确率、括号匹配、复杂结构使用等方面显著提升。

## 目录结构

```
SEDD-LaTeX-RLHF/
├── configs/                  # Hydra 配置文件
├── data/                     # 数据处理脚本（S1K-1.1 加载与清洗）
├── graph_lib.py              # 核心 graph 操作（含稀疏优化）
├── losses.py                 # SFT loss + PPO loss
├── noise_lib.py
├── reward.py                 # LaTeXReward 奖励函数
├── sampling.py               # 采样函数（含稀疏优化）
├── run_rl.py                 # PPO RLHF 主训练脚本
├── run_train.py              # SFT 训练脚本
├── trainRL.py                # RL 入口
├── train.py                  # SFT 入口
├── utils.py
├── requirements.txt
└── README.md
```

## 环境要求

- Python ≥ 3.10
- PyTorch 2.0+（推荐 CUDA 11.8+）
- transformers、omegaconf、hydra-core 等（见 requirements.txt）

```bash
conda create -n sedd-rl python=3.10
conda activate sedd-rl
pip install -r requirements.txt
```

## 数据准备

本项目使用 [S1K-1.1](https://github.com/simplescaling/s1) 数学数据集：

```bash
git clone https://github.com/simplescaling/s1.git /path/to/dataset/s1
# 数据会自动放在 cache_dir 下
```

## 快速开始

### 1. SFT 微调（可选，先跑通基线）

```bash
python train.py \
    data.train=s1K-1.1 \
    data.valid=s1K-1.1 \
    data.cache_dir=/path/to/dataset \
    training.batch_size=32 \
    training.n_iters=10000
```

### 2. RLHF 训练（PPO）

加载 SFT checkpoint 后运行：

```bash
python trainRL.py \
    rl.batch_size=4 \
    rl.kl_beta=0.1 \
    rl.ppo_clip=0.2 \
    rl.ppo_epochs=3 \
    rl.num_sigma_samples=3 \
    rl.reward_freq=10 \
    training.n_iters=10000
```

### 3. 采样生成

```bash
python run_sample.py --ckpt_path=/path/to/checkpoint.pth
```

## 核心贡献与创新点

1. **理论落地**：利用 SEDD 论文 Theorem 3.6（score entropy ≈ -log p），通过多 sigma 蒙特卡洛平均实现序列概率估计，使 PPO 能直接作用于离散扩散模型  
2. **工程优化**：针对大 vocab 彻底稀疏化（scatter_add_ + 循环），内存占用从 GiB 降至 MB 级  
3. **奖励设计**：专为 LaTeX 定制的多维度奖励函数，兼顾语法、数学内容与探索激励  
4. **完整 RLHF 流程**：SFT → ref_model → PPO + KL 正则 → 稳定对齐

## 致谢

- 原论文与官方仓库：https://github.com/locuslab/SEDD
- S1K 数据集：https://github.com/simplescaling/s1
- 本项目所有优化均为个人实现，用于学术研究与交流

## License

MIT License



欢迎 Star & Fork！如有问题欢迎提 Issue 😊
