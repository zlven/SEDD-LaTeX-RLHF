import datetime
import os
import gc
from copy import deepcopy
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from model import utils as mutils
from reward import LaTeXReward

torch.backends.cudnn.benchmark = True

def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30)
    )

def cleanup():
    dist.destroy_process_group()

def _run_rl(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    work_dir = cfg.work_dir
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")

    # 🔥 对齐原始：初始化目录（和原始一致）
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # 🔥 对齐原始：日志逻辑（和原始一致）
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(f"🚀 Starting RL Training with PPO (Rank {rank}/{world_size})")
    mprint(f"📁 Work directory: {work_dir}")
    mprint(f"Found {torch.cuda.device_count()} CUDA devices.")
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # 🔥 对齐原始：模型/DDP包装（和原始一致）
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)  # 原始代码包装了noise

    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)  # 原始代码包装了model

    # 加载预训练权重（和原始一致）
    pretrained_path = "/root/autodl-tmp/exp_local/model-small-latex0.0/checkpoints/checkpoint_5.pth"
    if rank == 0 and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
        score_model.load_state_dict(state_dict)
        mprint(f"✅ Loaded pretrained weights from {pretrained_path}")
    elif rank == 0:
        mprint("⚠️ Pretrained model not found, training from scratch")

    # 🔥 对齐原始：参数统计/EMA（和原始一致）
    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
    mprint(f"EMA: {ema}")

    # RL特有：参考模型
    ref_model = deepcopy(score_model).to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    if rank == 0:
        mprint("✅ Reference model loaded for KL regularization")

    # RL特有：奖励模型
    reward_model = LaTeXReward(
        syntax_weight=cfg.rl.reward.syntax_weight,
        math_weight=cfg.rl.reward.math_content_weight,
        length_penalty=cfg.rl.reward.length_penalty
    )
    if rank == 0:
        mprint("✅ Reward model loaded")

    # 🔥 对齐原始：优化器/状态初始化（和原始一致）
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0)

    # 🔥 对齐原始：加载checkpoint（和原始一致）
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])
    mprint(f"Starting RL training at step {initial_step + 1}")

    # 🔥 对齐原始：tokenizer加载（和原始一致）
    tokenizer = GPT2TokenizerFast.from_pretrained(
        '/root/autodl-tmp/sedd-models/gpt2/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
    )
    tokenizer.pad_token = tokenizer.unk_token  # 原始代码的关键配置
    EOS_TOKEN_ID = tokenizer.encode(tokenizer.eos_token)[0]

    # 🔥 对齐原始：数据加载（和原始一致）
    train_ds, eval_ds = data.get_dataloaders(cfg)
    train_iter = iter(train_ds)

    # 🔥 对齐原始：采样函数（和原始一致）
    sampling_eps = 1e-5
    sampling_shape = (cfg.rl.batch_size // world_size, cfg.model.length)
    sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    # RL参数（保留）
    ppo_clip = getattr(cfg.rl, 'ppo_clip', 0.2)
    kl_beta = getattr(cfg.rl, 'kl_beta', 0.1)
    ppo_epochs = getattr(cfg.rl, 'ppo_epochs', 3)
    num_sigma_samples = getattr(cfg.rl, 'num_sigma_samples', 3)
    reward_freq = getattr(cfg.rl, 'reward_freq', 10)
    num_train_steps = cfg.training.n_iters

    # RL损失函数（保留）
    ppo_loss_fn = losses.get_ppo_loss_fn(
        score_model, ref_model, graph, noise,
        reward_model=reward_model, beta=kl_beta, clip_ratio=ppo_clip
    )

    # 🔥 对齐原始：打印训练参数（补充原始风格）
    if rank == 0:
        mprint(f"🎯 Training Parameters:")
        mprint(f" Total steps: {num_train_steps}")
        mprint(f" Log freq: {cfg.training.log_freq}")
        mprint(f" Snapshot freq: {cfg.training.snapshot_freq}")
        mprint(f" Preemption save freq: {cfg.training.snapshot_freq_for_preemption}")
        mprint(f" PPO clip: {ppo_clip} | KL beta: {kl_beta} | PPO epochs: {ppo_epochs}")

    # RL采样函数（保留）
    def sampling_fn_with_logprob(model):
        with torch.enable_grad():
            samples = sampling_fn(model)
        B_local, L = samples.shape
        log_probs = torch.zeros(B_local, device=device)
        log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
        for _ in range(num_sigma_samples):
            sigmas = torch.rand(B_local, device=device) * (1.0 - sampling_eps) + sampling_eps
            entropy = graph.score_entropy(
                log_score_fn(samples, sigmas[:, None]),
                sigmas[:, None], samples, samples
            ).mean(dim=-1)
            log_probs -= entropy
        log_probs /= num_sigma_samples
        return samples, log_probs

    # 🔥 核心：训练循环（对齐原始保存逻辑）
    while state['step'] < num_train_steps:
        step = state['step'] + 1
        state['step'] = step

        # 🔥 对齐原始：数据加载逻辑（和原始一致）
        try:
            if cfg.data.train != "text8":
                batch = next(train_iter)['input_ids'].to(device)
            else:
                batch = next(train_iter).to(device)
        except StopIteration:
            train_iter = iter(train_ds)
            if cfg.data.train != "text8":
                batch = next(train_iter)['input_ids'].to(device)
            else:
                batch = next(train_iter).to(device)

        # RL更新逻辑（保留）
        if step % reward_freq == 0:
            gc.collect()
            torch.cuda.empty_cache()

            if rank == 0:
                mprint(f"🔄 PPO RL Step {step}: Generating rollouts...")

            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            trajectories, old_log_probs = sampling_fn_with_logprob(score_model)
            ema.restore(score_model.parameters())

            traj_list = [torch.zeros_like(trajectories) for _ in range(world_size)]
            prob_list = [torch.zeros_like(old_log_probs) for _ in range(world_size)]
            dist.all_gather(traj_list, trajectories)
            dist.all_gather(prob_list, old_log_probs)
            all_trajectories = torch.cat(traj_list, dim=0)
            all_old_log_probs = torch.cat(prob_list, dim=0)

            if rank == 0:
                sentences = []
                for seq in all_trajectories.cpu():
                    eos_pos = (seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        seq = seq[:eos_pos[0]]
                    sentence = tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                    sentences.append(sentence)

                rewards = []
                for text in sentences:
                    if text.strip():
                        r = reward_model.get_reward(text)
                        if not torch.is_tensor(r):
                            r = torch.tensor(r, device=device, dtype=torch.float32)
                        elif r.device != device:
                            r = r.to(device)
                        rewards.append(r)
                        reward_model.update_baseline(r.item())
                    else:
                        rewards.append(torch.tensor(0.0, device=device, dtype=torch.float32))
                rewards_tensor = torch.stack(rewards)

                for _ in range(ppo_epochs):
                    optimizer.zero_grad()
                    ppo_loss = ppo_loss_fn(all_trajectories.to(device), rewards_tensor, all_old_log_probs.to(device))
                    scaler.scale(ppo_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    mprint(f"📊 PPO RL Step {step}: PPO Loss: {ppo_loss.item():.5f} | Avg Reward: {rewards_tensor.mean().item():.3f}")
                    if sentences:
                        mprint(f" Sample: {sentences[0][:200]}...")

                gc.collect()
                torch.cuda.empty_cache()

        # 🔥 对齐原始：预抢占保存（和原始一致）
        if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
            utils.save_checkpoint(checkpoint_meta_dir, state)
            mprint(f"💾 Preemption checkpoint saved at step {step} to {checkpoint_meta_dir}")

        # 🔥 对齐原始：正式保存 + 采样生成（和原始一致）
        if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
            # 1. 正式保存checkpoint（按step拆分，和原始一致）
            save_step = step // cfg.training.snapshot_freq
            if rank == 0:
                ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth')
                utils.save_checkpoint(ckpt_path, state)
                mprint(f"💾 Formal checkpoint saved at step {step} to {ckpt_path}")

            # 2. 生成并保存样本（和原始一致）
            if cfg.training.snapshot_sampling:
                mprint(f"Generating text samples at step: {step}")
                this_sample_dir = os.path.join(sample_dir, f"iter_{step}")
                utils.makedirs(this_sample_dir)

                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample = sampling_fn(score_model)
                ema.restore(score_model.parameters())

                # 解码样本（和原始一致）
                sentences = []
                for seq in sample:
                    eos_positions = (seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        seq = seq[:eos_positions[0]]
                    sentence = tokenizer.decode(
                        seq, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    ).strip()
                    sentences.append(sentence)

                # 保存样本（和原始一致）
                file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                with open(file_name, 'w') as file:
                    for sentence in sentences:
                        file.write(sentence + "\n")
                        file.write("============================================================================================\n")

                dist.barrier()

    # 训练结束（保留）
    if rank == 0:
        mprint("✅ RL training completed!")

def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run_rl(rank, world_size, cfg)
    finally:
        cleanup()