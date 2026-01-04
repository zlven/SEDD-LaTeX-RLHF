import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn


def get_ppo_loss_fn(model, ref_model, graph, noise, reward_model=None,beta=0.1, clip_ratio=0.2):
    """
    PPO裁剪型替代损失函数
    - model: 当前策略网络（SEDD）
    - ref_model: 用于KL正则的SFT参考模型
    - beta: KL惩罚系数
    - clip_ratio: PPO更新的裁剪阈值
    """

    def ppo_loss_fn(trajectories, rewards, old_log_probs):
        # trajectories: [B, L] token序列
        # rewards: [B] 奖励值
        # old_log_probs: [B] 采样阶段预计算的对数概率
        device = trajectories.device

        trajectories = trajectories.to(device)
        rewards = rewards.to(device)
        old_log_probs = old_log_probs.to(device)
        # 蒙特卡洛近似计算对数概率（修复形状不匹配）
        sampling_eps = 1e-5
        num_sigmas = 10  # 可硬编码或从配置读取；根据需求调整
        # 生成[B, num_sigmas]形状的随机sigma（范围：sampling_eps ~ 1.0）
        sigmas = torch.rand((trajectories.shape[0], num_sigmas), device=trajectories.device) * (
                    1.0 - sampling_eps) + sampling_eps

        log_probs = torch.zeros(trajectories.shape[0], device=trajectories.device)
        log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)

        # 遍历每个sigma样本，累加对数概率
        for i in range(num_sigmas):
            sigma = sigmas[:, i]  # [B] 形状
            entropy = graph.score_entropy(
                log_score_fn(trajectories, sigma[:, None]),  # sigma: [B, 1]（匹配模型输入）
                sigma[:, None], trajectories, trajectories
            ).mean(dim=-1)  # [B]
            log_probs -= entropy  # 负熵作为-log p的代理值，累加

        log_probs /= num_sigmas  # 对多个sigma取平均 [B]

        # 策略比值：新策略概率 / 旧策略概率
        ratios = torch.exp(log_probs - old_log_probs.detach())

        # 优势值计算（简易版：奖励 - 基线；进阶可改用GAE广义优势估计）
        advantages = rewards - reward_model.baseline  # 全局reward_model

        # PPO双损失裁剪
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 与参考模型的KL散度计算（同样用蒙特卡洛近似）
        ref_log_probs = torch.zeros(trajectories.shape[0], device=trajectories.device)
        ref_log_score_fn = mutils.get_score_fn(ref_model, train=False, sampling=False)

        for i in range(num_sigmas):
            sigma = sigmas[:, i]  # [B]
            ref_entropy = graph.score_entropy(
                ref_log_score_fn(trajectories, sigma[:, None]),
                sigma[:, None], trajectories, trajectories
            ).mean(dim=-1)  # [B]
            ref_log_probs -= ref_entropy

        ref_log_probs /= num_sigmas  # [B]
        kl = (log_probs - ref_log_probs).mean()

        # 最终损失 = PPO策略损失 + KL正则损失
        total_loss = policy_loss + beta * kl
        return total_loss

    return ppo_loss_fn