import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from catsample import sample_categorical  # 假设你有这个外部采样函数

def get_graph(config, device):
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")

def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)

class Graph(abc.ABC):
    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """ Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask). """
        pass

    @abc.abstractmethod
    def rate(self, i):
        """ Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n]. """
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """ Computes the i-th row of the rate matrix Q. Can be used to compute the reverse rate. """
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """ Computes the i-th column of the transition matrix e^{sigma Q}. """
        pass

    def sample_transition(self, i, sigma):
        """ Samples the transition vector. """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """ Constructs the reverse rate. Which is score * transp_rate """
        normalized_rate = self.transp_rate(i) * score
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        """最终根治版: scatter_add 稀疏加 diag + multinomial """
        sampled = rate.clone()  # [B, L, dim]  ← 密集 clone
        # 稀疏加 diag 1.0 用 scatter_add
        diag_ones = torch.ones_like(i.unsqueeze(-1), dtype=sampled.dtype)  # [B, L, 1]
        sampled.scatter_add_(-1, i.unsqueeze(-1), diag_ones)  # 加到 i 位置
        # 确保非负 + 归一化
        sampled = torch.clamp(sampled, min=0)
        sampled = sampled / sampled.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # 逐位置 multinomial 采样
        B, L, dim = sampled.shape
        sampled_indices = i.clone()  # [B, L]
        for l in range(L):
            probs_l = sampled[:, l, :]  # [B, dim]
            sampled_indices[:, l] = torch.multinomial(probs_l, num_samples=1).squeeze(-1)
        return sampled_indices

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """ Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with e^{-{dsigma} E} score """
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """ Sample the limiting distribution. Returns the probability vector as well. """
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """ Computes the score entropy function (with requisite constant normalization) """
        pass

class Uniform(Graph):
    """ Everything goes to everything else. Normalized down by dimension to avoid blowup. """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
        ratio = 1 - self.dim / (esigm1 + self.dim)
        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(x == x0, ratio * neg_term,
                               torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term)
        # constant factor
        const = torch.where(x == x0, (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
                           ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim)
        # positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const

class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1

    @property
    def absorb(self):
        return True

    def rate(self, i):
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        # 废弃dense计算，在reverse_rate直接处理
        raise NotImplementedError("Use reverse_rate directly for memory efficiency")

    def transition(self, i, sigma):
        pass

    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(i == self.dim - 1, 1 - (-sigma).squeeze(-1).exp(), 0)[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        const = ratio * (ratio.log() - 1)
        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy

    def reverse_rate(self, i, score):
        """ 内存优化版：sparse操作，无one_hot """
        normalized_rate = score.clone()  # [B, L, dim]
        mask_id = self.dim - 1
        is_mask = (i == mask_id)  # [B, L]
        # mask列全0
        normalized_rate[..., mask_id] = 0
        # non-mask行全0 + diag = -diag_score
        non_mask = ~is_mask
        normalized_rate[non_mask] = 0
        if non_mask.any():
            diag_score = torch.gather(score, -1, i.unsqueeze(-1)).squeeze(-1)  # [B, L]
            b_idx, l_idx = non_mask.nonzero(as_tuple=True)
            normalized_rate[b_idx, l_idx, i[b_idx, l_idx]] = -diag_score[b_idx, l_idx]
        return normalized_rate