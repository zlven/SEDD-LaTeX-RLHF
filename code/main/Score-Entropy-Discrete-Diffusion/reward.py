# reward.py - 优化版 LaTeX 数学奖励函数（PPO RL 专用）

import re
import torch

class LaTeXReward:
    def __init__(self, syntax_weight=0.8, math_weight=0.7, length_penalty=0.05):
        self.syntax_weight = syntax_weight
        self.math_weight = math_weight
        self.length_penalty = length_penalty

        # 扩展 LaTeX 命令列表（增加容错）
        self.latex_commands = [
            'sin', 'cos', 'tan', 'log', 'ln', 'sqrt', 'frac', 'sum', 'int', 'lim',
            'infty', 'alpha', 'beta', 'gamma', 'delta', 'pi', 'theta', 'equiv',
            'approx', 'neq', 'leq', 'geq', 'subset', 'supset', 'text', 'mathbf',
            'mathbb', 'overline', 'underline', 'hat', 'tilde'
        ]

        self.math_symbols = ['+', '-', '*', '/', '=', '<', '>', '^', '_', '\\', '(', ')', '[', ']', '{', '}']

        # 基线
        self.baseline = 0.0
        self.baseline_decay = 0.99

    def get_reward(self, text: str) -> torch.Tensor:
        if not text.strip():
            return torch.tensor(-0.8)

        reward = 0.0

        # 1. 语法（宽松版）
        syntax_score = self.evaluate_syntax(text)
        reward += self.syntax_weight * syntax_score

        # 2. 数学内容（鼓励更多命令/符号）
        math_score = self.evaluate_math_content(text)
        reward += self.math_weight * math_score

        # 3. 长度（目标 50-150 字符）
        length_score = self.evaluate_length(text)
        reward -= self.length_penalty * length_score

        # 探索奖励（核心优化）
        # 配对 $...$（行内数学）
        dollar_pairs = text.count('$') // 2
        reward += dollar_pairs * 0.3

        # 行间 $$...$$
        reward += text.count('$$') * 0.4

        # 常见复杂结构
        complex_patterns = ['frac', 'sqrt', 'sum', 'int', 'lim', 'mathbf', 'mathbb']
        for pat in complex_patterns:
            if pat in text:
                reward += 0.25

        # 长度适中 bonus
        length = len(text)
        if 50 <= length <= 150:
            reward += 0.3

        # 保底（避免全 -1）
        reward = max(reward, -0.6)

        # 归一化
        reward = max(min(reward, 1.0), -1.0)

        return torch.tensor(reward, dtype=torch.float32)

    def update_baseline(self, reward: float):
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

    def evaluate_syntax(self, text: str) -> float:
        score = 0.0

        # 括号平衡（宽松：少扣分）
        pairs = [('{', '}'), ('[', ']'), ('(', ')')]
        for open_c, close_c in pairs:
            diff = abs(text.count(open_c) - text.count(close_c))
            score -= diff * 0.1  # 每对不平衡扣 0.1

        # $ 配对（宽松：奇数 $ 只扣少）
        dollar_count = text.count('$')
        if dollar_count % 2 == 1:
            score -= 0.2

        # 命令不完整（宽松）
        incomplete = len(re.findall(r'\\\w+{[^}]*$', text))
        score -= incomplete * 0.1

        return max(score, -0.8)

    def evaluate_math_content(self, text: str) -> float:
        score = 0.0

        # 命令数量
        commands = re.findall(r'\\(\w+)', text)
        valid = [cmd for cmd in commands if cmd in self.latex_commands]
        score += len(valid) * 0.15

        # 符号数量
        symbol_count = sum(text.count(s) for s in self.math_symbols)
        score += min(symbol_count * 0.05, 1.5)

        # 复杂结构
        for pat in ['frac', 'sqrt', 'sum', 'int', 'lim']:
            if pat in text:
                score += 0.3

        return min(score, 2.5)

    def evaluate_length(self, text: str) -> float:
        length = len(text)
        if length < 50:
            return 50 - length
        elif length > 150:
            return length - 150
        return 0.0