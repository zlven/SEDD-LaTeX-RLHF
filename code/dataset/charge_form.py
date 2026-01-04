import torch
import os
import shutil
from model.ema import ExponentialMovingAverage
from model import SEDD
from omegaconf import OmegaConf

# 配置路径（确保与实际模型匹配）
input_bin_path = "/root/autodl-tmp/sedd-models/sedd-medium/pytorch_model.bin"
output_dir = "/root/autodl-tmp/sedd-models/converted_sedd-medium"

# 创建目录
os.makedirs(os.path.join(output_dir, ".hydra"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints-meta"), exist_ok=True)

# 复制正确的config.yaml（必须是small模型的配置）
# # 确保本地的config.yaml是small模型的配置
# local_yaml_path = "config.yaml"  # 这里的config.yaml必须包含上述small模型的参数
# shutil.copy(local_yaml_path, os.path.join(output_dir, ".hydra/config.yaml"))

# 加载配置文件（关键：用该配置创建模型）
cfg = OmegaConf.load(os.path.join(output_dir, ".hydra/config.yaml"))

# 加载原bin文件权重
model_state_dict = torch.load(input_bin_path, map_location="cpu")
# 处理DDP前缀（如果有）
model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

# 创建与配置匹配的临时模型（确保结构一致）
dummy_model = SEDD(cfg).to("cpu")  # 使用small配置创建模型

# 检查权重与模型是否匹配（关键步骤）
try:
    dummy_model.load_state_dict(model_state_dict, strict=True)
    print("权重与模型结构匹配！")
except RuntimeError as e:
    print(f"权重与模型结构不匹配：{e}")
    # 若报错，说明config.yaml与bin文件不匹配，需重新检查配置

# 初始化EMA（基于正确的模型结构）
ema = ExponentialMovingAverage(dummy_model.parameters(), decay=cfg.training.ema)
# 用原模型权重初始化EMA的shadow_params
ema.shadow_params = [p.clone() for p in dummy_model.parameters()]

# 构造完整的EMA状态
ema_state_dict = {
    "decay": ema.decay,
    "num_updates": 0,
    "shadow_params": ema.shadow_params
}

# 保存checkpoint
checkpoint = {
    "model": model_state_dict,
    "ema": ema_state_dict,
    "optimizer": {},
    "step": 0
}
torch.save(checkpoint, os.path.join(output_dir, "checkpoints-meta/checkpoint.pth"))

print(f"转换完成，输出目录：{output_dir}")