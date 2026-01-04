import os
from huggingface_hub import snapshot_download

# 设置国内镜像地址（通过环境变量）
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"

# 模型名称和保存路径（AutoDL数据盘路径）
MODEL_NAMES = [
    "louaaron/sedd-small",
    "louaaron/sedd-medium"
]
SAVE_DIR = "/root/autodl-tmp/sedd-models"  # 保持不变

for model_name in MODEL_NAMES:
    print(f"开始下载 {model_name}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=f"{SAVE_DIR}/{model_name.split('/')[-1]}",
        local_dir_use_symlinks=False  # 不使用符号链接，直接保存文件
        # 移除了错误的 endpoint_url 参数
    )
    print(f"{model_name} 下载完成，保存至 {SAVE_DIR}\n")