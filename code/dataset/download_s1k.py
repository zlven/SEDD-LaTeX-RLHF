from datasets import load_dataset, DownloadConfig
import os

# 数据集保存路径
save_dir = "/root/autodl-tmp/dataset/s1K-1.1"
os.makedirs(save_dir, exist_ok=True)

# 创建下载配置对象（而非字典）
download_config = DownloadConfig()
download_config.base_url = "https://hf-mirror.com/datasets/"  # 设置镜像源

# 加载数据集并指定缓存目录和下载配置
dataset = load_dataset(
    "simplescaling/s1K-1.1",
    cache_dir=save_dir,
    download_config=download_config  # 传入配置对象
)

# 验证下载结果
print("数据集结构：", dataset)
print("第一条样本：", dataset["train"][0])