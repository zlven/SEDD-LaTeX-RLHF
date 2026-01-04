# split_train_valid.py
import os
from datasets import load_dataset
import random

# 加载预处理后的数据集
ds = load_dataset("parquet", data_files="/root/autodl-tmp/dataset/s1K-1.1-plain-math/data/train-00000-of-00001.parquet")["train"]

# 随机拆分80%训练，20%验证（固定种子保证可复现）
random.seed(42)
indices = list(range(len(ds)))
random.shuffle(indices)
train_idx = indices[:int(0.8*len(ds))]
valid_idx = indices[int(0.8*len(ds)):]

# 保存拆分后的数据集
train_ds = ds.select(train_idx)
valid_ds = ds.select(valid_idx)

train_ds.to_parquet("/root/autodl-tmp/dataset/s1K-1.1-train/data/train-00000-of-00001.parquet")
valid_ds.to_parquet("/root/autodl-tmp/dataset/s1K-1.1-valid/data/train-00000-of-00001.parquet")
print(f"拆分完成：训练集{len(train_ds)}条，验证集{len(valid_ds)}条")