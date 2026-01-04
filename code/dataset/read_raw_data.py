import os
from datasets import load_dataset

def main():
    # 原始数据集路径（和你的目录一致）
    RAW_PARQUET = "/root/autodl-tmp/dataset/s1K-1.1-plain-math/data/train-00000-of-00001.parquet"
    # 保存原始样本的TXT文件路径
    OUTPUT_TXT = "/root/autodl-tmp/pro_samples_前10条.txt"

    # 1. 加载原始数据集（不做任何过滤/处理）
    print(f"加载原始数据集：{RAW_PARQUET}")
    raw_ds = load_dataset(
        "parquet",
        data_files=RAW_PARQUET,
        split="train"
    )
    print(f"原始数据集总样本数：{len(raw_ds)}")

    # 2. 读取前10条完整样本
    print("\n=== 打印前10条原始样本（完整内容）===")
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("s1K-1.1 原始数据集 - 前10条样本\n")
        f.write("="*100 + "\n\n")

        for idx in range(10):
            sample = raw_ds[idx]
            # 提取核心字段（question + solution）
            question = sample["question"] if sample["question"] else "无"
            solution = sample["solution"] if sample["solution"] else "无"

            # 打印到控制台（方便快速查看）
            print(f"\n【原始样本 {idx+1}】")
            print(f"Question：{question}")
            print(f"Solution：{solution[:500]}..." if len(solution) > 500 else f"Solution：{solution}")
            print("-"*80)

            # 保存到TXT文件（完整内容，不截断）
            f.write(f"【原始样本 {idx+1}】\n")
            f.write(f"Question：{question}\n")
            f.write(f"Solution：{solution}\n")
            f.write("="*80 + "\n\n")

    print(f"\n✅ 原始样本已保存到：{OUTPUT_TXT}")


if __name__ == "__main__":
    main()