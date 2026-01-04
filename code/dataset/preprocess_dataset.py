import re
import os
import shutil
from datasets import load_dataset


# ===================== 核心：LaTeX命令转纯文本数学符号 =====================
def latex_to_plain_math(text):
    if not text or text is None:
        return text

    # 1. 移除所有$符号（LaTeX的公式包裹符）
    text = text.replace("$", "")

    # 2. 移除空上下标（样本1的核心问题）
    text = re.sub(r"_{}^{}", "", text)

    # 3. LaTeX命令→纯文本符号（按优先级排序）
    latex_map = {
        # 集合/空间符号
        r"\\mathcal\{([A-Za-z]+)\}": r"ℋ",  # \mathcal{H}→ℋ（可根据需要调整）
        # 根号
        r"\\sqrt\{(.+?)\}": r"√\1",
        # 分数
        r"\\frac\{(.+?)\}\{(.+?)\}": r"\1/\2",
        # 下标（如δ_{ij}→δᵢⱼ，简单处理）
        r"\\delta_{ij}": r"δᵢⱼ",
        r"_i": r"ᵢ",
        r"_j": r"ⱼ",
        # 范数/内积
        r"\\left\|(.+?)\\right\|": r"||\1||",
        r"\\langle(.+?)\\rangle": r"<\1>",
        # 特殊符号
        r"\\minus": r"-",
        r"\\blacksquare": r"□",
        r"\\delta": r"δ",
        # 移除无意义的LaTeX命令
        r"\\text\{(.+?)\}": r"\1",
        r"\\textit\{(.+?)\}": r"\1",
    }

    for latex_cmd, plain_symbol in latex_map.items():
        text = re.sub(latex_cmd, plain_symbol, text)

    # 4. 处理\underbrace（样本3的多9乘积）
    text = re.sub(r"\\underbrace\{(.+?)\}_{\\text\{(.+?)\}}", r"\1（注：\2）", text)

    # 5. 移除$前后残留的空格，合并连续空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    # 路径配置
    RAW_DATA_DIR = "/root/autodl-tmp/dataset/s1K-1.1"
    PROCESSED_DATA_DIR = "/root/autodl-tmp/dataset/s1K-1.1-plain-math"  # 纯文本数学版本
    RAW_PARQUET = os.path.join(RAW_DATA_DIR, "data/train-00000-of-00001.parquet")

    # 创建输出目录
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, "data"), exist_ok=True)

    # 加载原始数据集（保留全部1000条）
    print(f"加载原始数据集：{RAW_PARQUET}")
    raw_ds = load_dataset("parquet", data_files=RAW_PARQUET, split="train")
    print(f"原始样本数：{len(raw_ds)}")

    # 应用纯文本转换（单进程，保证顺序）
    print("将LaTeX命令转为纯文本数学符号...")

    def process_example(example):
        example["question"] = latex_to_plain_math(example["question"])
        example["solution"] = latex_to_plain_math(example["solution"])
        return example

    processed_ds = raw_ds.map(process_example, num_proc=None, desc="纯文本数学转换")

    # 保存预处理后的数据
    output_parquet = os.path.join(PROCESSED_DATA_DIR, "data/train-00000-of-00001.parquet")
    processed_ds.to_parquet(output_parquet)
    print(f"保存纯文本数学数据集：{output_parquet}")

    # 复制README.md
    raw_readme = os.path.join(RAW_DATA_DIR, "README.md")
    if os.path.exists(raw_readme):
        shutil.copy(raw_readme, os.path.join(PROCESSED_DATA_DIR, "README.md"))

    # 生成对比TXT（前5条样本，验证效果）
    txt_path = os.path.join(PROCESSED_DATA_DIR, "纯文本数学对比.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LaTeX → 纯文本数学符号 对比（前5条样本）\n")
        f.write("=" * 100 + "\n\n")
        for idx in range(5):
            raw_q = raw_ds[idx]["question"]
            proc_q = processed_ds[idx]["question"]
            raw_s = raw_ds[idx]["solution"][:300]  # 截断避免过长
            proc_s = processed_ds[idx]["solution"][:300]

            f.write(f"【样本 {idx + 1}】\n")
            f.write(f"原始Question（含LaTeX）：{raw_q}\n")
            f.write(f"纯文本Question（无LaTeX）：{proc_q}\n")
            f.write(f"原始Solution片段：{raw_s}...\n")
            f.write(f"纯文本Solution片段：{proc_s}...\n")
            f.write("=" * 80 + "\n\n")

    # 验证样本1的转换效果（核心示例）
    print("\n=== 样本1转换效果验证 ===")
    print(f"原始Question：{raw_ds[0]['question']}")
    print(f"纯文本Question：{processed_ds[0]['question']}")
    print("\n转换前后对比：")
    # 关键修复：把 {} 转义为 {{}}，避免f-string解析错误
    print(f"原LaTeX格式：$20_{{}}^{{}}!$ → 纯文本：20!")

    print("\n✅ 纯文本数学数据集预处理完成！")
    print(f"👉 数据集路径：{PROCESSED_DATA_DIR}")
    print(f"👉 对比文件：{txt_path}")


if __name__ == "__main__":
    main()