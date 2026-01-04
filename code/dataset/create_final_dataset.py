import json
import os

# ========== 配置参数 ==========
# 第四轮清洗（最终清洗）结果的输入路径
FINAL_CLEANED_JSON_PATH = "/root/autodl-tmp/dataset_cleaned/step3_latex_normalized.json"

# 最终SFT数据集的输出目录和文件名
SFT_OUTPUT_DIR = "/root/autodl-tmp/sft_dataset"
FINAL_SFT_JSONL_PATH = os.path.join(SFT_OUTPUT_DIR, "s1k_cleaned_final.jsonl")

os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)


# ========== 主执行流程 ==========
def create_final_sft_file():
    """
    读取最终清洗好的JSON文件，并生成一个只包含 question 和 solution 的 .jsonl 文件。
    """
    print(f"⏳ 正在加载最终清洗数据: {FINAL_CLEANED_JSON_PATH}")
    try:
        with open(FINAL_CLEANED_JSON_PATH, "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件！请确认已完成所有清洗步骤: {FINAL_CLEANED_JSON_PATH}")
        return

    print(f"🚀 正在生成最终 SFT 格式文件 (.jsonl)...")
    count = 0
    with open(FINAL_SFT_JSONL_PATH, "w", encoding="utf-8") as f:
        for item in cleaned_data:
            # 只保留SFT训练最核心的两个字段
            sft_entry = {
                "question": item["cleaned_question"],
                "solution": item["cleaned_solution"],
            }
            # 将每个字典作为一行写入 .jsonl 文件
            f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"🎉 最终 SFT 数据集生成完毕！")
    print(f"📄 文件路径: {FINAL_SFT_JSONL_PATH}")
    print(f"📝 总计写入 {count} 条记录。")


if __name__ == "__main__":
    create_final_sft_file()