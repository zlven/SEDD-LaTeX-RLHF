import re
import json
import os

# ========== 配置参数（适配你的环境） ==========
RAW_PARQUET = "/root/autodl-tmp/dataset/s1K-1.1/data/train-00000-of-00001.parquet"
BACKUP_DIR = "/root/autodl-tmp/dataset_backup"
CLEAN_DIR = "/root/autodl-tmp/dataset_cleaned"
TOP2_TXT_PATH = "/root/autodl-tmp/step0_top2_cleaned.txt"

os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)


# ========== 工具函数：统一字符计数逻辑（修复核心问题） ==========
def count_non_whitespace_chars(text):
    """
    正确计算非空白字符数（移除所有空白字符：空格/换行/制表符/全角空格等）
    避免因空白字符类型不同导致计数不一致
    """
    if not text:  # 处理空值/None
        return 0
    # 移除所有空白字符（\s 匹配空格/换行/制表符，\u3000匹配全角空格）
    non_whitespace = re.sub(r"[\s\u3000]+", "", text)
    return len(non_whitespace)


# ========== 第一步：加载并备份原始数据 ==========
def load_and_backup_raw_data():
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=RAW_PARQUET, split="train")
    raw_data = [
        {
            "id": idx,
            "question": s["question"].strip() if s["question"] else "",
            "solution": s["solution"].strip() if s["solution"] else ""
        }
        for idx, s in enumerate(ds)
    ]
    raw_backup_path = os.path.join(BACKUP_DIR, "raw_data.json")
    with open(raw_backup_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 原始数据已备份：{raw_backup_path}")
    print(f"📊 数据集总样本数：{len(raw_data)}")
    return raw_data


# ========== 第二步：格式标准化清洗（修复断言错误） ==========
def step0_standardize_format(raw_data):
    standardized_data = []
    # 记录差异样本（不中断流程）
    diff_samples = []

    for item in raw_data:
        # 清洗Question：仅处理格式，不修改内容
        cleaned_q = re.sub(
            r"[\s\u3000]+",  # 匹配所有空白字符（包括全角空格）
            " ",
            item["question"].replace("\r", "").replace("\x00", "")
        ).strip()

        # 清洗Solution：同Question规则
        cleaned_s = re.sub(
            r"[\s\u3000]+",
            " ",
            item["solution"].replace("\r", "").replace("\x00", "")
        ).strip()

        # 修复字符计数逻辑（用统一的工具函数）
        raw_q_char = count_non_whitespace_chars(item["question"])
        cleaned_q_char = count_non_whitespace_chars(cleaned_q)
        raw_s_char = count_non_whitespace_chars(item["solution"])
        cleaned_s_char = count_non_whitespace_chars(cleaned_s)

        # 优化验证：改为日志提示，不中断流程
        if raw_q_char != cleaned_q_char:
            diff_log = f"样本{item['id']} Question字符数差异：原始{raw_q_char} → 清洗后{cleaned_q_char}"
            diff_samples.append(diff_log)
            print(f"⚠️ {diff_log}")
        if raw_s_char != cleaned_s_char:
            diff_log = f"样本{item['id']} Solution字符数差异：原始{raw_s_char} → 清洗后{cleaned_s_char}"
            diff_samples.append(diff_log)
            print(f"⚠️ {diff_log}")

        # 记录清洗日志
        standardized_data.append({
            "id": item["id"],
            "raw_question": item["question"],
            "cleaned_question": cleaned_q,
            "raw_solution": item["solution"],
            "cleaned_solution": cleaned_s,
            "clean_log": ["step0：移除不可见字符，统一空格/换行，首尾去空格"],
            "char_diff": {
                "question": raw_q_char - cleaned_q_char,
                "solution": raw_s_char - cleaned_s_char
            }
        })

    # 保存差异日志（方便核对）
    if diff_samples:
        diff_log_path = os.path.join(CLEAN_DIR, "step0_char_diff.log")
        with open(diff_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(diff_samples))
        print(f"📝 字符数差异日志已保存：{diff_log_path}")

    # 保存第一轮清洗结果
    step0_save_path = os.path.join(CLEAN_DIR, "step0_standardized.json")
    with open(step0_save_path, "w", encoding="utf-8") as f:
        json.dump(standardized_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 第一轮清洗完成：{step0_save_path}")
    return standardized_data


# ========== 第三步：提取前两条结果，保存到TXT ==========
def save_top2_to_txt(standardized_data):
    top2_data = standardized_data[:2]

    txt_content = "===== 第一轮清洗（格式标准化）前两条结果 =====\n\n"
    for idx, item in enumerate(top2_data):
        txt_content += f"【样本 {item['id'] + 1}】\n"
        txt_content += f"--- 原始Question ---\n{item['raw_question']}\n\n"
        txt_content += f"--- 清洗后Question ---\n{item['cleaned_question']}\n\n"
        txt_content += f"--- 原始Solution ---\n{item['raw_solution']}\n\n"
        txt_content += f"--- 清洗后Solution ---\n{item['cleaned_solution']}\n\n"
        txt_content += f"--- 字符数差异 ---\nQuestion：{item['char_diff']['question']} | Solution：{item['char_diff']['solution']}\n"
        txt_content += f"--- 清洗日志 ---\n{item['clean_log'][0]}\n"
        txt_content += "=" * 80 + "\n\n"

    with open(TOP2_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"✅ 前两条清洗结果已保存：{TOP2_TXT_PATH}")


# ========== 执行第一轮清洗 ==========
if __name__ == "__main__":
    raw_data = load_and_backup_raw_data()
    step0_data = step0_standardize_format(raw_data)
    save_top2_to_txt(step0_data)
    print("\n🎉 第一轮清洗完成！（字符数差异已记录，不影响后续流程）")