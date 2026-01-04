import re
import json
import os

# ========== 配置参数 (请确保与你的环境匹配) ==========
# 第二轮清洗结果的输入路径
STEP1_RESULT_PATH = "/root/autodl-tmp/dataset_cleaned/step1_format_fixed.json"
# 第三轮清洗结果的输出目录
CLEAN_DIR = "/root/autodl-tmp/dataset_cleaned"
# 第三轮清洗结果的完整保存路径
STEP2_SAVE_PATH = os.path.join(CLEAN_DIR, "step2_answer_extracted.json")
# 用于人工检查的前两条结果的TXT保存路径
TOP2_REVIEW_TXT_PATH = os.path.join(CLEAN_DIR, "step2_top2_review.txt")

os.makedirs(CLEAN_DIR, exist_ok=True)


# ========== 工具函数：加载JSON数据 ==========
def load_json_data(file_path):
    """从指定路径加载JSON文件"""
    print(f"⏳ 正在加载第二轮清洗后的数据: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 数据加载成功，共 {len(data)} 条样本。")
        return data
    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件！请检查路径是否正确: {file_path}")
        return None


# ========== 第三步：提取并标准化最终答案 ==========
def step2_extract_and_format_answer(data):
    """
    第三轮清洗：
    1. 从 solution 中用正则表达式查找 \boxed{...} 内的最终答案。
    2. 将提取的答案存入新字段 "final_answer"。
    3. 在 solution 末尾追加标准化的最终答案文本块（如果找到答案）。
    """
    print("\n🚀 开始第三轮清洗：提取并标准化最终答案...")
    processed_data = []
    found_count = 0

    for item in data:
        new_item = json.loads(json.dumps(item))  # 使用深拷贝以进行安全修改
        s = new_item["cleaned_solution"]

        # 你的数据集元数据中显示，答案被包裹在 \boxed{...} 中
        match = re.search(r'\\boxed\{(.*?)\}', s, re.DOTALL)

        final_answer = None
        if match:
            final_answer = match.group(1).strip()
            found_count += 1

            # 构造一个标准化的答案文本块，这种格式有助于SFT
            # 注意双反斜杠\\boxed，以在字符串中正确表示\boxed
            answer_section = f"\n\n#### Final Answer\nThe final answer is $\\boxed{{{final_answer}}}$"

            # 追加到solution末尾
            s += answer_section

            new_item["clean_log"].append(f"step2: 成功提取到 \\boxed 答案")
        else:
            # 对于你的样本1这种只有数字的简单答案，也可以做一个特殊处理
            # 这里我们先只处理\boxed的情况，保持逻辑简单
            if re.fullmatch(r'[\d\.]+', s.strip()):
                final_answer = s.strip()
                answer_section = f"\n\n#### Final Answer\nThe final answer is $\\boxed{{{final_answer}}}$"
                s += answer_section
                new_item["clean_log"].append(f"step2: 将纯数字解格式化为答案")
            else:
                new_item["clean_log"].append("step2: 未找到 \\boxed 格式的最终答案")

        new_item["final_answer"] = final_answer
        new_item["cleaned_solution"] = s.strip()
        processed_data.append(new_item)

    print(f"📊 最终答案提取统计：在 {len(data)} 个样本中，共找到并处理了 {found_count} 个答案。")
    print(f"✅ 第三轮清洗完成。")
    return processed_data


# ========== 保存前两条结果到TXT文件以供检查 ==========
def save_top2_to_txt_for_review(data_before, data_after, output_path):
    """将前两条数据的清洗前后对比保存到TXT文件。"""
    print(f"📝 正在将前两条结果的清洗前后对比保存到TXT文件...")

    txt_content = "===== 第三轮清洗（提取并标准化最终答案）前后对比 =====\n\n"

    for i in range(min(2, len(data_after))):
        item_before = data_before[i]
        item_after = data_after[i]

        txt_content += f"【样本 {item_after['id'] + 1}】\n"
        txt_content += "=" * 60 + "\n\n"

        txt_content += f"--- 清洗前 Solution ---\n{item_before['cleaned_solution']}\n\n"
        txt_content += f"--- 清洗后 Solution ---\n{item_after['cleaned_solution']}\n\n"

        final_answer_text = item_after['final_answer'] if item_after['final_answer'] is not None else "未提取到"
        txt_content += f"--- 提取的 Final Answer ---\n{final_answer_text}\n\n"

        txt_content += f"--- 更新后的 Clean Log ---\n"
        for log in item_after['clean_log']:
            txt_content += f"- {log}\n"

        txt_content += "\n" + "=" * 80 + "\n\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"✅ 前两条清洗结果对比已保存到: {output_path}")


# ========== 主执行流程 ==========
if __name__ == "__main__":
    step1_data = load_json_data(STEP1_RESULT_PATH)
    if step1_data:
        step2_data = step2_extract_and_format_answer(step1_data)

        with open(STEP2_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(step2_data, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 第三轮清洗结果已成功保存到: {STEP2_SAVE_PATH}")

        save_top2_to_txt_for_review(step1_data, step2_data, TOP2_REVIEW_TXT_PATH)
