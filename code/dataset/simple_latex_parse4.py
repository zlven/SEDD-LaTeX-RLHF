import re
import json
import os

# ========== 配置参数 (请确保与你的环境匹配) ==========
STEP2_RESULT_PATH = "/root/autodl-tmp/dataset_cleaned/step2_answer_extracted.json"
CLEAN_DIR = "/root/autodl-tmp/dataset_cleaned"
STEP3_SAVE_PATH = os.path.join(CLEAN_DIR, "step3_latex_normalized.json")
TOP2_REVIEW_TXT_PATH = os.path.join(CLEAN_DIR, "step3_top2_review.txt")

os.makedirs(CLEAN_DIR, exist_ok=True)


# ========== 工具函数：加载JSON数据 ==========
def load_json_data(file_path):
    """从指定路径加载JSON文件"""
    print(f"⏳ 正在加载第三轮清洗后的数据: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 数据加载成功，共 {len(data)} 条样本。")
        return data
    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件！请检查路径是否正确: {file_path}")
        return None


# ========== 第四步（最终修正版）：LaTeX 标准化与统一 ==========
def step3_normalize_latex(data):
    """
    第四轮清洗：
    使用替换函数来稳健地修正LaTeX格式。
    """
    print("\n🚀 开始第四轮清洗：LaTeX 标准化与统一...")
    processed_data = []

    subscript_commands = ['delta', 'alpha', 'beta', 'gamma', 'sum', 'int', 'lim', 'log']
    unicode_to_latex = {
        '∈': r' \in ', '≥': r' \geq ', '≤': r' \leq ', '≠': r' \neq ',
        '→': r' \to ', '×': r' \times ', '÷': r' \div ', '…': r' \ldots '
    }

    for item in data:
        new_item = json.loads(json.dumps(item))

        for key in ["cleaned_question", "cleaned_solution"]:
            if key not in new_item or not new_item[key]:
                continue
            text = new_item[key]

            # 1. 统一数学环境定界符
            text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
            text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

            # 2. 【核心修正】使用替换函数来修正错误的下标语法
            for cmd in subscript_commands:
                pattern = rf'\\{cmd}\{{(.*?)\}}'

                # 定义一个替换函数
                def create_replacement(match, command=cmd):
                    # 将命令名作为默认参数传入，以固定其在循环中的值
                    content = match.group(1)
                    return f'\\{command}_{{{content}}}'

                text = re.sub(pattern, create_replacement, text)

            # 3. 替换Unicode数学符号
            for uni_char, latex_cmd in unicode_to_latex.items():
                text = text.replace(uni_char, latex_cmd)

            # 4. 最终清理多余空格
            text = re.sub(r'\s+', ' ', text).strip()

            new_item[key] = text

        new_item["clean_log"].append("step3(final-revised): 标准化LaTeX格式")
        processed_data.append(new_item)

    print(f"✅ 第四轮清洗完成。")
    return processed_data


# ... (保存TXT文件的函数和主流程代码保持不变) ...
def save_top2_to_txt_for_review(data_before, data_after, output_path):
    print(f"📝 正在将前两条结果的清洗前后对比保存到TXT文件...")
    txt_content = "===== 第四轮清洗（LaTeX标准化）前后对比 =====\n\n"
    for i in range(min(2, len(data_after))):
        item_before = data_before[i]
        item_after = data_after[i]
        txt_content += f"【样本 {item_after['id'] + 1}】\n"
        txt_content += "=" * 60 + "\n\n"
        txt_content += f"--- 清洗前 Solution ---\n{item_before['cleaned_solution']}\n\n"
        txt_content += f"--- 清洗后 Solution ---\n{item_after['cleaned_solution']}\n\n"
        txt_content += "\n" + "=" * 80 + "\n\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"✅ 前两条清洗结果对比已保存到: {output_path}")


if __name__ == "__main__":
    step2_data = load_json_data(STEP2_RESULT_PATH)
    if step2_data:
        step3_data = step3_normalize_latex(step2_data)
        with open(STEP3_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(step3_data, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 第四轮清洗结果已成功保存到: {STEP3_SAVE_PATH}")
        save_top2_to_txt_for_review(step2_data, step3_data, TOP2_REVIEW_TXT_PATH)