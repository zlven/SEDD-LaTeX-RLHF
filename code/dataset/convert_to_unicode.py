import json
import os
from pylatexenc.latex2text import LatexNodes2Text

# ========== 配置参数 ==========
# 第四轮清洗（最终LaTeX标准化）结果的输入路径
FINAL_LATEX_JSON_PATH = "/root/autodl-tmp/dataset_cleaned/step3_latex_normalized.json"

# 最终Unicode数据集的输出目录和文件名
UNICODE_OUTPUT_DIR = "/root/autodl-tmp/sft_dataset"
FINAL_UNICODE_JSONL_PATH = os.path.join(UNICODE_OUTPUT_DIR, "s1k_unicode.jsonl")

os.makedirs(UNICODE_OUTPUT_DIR, exist_ok=True)


# ========== 主执行流程 ==========
def convert_latex_to_unicode():
    """
    读取清洗好的LaTeX格式数据，将其中的LaTeX代码转换为Unicode字符，
    并保存为新的 .jsonl 文件。
    """
    print(f"⏳ 正在加载最终清洗过的LaTeX数据: {FINAL_LATEX_JSON_PATH}")
    try:
        with open(FINAL_LATEX_JSON_PATH, "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件！请确认已完成所有清洗步骤: {FINAL_LATEX_JSON_PATH}")
        return

    # 初始化转换器
    # math_spec='unicode' 会将 \alpha 转换为 α, \geq 转换为 ≥ 等
    latex_converter = LatexNodes2Text(math_spec='unicode')

    print(f"🚀 正在将LaTeX转换为Unicode并生成新的 .jsonl 文件...")
    count = 0
    with open(FINAL_UNICODE_JSONL_PATH, "w", encoding="utf-8") as f:
        for item in cleaned_data:

            # 分别转换 question 和 solution
            try:
                unicode_question = latex_converter.latex_to_text(item["cleaned_question"])
            except Exception as e:
                print(f"⚠️ 警告：转换Question (ID: {item['id']}) 时发生错误，将使用原始内容。错误: {e}")
                unicode_question = item["cleaned_question"]

            try:
                unicode_solution = latex_converter.latex_to_text(item["cleaned_solution"])
            except Exception as e:
                print(f"⚠️ 警告：转换Solution (ID: {item['id']}) 时发生错误，将使用原始内容。错误: {e}")
                unicode_solution = item["cleaned_solution"]

            # 创建新的SFT条目
            sft_entry = {
                "question": unicode_question,
                "solution": unicode_solution,
            }

            f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"🎉 Unicode SFT 数据集生成完毕！")
    print(f"📄 文件路径: {FINAL_UNICODE_JSONL_PATH}")
    print(f"📝 总计写入 {count} 条记录。")

    # 打印一个转换示例以供检查
    if cleaned_data:
        print("\n===== 转换示例 (第一条数据) =====")
        original_q = cleaned_data[0]['cleaned_question']
        converted_q = latex_converter.latex_to_text(original_q)
        print(f"--- 原始 Question (LaTeX) ---\n{original_q}\n")
        print(f"--- 转换后 Question (Unicode) ---\n{converted_q}\n")

        original_s = cleaned_data[0]['cleaned_solution']
        converted_s = latex_converter.latex_to_text(original_s)
        print(f"--- 原始 Solution (LaTeX) ---\n{original_s}\n")
        print(f"--- 转换后 Solution (Unicode) ---\n{converted_s}\n")
        print("=" * 40)


if __name__ == "__main__":
    convert_latex_to_unicode()