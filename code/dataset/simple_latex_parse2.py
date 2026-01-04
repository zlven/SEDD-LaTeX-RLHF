import re
import json
import os

# ========== 配置参数 (请确保与你的环境匹配) ==========
# 第一轮清洗结果的输入路径
STEP0_RESULT_PATH = "/root/autodl-tmp/dataset_cleaned/step0_standardized.json"
# 第二轮清洗结果的输出目录
CLEAN_DIR = "/root/autodl-tmp/dataset_cleaned"
# 第二轮清洗结果的保存路径
STEP1_SAVE_PATH = os.path.join(CLEAN_DIR, "step1_format_fixed.json")

os.makedirs(CLEAN_DIR, exist_ok=True)


# ========== 工具函数：加载JSON数据 ==========
def load_json_data(file_path):
    """从指定路径加载JSON文件"""
    print(f"⏳ 正在加载第一轮清洗后的数据: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 数据加载成功，共 {len(data)} 条样本。")
        return data
    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件！请检查路径是否正确: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ 错误：JSON文件格式不正确，无法解析: {file_path}")
        return None


# ========== 第二步（修正版）：修正LaTeX格式并移除Markdown ==========
def step1_fix_formatting_and_markdown_revised(data):
    """
    第二轮清洗（修正版）：
    采用更稳健的策略来规范化格式，避免删除必要的空格。
    """
    print("\n🚀 开始第二轮清洗（修正版）：修正LaTeX格式并移除Markdown...")
    processed_data = []

    for item in data:
        q = item["cleaned_question"]
        s = item["cleaned_solution"]

        texts_to_clean = {"question": q, "solution": s}
        cleaned_texts = {}

        for key, text in texts_to_clean.items():
            # 1. 替换非标准LaTeX命令
            text = text.replace("\\minus{}", "-")

            # 2. 移除Markdown标记
            text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'\1\2', text)  # 加粗
            text = re.sub(r'\*(.*?)\*|_(.*?)_', r'\1\2', text)  # 斜体

            # 3. 规范化LaTeX定界符周围的空格（核心修正）
            #   - 在所有 $ 定界符周围添加空格
            text = re.sub(r'\$', ' $ ', text)
            #   - 在 \( 和 \) 定界符周围添加空格
            text = re.sub(r'\\\((.+?)\\\)', r' \\( \1 \\) ', text)
            #   - 在 \[ 和 \] 定界符周围添加空格
            text = re.sub(r'\\\[(.+?)\\\]', r' \\[ \1 \\] ', text)

            # 4. 移除LaTeX定界符内部的多余空格
            #   - 清理 $ ... $ 内部
            text = re.sub(r'\$\s+(.*?)\s+\$', lambda m: f'${m.group(1).strip()}$', text)
            #   - 清理 \( ... \) 内部
            text = re.sub(r'\\\(\s+(.*?)\s+\\\)', lambda m: f'\\({m.group(1).strip()}\\)', text)
            #   - 清理 \[ ... \] 内部
            text = re.sub(r'\\\[\s+(.*?)\s+\\\]', lambda m: f'\\[{m.group(1).strip()}\\]', text)

            # 5. 最后，合并所有多余的空格
            text = re.sub(r'\s+', ' ', text).strip()

            cleaned_texts[key] = text

        # 更新数据项
        item["cleaned_question"] = cleaned_texts["question"]
        item["cleaned_solution"] = cleaned_texts["solution"]
        item["clean_log"].append("step1(revised): 修正LaTeX格式并移除Markdown标记")
        processed_data.append(item)

    print(f"✅ 第二轮清洗完成，处理了 {len(processed_data)} 条数据。")
    return processed_data


# ========== 主执行流程 ==========
if __name__ == "__main__":
    # 1. 加载第一轮清洗的结果
    step0_data = load_json_data(STEP0_RESULT_PATH)

    if step0_data:
        # 2. 执行修正后的第二轮清洗
        step1_data = step1_fix_formatting_and_markdown_revised(step0_data)

        # 3. 保存第二轮清洗的结果
        with open(STEP1_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(step1_data, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 第二轮清洗结果已成功保存到: {STEP1_SAVE_PATH}")
