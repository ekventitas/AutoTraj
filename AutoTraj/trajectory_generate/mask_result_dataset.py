import json
import re
from pathlib import Path

# ===== 配置 =====
input_file = Path("/hy-tmp/AutoTraj/trajectory_generate/datasets/sft_high_quality_trajectory.json")
output_file = Path("/hy-tmp/AutoTraj/trajectory_generate/datasets/sft_high_quality_trajectory_masked.json")

# 正则匹配 <result>...</result>
result_pattern = re.compile(r"<result>.*?</result>", re.DOTALL)

# ===== 处理数据集 =====
with input_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
for i, item in enumerate(data):
    new_item = item.copy()

    # 如果 output 是字符串类型
    if isinstance(new_item.get("output", ""), str):
        # 删除 <result> 标签里的内容，只保留标签
        new_output = result_pattern.sub("<result></result>", new_item["output"])
        new_item["output"] = new_output
    else:
        print(f"Warning: sample {i} has no 'output' or not a string, skipping")

    new_data.append(new_item)

# ===== 保存处理后的数据集 =====
with output_file.open("w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"完成处理，输出文件: {output_file}")
