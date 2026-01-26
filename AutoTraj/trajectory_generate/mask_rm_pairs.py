import json
import re

# 输入文件路径
input_file = "/hy-tmp/AutoTraj/trajectory_generate/datasets/rm_compare_pairs.json"
# 输出文件路径
output_file = "/hy-tmp/AutoTraj/trajectory_generate/datasets/rm_compare_pairs_masked.json"

# 用于清空 <result> 标签中的内容
def remove_result_content(text: str) -> str:
    # 替换 <result>...</result> 中的内容为空
    return re.sub(r"<result>.*?</result>", "<result></result>", text, flags=re.DOTALL)

# 读取原始数据集
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 遍历每一条数据
for item in data:
    if "chosen" in item and item["chosen"]:
        item["chosen"] = remove_result_content(item["chosen"])
    if "rejected" in item and item["rejected"]:
        item["rejected"] = remove_result_content(item["rejected"])

# 保存清洗后的数据集
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"清洗完成，新数据集已保存到: {output_file}")
