import json

# 两个待合并的文件路径
file1 = "./PATH_TO_LOCAL_RESOURCE"
file2 = "./PATH_TO_LOCAL_RESOURCE"

# 输出路径（可以根据需要修改）
output_file = "./PATH_TO_LOCAL_RESOURCE"

# 读取两个 JSON 文件
with open(file1, "r") as f1, open(file2, "r") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# 合并策略：根据类型合并
if isinstance(data1, list) and isinstance(data2, list):
    merged_data = data1 + data2
elif isinstance(data1, dict) and isinstance(data2, dict):
    merged_data = {**data1, **data2}  # 注意：key 重复会被后者覆盖
else:
    raise ValueError("两个 JSON 文件类型不一致，无法合并")

# 保存合并后的结果
with open(output_file, "w") as f_out:
    json.dump(merged_data, f_out, indent=2)

print(f"合并完成，结果已保存到：{output_file}")
