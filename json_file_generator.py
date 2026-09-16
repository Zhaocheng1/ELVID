# import os
# import json
#
# # 根路径
# root_dir = "/home/ubuntu/zhaocheng/infrared_visible_video_dataset/generated_visio_npy/npy_files_Train"
# output_ir_json = "ir_dataset.json"
# output_vis_json = "vis_dataset.json"
#
#
# def generate_json(data_type='ir', label=0):
#     dataset_dict = {}
#
#     for subfolder in sorted(os.listdir(root_dir)):
#         subfolder_path = os.path.join(root_dir, subfolder)
#         data_folder = os.path.join(subfolder_path, data_type)
#
#         if not os.path.isdir(data_folder):
#             continue
#
#         for fname in sorted(os.listdir(data_folder)):
#             if fname.endswith('.npy'):
#                 fpath = os.path.join(data_folder, fname)
#                 # 使用完整路径作为键
#                 dataset_dict[fpath] = {
#                     "data": {data_type: fpath},
#                     "label": label
#                 }
#
#     return dataset_dict
#
#
# # 生成两个独立的 json 数据集
# ir_data = generate_json('ir', label=0)
# vis_data = generate_json('vis', label=1)
#
# # 保存到文件
# with open(output_ir_json, "w") as f:
#     json.dump(ir_data, f, indent=2)
#
# with open(output_vis_json, "w") as f:
#     json.dump(vis_data, f, indent=2)
#
# print(f"已生成 JSON：{output_ir_json}（IR）和 {output_vis_json}（VIS）")

import os
import json

ir_root = "/home/ubuntu/zhaocheng/infrared_visible_video_dataset/m3svd_videos/Test_ir"
output_json_path = "test_ir_m3svd_only.json"

result = {}

# 遍历所有子文件夹
for subdir in os.listdir(ir_root):
    sub_path = os.path.join(ir_root, subdir)
    if not os.path.isdir(sub_path):
        continue
    for fname in os.listdir(sub_path):
        if fname.endswith(".npy"):
            key = f"{subdir}_{fname}"
            result[key] = {
                "path": os.path.join(sub_path, fname),
                "folder": subdir,
                "label": 0
            }

# 保存为 JSON 文件
with open(output_json_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"✅ 已生成 JSON 文件: {output_json_path}，共包含 {len(result)} 个 ir 样本")