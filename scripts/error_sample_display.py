from pathlib import Path
import json

# 设置目标文件夹路径
target_dir = Path(
    "/data2/yuyangxin/Agent/outputs/agent_results_confuse_qwen3[qwen3_32b][open_calibration][open_clustering][open_semantic][open_expert]_old/final_output"
)

# 初始化两个列表，分别存储不同GT的错误样本
errors_gt_1 = []
errors_gt_fake = []

# 遍历文件夹下的所有 .json 文件
for json_file in target_dir.glob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pred_result = data["pred_result"]
        gt_label = data["gt_label"]
        expert_model = data["expert_model"]

        # 检查是否预测错误
        if pred_result != gt_label:
            error_count = 0
            for model_result in expert_model.values():
                # 根据阈值0.5划分为1或0
                if model_result >= 0.5:
                    model_result = 1
                else:
                    model_result = 0
                if model_result != gt_label:
                    error_count += 1
            if error_count == len(expert_model):
                print(f"所有专家模型均预测错误，文件: {json_file.name}")
                continue  # 跳过所有专家模型均预测错误的样本

            # 提取需要的字段
            sample_info = {
                "expert_model": data["expert_model"],
                "gt_label": gt_label,
                "pred_result": pred_result,
                "image_path": data["image_path"],
                "json_file": str(json_file),
            }

            # 根据 gt_label 的值分类保存
            # 注意：兼容了数字类型的 1 和字符串类型的 "1"
            if gt_label == 1 or gt_label == "1":
                errors_gt_1.append(sample_info)
            elif gt_label == 0:
                errors_gt_fake.append(sample_info)
            else:
                raise ValueError(f"未知的 gt_label 值: {gt_label}，文件: {json_file.name}")

    except Exception as e:
        print(f"读取文件 {json_file.name} 时出错: {e}")

# 保存 GT 为 1 的错误样本
output_path_1 = Path("error_samples_gt_1.json")
with open(output_path_1, "w", encoding="utf-8") as f:
    json.dump(errors_gt_1, f, indent=4, ensure_ascii=False)

# 保存 GT 为 fake 的错误样本
output_path_fake = Path("error_samples_gt_fake.json")
with open(output_path_fake, "w", encoding="utf-8") as f:
    json.dump(errors_gt_fake, f, indent=4, ensure_ascii=False)

print(f"处理完成。")
print(f"GT=1 的错误样本: {len(errors_gt_1)} 个，保存至 {output_path_1}")
print(f"GT=fake 的错误样本: {len(errors_gt_fake)} 个，保存至 {output_path_fake}")
