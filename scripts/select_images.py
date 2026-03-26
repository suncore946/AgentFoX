import json
import random
import os


def select_images(input_json_path, output_json_path, num_per_label=2000):
    """
    从JSON文件中读取图像数据，根据gt_label分为0和1两类，
    从每一类中随机挑选指定数量的图像（最多num_per_label个），
    并保存到新的JSON文件中。

    :param input_json_path: 输入JSON文件路径
    :param output_json_path: 输出JSON文件路径
    :param num_per_label: 每类挑选的数量（默认2000）
    """
    # 读取输入JSON文件
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input file {input_json_path} does not exist.")

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 分离gt_label为0和1的图像
    label_0_images = {key: value for key, value in data.items() if value.get("gt_label") == 0}
    label_1_images = {key: value for key, value in data.items() if value.get("gt_label") == 1}

    print(f"Found {len(label_0_images)} images with gt_label=0")
    print(f"Found {len(label_1_images)} images with gt_label=1")

    # 从label_0中随机挑选
    if len(label_0_images) > num_per_label:
        selected_0_keys = random.sample(list(label_0_images.keys()), num_per_label)
    else:
        selected_0_keys = list(label_0_images.keys())
        print(f"Warning: Only {len(selected_0_keys)} images available for gt_label=0, selecting all.")
    selected_0 = {key: label_0_images[key] for key in selected_0_keys}

    # 从label_1中随机挑选
    if len(label_1_images) > num_per_label:
        selected_1_keys = random.sample(list(label_1_images.keys()), num_per_label)
    else:
        selected_1_keys = list(label_1_images.keys())
        print(f"Warning: Only {len(selected_1_keys)} images available for gt_label=1, selecting all.")
    selected_1 = {key: label_1_images[key] for key in selected_1_keys}

    # 合并选中的图像
    selected_data = {**selected_0, **selected_1}

    # 保存到输出JSON文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(selected_data, f, indent=4, ensure_ascii=False)

    print(f"Selected {len(selected_data)} images and saved to {output_json_path}")


# 示例用法：替换为您的文件路径
if __name__ == "__main__":
    input_path = r"E:\桌面\forensic_agent\adaptive_forensic_agent\configs\combined_results.json"
    output_path = r"E:\桌面\forensic_agent\adaptive_forensic_agent\configs\selected_results_100.json"
    select_images(input_path, output_path, num_per_label=100)
