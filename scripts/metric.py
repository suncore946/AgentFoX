from collections import defaultdict
import json
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_metrics_for_single_json(file_path, dataset_name="base", sub_dataset=""):
    """
    计算单个JSON文件的分类指标

    Args:
        file_path (str): JSON文件路径

    Returns:
        dict: 包含文件名和各项指标的字典
    """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 首先收集所有图像的gt_labels和图像顺序（以保持一致）
        image_paths = sorted(data.keys())  # 排序以确保顺序一致
        gt_labels = []
        for path in image_paths:
            gt_labels.append(data[path]["gt_label"])

        # 提取所有模型名称（从第一个图像中获取）
        if not image_paths:
            raise ValueError("No images found in the data.")
        first_image_data = data[image_paths[0]]
        model_names = [key for key in first_image_data if key != "gt_label"]

        # 初始化结果字典
        result = {}
        for model in model_names:
            pred_labels = []
            for path in image_paths:
                image_data = data[path]
                if model not in image_data:
                    raise ValueError(f"Model '{model}' not found for image '{path}'.")
                prob = image_data[model]["pred_prob"]
                pred_labels.append(prob)
            result[model] = {"pred_labels": pred_labels, "gt_labels": gt_labels}  # 所有模型共享相同的gt_labels
    except Exception as e:
        raise ValueError(f"读取文件 {file_path} 时出错: {e}")

    ret = []
    for model_name, labels in result.items():
        pred_labels = labels["pred_labels"]
        gt_labels = labels["gt_labels"]
        # 转换为numpy数组
        pred_labels = np.array(pred_labels)
        gt_labels = np.array(gt_labels)

        # 将预测概率转换为二分类结果 (阈值0.5)
        pred_binary = (pred_labels >= 0.5).astype(int)

        # 计算总体指标
        accuracy = accuracy_score(gt_labels, pred_binary)
        f1 = f1_score(gt_labels, pred_binary, zero_division=0)

        # 添加判断
        if len(np.unique(gt_labels)) < 2 or len(np.unique(pred_labels)) < 2:
            auc = None
        else:
            auc = roc_auc_score(gt_labels, pred_labels)

        # 分别计算real和fake的指标
        real_mask = gt_labels == 0
        fake_mask = gt_labels == 1

        # Real类别指标 (gt_label = 0)
        if np.sum(real_mask) > 0:
            real_pred = pred_binary[real_mask]
            real_gt = gt_labels[real_mask]
            real_acc = accuracy_score(real_gt, real_pred)
            real_f1 = f1_score(real_gt, real_pred, pos_label=0, zero_division=0)
            real_count = np.sum(real_mask)
        else:
            real_acc = real_f1 = 0.0
            real_count = 0

        # Fake类别指标 (gt_label = 1)
        if np.sum(fake_mask) > 0:
            fake_pred = pred_binary[fake_mask]
            fake_gt = gt_labels[fake_mask]
            fake_acc = accuracy_score(fake_gt, fake_pred)
            fake_f1 = f1_score(fake_gt, fake_pred, pos_label=1, zero_division=0)
            fake_count = np.sum(fake_mask)
        else:
            fake_acc = fake_f1 = 0.0
            fake_count = 0
        ret.append(
            {
                "method": model_name,
                "dataset": dataset_name,
                "sub_name": sub_dataset if sub_dataset else file_path.stem.split("_")[0],
                "total_samples": len(pred_labels),
                "overall_accuracy": accuracy,
                "overall_f1": f1,
                "overall_auc": auc,
                "real_samples": real_count,
                "real_accuracy": real_acc,
                "real_f1": real_f1,
                "fake_samples": fake_count,
                "fake_accuracy": fake_acc,
                "fake_f1": fake_f1,
            }
        )

    # 返回为csv文件
    csv_data = pd.DataFrame(ret)
    # 保存csv_data
    with open(file_path.with_suffix(".csv"), "w", encoding="utf-8") as f:
        csv_data.to_csv(f, index=False)
    print(f"CSV文件已保存到: {file_path.with_suffix('.csv')}")
    return csv_data


def calculate_summary_stats(df):
    """
    计算汇总统计信息
    """
    return {
        "total_samples": df["total_samples"].sum(),
        "total_real_samples": df["real_samples"].sum(),  # 新增：总Real样本数
        "total_fake_samples": df["fake_samples"].sum(),  # 新增：总Fake样本数
        "avg_overall_acc": df["overall_accuracy"].mean(),
        "avg_overall_f1": df["overall_f1"].mean(),
        "avg_overall_auc": df["overall_auc"].mean(),
        "avg_real_acc": df["real_accuracy"].mean(),
        "avg_fake_acc": df["fake_accuracy"].mean(),
    }


def calculate_metrics_from_json_folder_to_csv(folder_path, output_csv_path=None, filters=None):
    """
    遍历文件夹下的所有JSON文件，计算每个文件的分类指标并导出为CSV

    Args:
        folder_path (str or Path): 包含JSON文件的文件夹路径
        output_csv_path (str or Path): 输出CSV文件路径，如果为None则自动生成
        filters (list of str): 只处理文件名中包含这些字符串中任意一个的JSON文件

    Returns:
        str: CSV文件路径
    """
    results_list = []

    def traverse_directory(current_path, model_name, dataset_name, sub_dataset=None):
        """递归遍历目录并处理JSON文件"""
        # 处理当前目录中的JSON文件
        for json_file in current_path.glob("*.json"):
            if filters and not any(f in json_file.as_posix() for f in filters):
                continue  # 跳过不匹配的文件
            sub_name = _get_sub_name(json_file, dataset_name, sub_dataset)
            result = calculate_metrics_for_single_json(json_file, model_name, dataset_name, sub_name)
            if result is not None:
                results_list.append(result)

        # 递归处理子目录
        for sub_folder in sorted(current_path.iterdir()):
            if sub_folder.is_dir() and list(sub_folder.glob("*.json")):
                new_sub_dataset = _get_new_sub_dataset(sub_dataset, dataset_name, sub_folder.name)
                traverse_directory(sub_folder, model_name, dataset_name, new_sub_dataset)

    def _get_sub_name(json_file, dataset_name, sub_dataset):
        """获取子数据集名称"""
        base_name = json_file.stem.split("_")[0]
        if sub_dataset == dataset_name:
            return base_name
        else:
            return f"{sub_dataset}/{base_name}"

    def _get_new_sub_dataset(sub_dataset, dataset_name, folder_name):
        """获取新的子数据集路径"""
        if sub_dataset == dataset_name:
            return folder_name
        else:
            return f"{sub_dataset}/{folder_name}"

    # 遍历主文件夹的子目录
    for sub_folder in folder_path.iterdir():
        if sub_folder.is_dir():
            traverse_directory(sub_folder, folder_path.name, sub_folder.name, sub_folder.name)

    if len(results_list) == 0:
        print("未找到有效的JSON文件或数据")
        return None

    # 转换为DataFrame
    df = pd.DataFrame(results_list)

    # 重新排列列的顺序，使其更易读
    column_order = [
        "method",
        "dataset",
        "sub_name",
        "total_samples",
        "overall_accuracy",
        "overall_f1",
        "overall_auc",
        "real_samples",
        "real_accuracy",
        "real_f1",
        "fake_samples",
        "fake_accuracy",
        "fake_f1",
    ]
    df = df[column_order]

    # 保存为CSV
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    # 计算并添加汇总统计
    summary_stats = calculate_summary_stats(df)

    # 将汇总统计也添加到CSV中
    with open(output_csv_path, "a", encoding="utf-8") as f:
        f.write("\n\nSummary Statistics:\n")
        f.write(f"Total Files,{len(df)}\n")
        f.write(f"Total Samples,{summary_stats['total_samples']}\n")
        f.write(f"Total Real Samples,{summary_stats['total_real_samples']}\n")  # Added
        f.write(f"Total Fake Samples,{summary_stats['total_fake_samples']}\n")  # Added
        f.write(f"Average Overall Accuracy,{summary_stats['avg_overall_acc']:.4f}\n")
        f.write(f"Average Overall F1,{summary_stats['avg_overall_f1']:.4f}\n")
        f.write(f"Average Overall AUC,{summary_stats['avg_overall_auc']:.4f}\n")
        f.write(f"Average Real Accuracy,{summary_stats['avg_real_acc']:.4f}\n")
        f.write(f"Average Fake Accuracy,{summary_stats['avg_fake_acc']:.4f}\n")

    return output_csv_path


def calculate_metrics_from_json_folder(folder_path, filters=None):
    """
    遍历文件夹下的所有JSON文件，计算分类指标 (保持原有功能)

    Args:
        folder_path (str): 文件夹路径
        filters (list of str): 只处理文件名中包含这些字符串中任意一个的JSON文件
    """
    all_pred_labels = []
    all_gt_labels = []

    # 遍历文件夹下的所有JSON文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            if filters and not any(f in filename for f in filters):
                continue  # 跳过不匹配的文件
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 提取每个样本的pred_label和gt_label
                for key, value in data.items():
                    if isinstance(value, dict) and "pred_label" in value and "gt_label" in value:
                        pred_label = value["pred_label"]
                        gt_label = value["gt_label"]

                        all_pred_labels.append(pred_label)
                        all_gt_labels.append(gt_label)

            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")

    if len(all_pred_labels) == 0:
        print("未找到有效的预测数据")
        return None

    # 转换为numpy数组
    pred_labels = np.array(all_pred_labels)
    gt_labels = np.array(all_gt_labels)

    # 将预测概率转换为二分类结果 (阈值0.5)
    pred_binary = (pred_labels >= 0.5).astype(int)

    # 计算指标
    accuracy = accuracy_score(gt_labels, pred_binary)
    f1 = f1_score(gt_labels, pred_binary)
    auc = roc_auc_score(gt_labels, pred_labels)

    # 分别计算real和fake的指标
    real_mask = gt_labels == 0
    fake_mask = gt_labels == 1

    # Real类别指标 (gt_label = 0)
    if np.sum(real_mask) > 0:
        real_pred = pred_binary[real_mask]
        real_gt = gt_labels[real_mask]
        real_acc = accuracy_score(real_gt, real_pred)
        real_f1 = f1_score(real_gt, real_pred, pos_label=0, zero_division=0)
    else:
        real_acc = real_f1 = 0.0

    # Fake类别指标 (gt_label = 1)
    if np.sum(fake_mask) > 0:
        fake_pred = pred_binary[fake_mask]
        fake_gt = gt_labels[fake_mask]
        fake_acc = accuracy_score(fake_gt, fake_pred)
        fake_f1 = f1_score(fake_gt, fake_pred, pos_label=1, zero_division=0)
    else:
        fake_acc = fake_f1 = 0.0

    results = {
        "overall": {"accuracy": accuracy, "f1_score": f1, "auc": auc, "total_samples": len(pred_labels)},
        "real_class": {"accuracy": real_acc, "f1_score": real_f1, "sample_count": np.sum(real_mask)},
        "fake_class": {"accuracy": fake_acc, "f1_score": fake_f1, "sample_count": np.sum(fake_mask)},
    }

    return results


def print_metrics(results):
    """
    打印指标结果
    """
    if results is None:
        return

    print("=" * 50)
    print("分类指标统计")
    print("=" * 50)

    print(f"总体指标 (总样本数: {results['overall']['total_samples']}):")
    print(f"  准确率 (ACC): {results['overall']['accuracy']:.4f}")
    print(f"  F1分数: {results['overall']['f1_score']:.4f}")
    print(f"  AUC: {results['overall']['auc']:.4f}")

    print(f"\nReal类别指标 (样本数: {results['real_class']['sample_count']}):")
    print(f"  准确率 (ACC): {results['real_class']['accuracy']:.4f}")
    print(f"  F1分数: {results['real_class']['f1_score']:.4f}")

    print(f"\nFake类别指标 (样本数: {results['fake_class']['sample_count']}):")
    print(f"  准确率 (ACC): {results['fake_class']['accuracy']:.4f}")
    print(f"  F1分数: {results['fake_class']['f1_score']:.4f}")


def main(args):
    """
    命令行主函数
    """
    # 处理单个文件模式
    if args.single_file:
        file_path = Path(args.single_file)
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return
        # 对于单个文件，如果指定了filters，也可以应用过滤，但这里假设用户指定了文件就直接处理
        if args.filters and not any(f in file_path.name for f in args.filters):
            print(f"文件 {file_path.name} 不匹配过滤条件，跳过")
            return
        return calculate_metrics_for_single_json(file_path)

    assert args.folder_path is not None
    assert os.path.exists(args.folder_path), f"错误: 文件夹不存在: {args.folder_path}"

    # 处理文件夹模式
    if args.mode in ["print", "both"]:
        # 计算并打印指标
        if args.aggregation:
            folder_path = Path(args.folder_path)
            # 遍历文件夹下的所有子文件夹路径
            for sub_folder in folder_path.iterdir():
                if sub_folder.is_dir():
                    print(f"\nProcessing folder: {sub_folder.name}")
                    results = calculate_metrics_from_json_folder(sub_folder, filters=args.filters)
                    print_metrics(results)
        else:
            results = calculate_metrics_from_json_folder(args.folder_path, filters=args.filters)
            print_metrics(results)

    if args.mode in ["csv", "both"]:
        # 导出CSV
        if args.aggregation:
            folder_path = Path(args.folder_path)
            # 遍历文件夹下的所有子文件夹路径
            for sub_folder in folder_path.iterdir():
                if sub_folder.is_dir():
                    if args.output is None:
                        output_csv_path = folder_path / f"{sub_folder.name}_metrics_results.csv"
                    else:
                        output_csv_path = Path(args.output) / f"{sub_folder.name}_metrics_results.csv"
                    if not output_csv_path.parent.exists():
                        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    if output_csv_path.exists():
                        print(f"警告: 输出文件已存在，将跳过: {output_csv_path}")
                    else:
                        csv_path = calculate_metrics_from_json_folder_to_csv(sub_folder, output_csv_path, filters=args.filters)
                        print(f"\nCSV文件已保存到: {csv_path}")
        else:
            output_path = args.output if args.output else None
            output_csv_path = (
                Path(args.folder_path) / "metrics_results.csv" if output_path is None else Path(output_path) / "metrics_results.csv"
            )
            csv_path = calculate_metrics_from_json_folder_to_csv(Path(args.folder_path), output_csv_path, filters=args.filters)
            if csv_path:
                print(f"\nCSV文件已保存到: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算JSON文件夹中的分类指标")
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--folder-path", help="包含JSON文件的文件夹路径")  # 原 folder_path 改为可选参数，无默认值
    mutex.add_argument("--single-file", help="计算单个JSON文件的指标")  # 无默认值

    parser.add_argument(
        "--mode", choices=["print", "csv", "both"], default="csv", help="输出模式: print(仅打印), csv(导出CSV), both(打印+导出CSV)"
    )
    parser.add_argument("--output", "-o", default="./aigc_model/output", help="输出CSV文件路径(仅在csv或both模式下有效)")
    parser.add_argument("--aggregation", action="store_true", default=False, help="是否是多个模型的聚合")
    parser.add_argument(
        "--filters", nargs="+", default=None, help="只处理文件名中包含这些字符串中任意一个的JSON文件（多个字符串以空格分隔）"
    )

    args = parser.parse_args()
    print(f"参数: {args}")
    main(args)
