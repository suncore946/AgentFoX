#!/usr/bin/env python3
"""
统计 expert_model 中每个模型的预测情况，以及不同组合的占比和准确性
"""

import json
import os
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def load_json_files(directory):
    """加载目录下所有 JSON 文件"""
    json_files = []
    directory = Path(directory)
    
    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_files.append(data)
        except Exception as e:
            print(f"⚠️ 读取文件 {json_file} 时出错: {e}")
            continue
    
    return json_files

def get_model_predictions(expert_model, threshold=0.5):
    """
    获取每个模型的预测结果
    如果分数 < 0.5，则预测为自然图像（正确预测为自然图像）
    如果分数 >= 0.5，则预测为 AI 生成（正确预测为 AI 生成）
    
    返回: dict, key 为模型名，value 为 True（预测正确）或 False（预测错误）
    """
    predictions = {}
    for model_name, score in expert_model.items():
        pred_label = 0 if score < threshold else 1
        predictions[model_name] = pred_label
    
    return predictions

def check_pred_result_correct(pred_result, gt_label):
    """
    检查 pred_result 是否正确
    如果 pred_result == 0 且 gt_label == 0，则正确
    如果 pred_result == 1 且 gt_label == 1，则正确
    """
    return pred_result == gt_label

def calculate_f1(tp, fp, tn, fn):
    """计算 F1 分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def analyze_predictions(data_list):
    """分析所有样本的预测情况"""
    
    # 获取所有模型名称
    all_models = set()
    for data in data_list:
        if 'expert_model' in data:
            all_models.update(data['expert_model'].keys())
    all_models = sorted(list(all_models))
    
    print(f"📊 检测到的模型: {', '.join(all_models)}")
    print(f"📁 总样本数: {len(data_list)}\n")
    
    # 统计每个模型的预测正确性和混淆矩阵
    model_correct = defaultdict(int)
    model_total = defaultdict(int)
    model_confusion = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
    
    # 统计所有可能的组合情况
    combination_stats = defaultdict(lambda: {
        'total': 0, 
        'correct': 0,
        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
    })
    
    for data in data_list:
        if 'expert_model' not in data or 'gt_label' not in data:
            continue
        
        expert_model = data['expert_model']
        gt_label = data['gt_label']
        pred_result = data.get('pred_result', None)
        
        # 获取每个模型的预测结果
        model_preds = get_model_predictions(expert_model)
        
        # 统计每个模型的正确性和混淆矩阵
        for model_name, pred_label in model_preds.items():
            model_total[model_name] += 1
            if pred_label == gt_label:
                model_correct[model_name] += 1
            
            # 计算混淆矩阵
            if pred_label == 1 and gt_label == 1:
                model_confusion[model_name]['tp'] += 1
            elif pred_label == 1 and gt_label == 0:
                model_confusion[model_name]['fp'] += 1
            elif pred_label == 0 and gt_label == 0:
                model_confusion[model_name]['tn'] += 1
            elif pred_label == 0 and gt_label == 1:
                model_confusion[model_name]['fn'] += 1
        
        # 构建组合键：表示每个模型的预测是否正确
        combination_key = tuple(
            (model_preds.get(model, None) == gt_label) 
            for model in all_models
            if model in model_preds
        )
        
        # 统计该组合的总数和正确数（基于 pred_result）
        combination_stats[combination_key]['total'] += 1
        if pred_result is not None:
            if check_pred_result_correct(pred_result, gt_label):
                combination_stats[combination_key]['correct'] += 1
            
            # 计算组合的混淆矩阵（基于 pred_result）
            if pred_result == 1 and gt_label == 1:
                combination_stats[combination_key]['tp'] += 1
            elif pred_result == 1 and gt_label == 0:
                combination_stats[combination_key]['fp'] += 1
            elif pred_result == 0 and gt_label == 0:
                combination_stats[combination_key]['tn'] += 1
            elif pred_result == 0 and gt_label == 1:
                combination_stats[combination_key]['fn'] += 1
    
    return all_models, model_correct, model_total, model_confusion, combination_stats

def format_combination_key(combination_key, model_names):
    """格式化组合键为可读字符串"""
    correct_models = []
    wrong_models = []
    
    for i, is_correct in enumerate(combination_key):
        if i < len(model_names):
            if is_correct:
                correct_models.append(model_names[i])
            else:
                wrong_models.append(model_names[i])
    
    if correct_models and wrong_models:
        return f"{'+'.join(correct_models)}✓, {'+'.join(wrong_models)}✗"
    elif correct_models:
        return f"{'+'.join(correct_models)}✓ (全部正确)"
    elif wrong_models:
        return f"{'+'.join(wrong_models)}✗ (全部错误)"
    else:
        return "无模型"

def print_results(all_models, model_correct, model_total, model_confusion, combination_stats, total_samples):
    """打印统计结果"""
    
    print("=" * 80)
    print("📈 各模型预测指标")
    print("=" * 80)
    print(f"{'模型':<15s} {'准确率':<10s} {'精确率':<10s} {'召回率':<10s} {'F1分数':<10s}")
    print("-" * 80)
    for model in all_models:
        correct = model_correct.get(model, 0)
        total = model_total.get(model, 0)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        confusion = model_confusion.get(model, {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        precision, recall, f1 = calculate_f1(
            confusion['tp'], confusion['fp'], 
            confusion['tn'], confusion['fn']
        )
        
        print(f"{model:15s} {accuracy:>6.2f}%   {precision*100:>6.2f}%   {recall*100:>6.2f}%   {f1*100:>6.2f}%")
    print()
    
    print("=" * 80)
    print("📊 模型预测组合统计（按占比排序）")
    print("=" * 80)
    print("说明：✓ 表示该模型预测正确，✗ 表示该模型预测错误")
    print("      准确率和F1基于 pred_result 与 gt_label 的比较")
    print("-" * 80)
    
    # 按总数排序
    sorted_combinations = sorted(
        combination_stats.items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )
    
    print(f"{'组合情况':<50s} {'样本数':<8s} {'占比':<8s} {'准确率':<8s} {'F1分数':<8s}")
    print("-" * 80)
    
    for combination_key, stats in sorted_combinations:
        total_count = stats['total']
        correct_count = stats['correct']
        percentage = (total_count / total_samples * 100) if total_samples > 0 else 0
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        # 计算 F1 分数
        precision, recall, f1 = calculate_f1(
            stats['tp'], stats['fp'],
            stats['tn'], stats['fn']
        )
        
        combination_str = format_combination_key(combination_key, all_models)
        print(f"{combination_str:<50s} {total_count:<8d} {percentage:>6.2f}%  {accuracy:>6.2f}%  {f1*100:>6.2f}%")
    
    print()
    print("=" * 80)
    print("📋 详细统计信息")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"有效样本数（有 expert_model 和 gt_label）: {sum(s['total'] for s in combination_stats.values())}")
    print()

# ========================
# 🖼️ 美化版 UpSet 风格图表函数
# ========================

def plot_upset_style_chart(all_models, combination_stats, total_samples, top_n=20, save_dir=None):
    """生成类似UpSet的图表，包含ACC/F1折线图 + 点阵图"""
    print(f"🎨 正在生成 UpSet 风格图表（显示前{top_n}个组合）...")

    # 计算每个组合的ACC和F1，并按ACC排序
    combinations_with_metrics = []
    for combination_key, stats in combination_stats.items():
        if stats['total'] > 0:
            total_count = stats['total']
            correct_count = stats['correct']
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            
            precision, recall, f1 = calculate_f1(
                stats['tp'], stats['fp'],
                stats['tn'], stats['fn']
            )
            
            combinations_with_metrics.append({
                'key': combination_key,
                'stats': stats,
                'accuracy': accuracy,
                'f1': f1 * 100,
                'total': total_count
            })

    # 按ACC降序排序，取前N个
    sorted_combinations = sorted(
        combinations_with_metrics,
        key=lambda x: x['accuracy'],
        reverse=True
    )[:top_n]

    if not sorted_combinations:
        print("⚠️ 没有足够的组合数据来生成图表")
        return None

    # 准备数据
    combination_data = []
    sizes = []
    accuracies = []
    f1_scores = []
    combination_labels = []

    for combo_info in sorted_combinations:
        combination_key = combo_info['key']
        stats = combo_info['stats']
        sizes.append(stats['total'])
        accuracies.append(combo_info['accuracy'])
        f1_scores.append(combo_info['f1'])

        # 构建每个模型的预测正确性列表
        model_correct = [combination_key[i] for i in range(len(all_models))]
        combination_data.append(model_correct)

        # 生成组合标签（用于x轴，不显示）
        combination_str = format_combination_key(combination_key, all_models)
        if len(combination_str) > 25:
            combination_str = combination_str[:22] + "..."
        combination_labels.append(combination_str)

    # 创建图表 - 2个子图：上方折线图，下方点阵图
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1])

    # 上方：ACC 和 F1 折线图
    ax0 = fig.add_subplot(gs[0])

    x_pos = np.arange(len(accuracies))

    # 绘制折线图
    line1 = ax0.plot(x_pos, accuracies, 'o-', color='#377eb8', linewidth=2.5,
                     markersize=8, label='Accuracy (%)', markerfacecolor='#377eb8',
                     markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    line2 = ax0.plot(x_pos, f1_scores, 's-', color='#ec0629', linewidth=2.5,
                     markersize=8, label='F1 (%)', markerfacecolor='#ec0629',
                     markeredgecolor='white', markeredgewidth=1.5, zorder=3)

    # 添加数值标签（调整位置避免重叠）
    for i in range(len(accuracies)):
        if i > 9:
            # Accuracy 标签放在点上方
            ax0.text(i, accuracies[i] - 9, f'{accuracies[i]:.1f}',
                    ha='center', va='bottom', fontsize=12, color='#377eb8', fontweight='bold')
            # F1 标签放在点下方
            ax0.text(i, f1_scores[i] + 9, f'{f1_scores[i]:.1f}',
                    ha='center', va='top', fontsize=12, color="#ec0629", fontweight='bold')
        else:
            # Accuracy 标签放在点上方
            ax0.text(i, accuracies[i] + 8, f'{accuracies[i]:.1f}',
                    ha='center', va='bottom', fontsize=12, color='#377eb8', fontweight='bold')
            # F1 标签放在点下方
            ax0.text(i, f1_scores[i] - 8, f'{f1_scores[i]:.1f}',
                    ha='center', va='top', fontsize=12, color='#ec0629', fontweight='bold')

    ax0.set_xlabel('')
    # ax0.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax0.legend(loc='lower right', fontsize=14, framealpha=0.9)
    ax0.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax0.set_ylim([0, max(max(accuracies), max(f1_scores)) * 1.15])
    ax0.set_xlim(-0.5, len(accuracies) - 0.5)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(True)
    ax0.tick_params(axis='y', labelsize=14)

    # 下方：UpSet 风格点阵图
    ax2 = fig.add_subplot(gs[1], sharex=ax0)

    num_models = len(all_models)
    num_combinations = len(combination_data)

    # 设置 y 轴刻度为模型名称（支持换行）
    # 对 PatchShuffle 换行处理
    yticklabels = []
    for model in all_models:
        if model == "PatchShuffle":
            yticklabels.append("Patch\nShuffle")
        else:
            yticklabels.append(model)

    ax2.set_yticks(range(num_models))
    ax2.set_yticklabels(yticklabels, fontsize=14, verticalalignment='center')

    # 绘制点阵图
    for i in range(num_combinations):
        for j in range(num_models):
            is_correct = combination_data[i][j]
            y = j  # 模型行
            x = i  # 组合列

            if is_correct:
                ax2.plot(x, y, 'o', color="#8f3aa4", markersize=10,
                         markerfacecolor='#8f3aa4', markeredgecolor='white', markeredgewidth=1, zorder=2)
            else:
                ax2.plot(x, y, 'o', color="#9B8989", markersize=8,
                         markerfacecolor='white', markeredgecolor="#9B8989", markeredgewidth=1, alpha=0.6, zorder=2)

    num_models = len(all_models)
    num_combinations = len(combination_data)

    for j in range(num_models):
        if j % 2 == 0:  # 偶数索引的行添加背景
            ax2.axhspan(j - 0.2, j + 0.2, color="#e1f3fa", alpha=0.6, zorder=0)
            
    for i, combo in enumerate(combination_data):
        correct_indices = [j for j, is_correct in enumerate(combo) if is_correct]

        if len(correct_indices) > 1:
            correct_indices_sorted = sorted(correct_indices)
            x_vals = [i] * len(correct_indices_sorted)
            y_vals = correct_indices_sorted
            ax2.plot(x_vals, y_vals, '-', color="#8f3aa4", 
                    linewidth=3, alpha=0.8, zorder=4, solid_capstyle='round')
        elif len(correct_indices) == 1:
            ax2.plot([i, i], [correct_indices[0], correct_indices[0]], 
                    '-', color='#8f3aa4', linewidth=2, alpha=0.6, zorder=4)

    # 设置 x 轴
    ax2.set_xticks([])
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_title('')

    # 移除网格和边框
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # 反转 y 轴，使第一个模型在顶部
    ax2.invert_yaxis()

    # 设置 x 轴范围
    ax2.set_xlim(-0.5, num_combinations - 0.5)

    plt.tight_layout()

    # 保存图表
    if save_dir:
        save_path = Path(save_dir) / "prediction_chart.svg"
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"✅ 图表已保存: {save_path}")
    else:
        save_path = Path("prediction_chart.svg")
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"✅ 图表已保存: {save_path}")

    plt.show()
    plt.close()
    return save_path

def main():
    # 设置目录路径
    directory = "/data2/yuyangxin/Agent/outputs/agent_results_confuse_qwen3[qwen3_32b][open_calibration][open_clustering][open_semantic][open_expert]/final_output"
    save_dir = "./output/figures/LALM4Forensic"  # 图表保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)    
    print("🔍 开始分析 JSON 文件...")
    print(f"📂 目录: {directory}\n")
    
    # 加载所有 JSON 文件
    data_list = load_json_files(directory)
    
    if not data_list:
        print("❌ 未找到任何 JSON 文件")
        return
    
    # 分析预测情况
    all_models, model_correct, model_total, model_confusion, combination_stats = analyze_predictions(data_list)
    
    # 计算总样本数
    total_samples = len(data_list)
    
    # 打印结果（按占比排序）
    print_results(all_models, model_correct, model_total, model_confusion, combination_stats, total_samples)
    
    # 绘制图表
    print("\n" + "=" * 80)
    print("📈 开始生成可视化图表...")
    print("=" * 80)
    
    # 绘制UpSet风格的折线图
    plot_upset_style_chart(all_models, combination_stats, total_samples, top_n=20, save_dir=save_dir)
    
    print("\n✅ 所有分析完成！")

if __name__ == "__main__":
    main()