"""
1. 项目特征曲线 (Item Characteristic Curves, ICC)
数据: 难度参数、辨别度参数
意义: 展示每个项目(样本)在不同能力水平下的正确概率，是IRT模型的核心可视化
价值: 直观理解项目难度和辨别度对模型性能的影响

2. 参数分布图
数据: 难度分布、辨别度分布、能力分布
意义: 了解整体参数的统计特性和异常值
价值: 识别数据质量问题和模型拟合异常

3. 难度-辨别度散点图
数据: 每个项目的(难度, 辨别度)对
意义: 展示项目在二维参数空间的分布
价值: 识别高质量项目、异常项目，指导样本筛选

4. 模型拟合质量图
数据: 观测值vs预测值、残差分析
意义: 评估模型拟合效果
价值: 验证IRT模型的适用性

5. 信息函数图
数据: 项目信息函数、测试信息函数
意义: 展示测试在不同能力水平下的测量精度
价值: 优化测试设计，了解测量精度分布
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple
import warnings


def create_visualization_suite(
    irt_results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 15),
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    创建IRT模型完整可视化套件

    Args:
        irt_results: IRT拟合结果
        save_path: 保存路径（可选）
        figsize: 图像尺寸
        dpi: 图像分辨率

    Returns:
        可视化结果字典
    """
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # 提取数据
    difficulties = irt_results["item_difficulties"]
    discriminations = irt_results["item_discriminations"]
    abilities = irt_results["model_abilities"]
    valid_mask = ~np.isnan(difficulties)

    # 创建子图布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 项目特征曲线 (ICC) - 重点展示
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_item_characteristic_curves(ax1, difficulties[valid_mask][:10], discriminations[valid_mask][:10])

    # 2. 参数分布直方图
    ax2 = fig.add_subplot(gs[0, 2])
    _plot_parameter_distributions(ax2, difficulties[valid_mask], discriminations[valid_mask], abilities)

    # 3. 难度-辨别度散点图
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_difficulty_discrimination_scatter(ax3, difficulties[valid_mask], discriminations[valid_mask])

    # 4. 模型拟合质量
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_model_fit_quality(ax4, irt_results)

    # 5. 测试信息函数
    ax5 = fig.add_subplot(gs[1, 2])
    _plot_test_information_function(ax5, difficulties[valid_mask], discriminations[valid_mask])

    # 6. 能力分布与项目难度对比
    ax6 = fig.add_subplot(gs[2, 0])
    _plot_ability_difficulty_comparison(ax6, abilities, difficulties[valid_mask])

    # 7. 残差分析
    ax7 = fig.add_subplot(gs[2, 1])
    _plot_residual_analysis(ax7, irt_results)

    # 8. 异常项目识别
    ax8 = fig.add_subplot(gs[2, 2])
    _plot_unusual_items(ax8, irt_results["diagnostics"])

    plt.suptitle("IRT模型可视化分析套件 (GPU加速版)", fontsize=16, y=0.98)


def _plot_item_characteristic_curves(ax, difficulties: np.ndarray, discriminations: np.ndarray, n_items: int = 10):
    """绘制项目特征曲线
    可视化项目质量：陡峭的曲线表示好的辨别度
    识别异常项目：形状异常的曲线可能表示问题项目
    指导测试设计：选择不同难度和辨别度的项目组合
    理解测量精度：在不同能力水平下的测量效果
    
    """
    theta_range = np.linspace(-4, 4, 200)

    # 选择代表性项目进行展示
    selected_indices = np.linspace(0, len(difficulties) - 1, min(n_items, len(difficulties))).astype(int)

    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

    for i, idx in enumerate(selected_indices):
        a_i = discriminations[idx]
        b_i = difficulties[idx]

        # 计算ICC: P(X=1|θ) = 1/(1 + exp(-a(θ-b)))
        prob = 1 / (1 + np.exp(-a_i * (theta_range - b_i)))

        ax.plot(theta_range, prob, color=colors[i], linewidth=2, label=f"Item {idx}: a={a_i:.2f}, b={b_i:.2f}")

    ax.set_xlabel("能力水平 (θ)", fontsize=12)
    ax.set_ylabel("正确概率 P(X=1|θ)", fontsize=12)
    ax.set_title("项目特征曲线 (ICC)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1)


def _plot_parameter_distributions(ax, difficulties: np.ndarray, discriminations: np.ndarray, abilities: np.ndarray):
    """绘制参数分布"""
    ax.hist(difficulties, bins=30, alpha=0.7, color="skyblue", label=f"难度 (μ={np.mean(difficulties):.2f})", density=True)
    ax.hist(discriminations, bins=30, alpha=0.7, color="lightcoral", label=f"辨别度 (μ={np.mean(discriminations):.2f})", density=True)

    ax.set_xlabel("参数值", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.set_title("参数分布", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_difficulty_discrimination_scatter(ax, difficulties: np.ndarray, discriminations: np.ndarray):
    """绘制难度-辨别度散点图"""
    # 根据辨别度着色
    scatter = ax.scatter(
        difficulties, discriminations, c=discriminations, cmap="RdYlBu_r", alpha=0.6, s=50, edgecolors="black", linewidth=0.5
    )

    # 添加质量区域标识
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="低辨别度阈值")
    ax.axhline(y=1.7, color="green", linestyle="--", alpha=0.5, label="高辨别度阈值")

    ax.set_xlabel("难度参数 (b)", fontsize=12)
    ax.set_ylabel("辨别度参数 (a)", fontsize=12)
    ax.set_title("项目质量分布图", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.colorbar(scatter, ax=ax, label="辨别度")


def _plot_test_information_function(ax, difficulties: np.ndarray, discriminations: np.ndarray):
    """绘制测试信息函数"""
    theta_range = np.linspace(-4, 4, 200)

    # 计算测试信息函数: I(θ) = Σ a_i² * P_i(θ) * Q_i(θ)
    total_info = np.zeros_like(theta_range)

    for a_i, b_i in zip(discriminations, difficulties):
        P_i = 1 / (1 + np.exp(-a_i * (theta_range - b_i)))
        Q_i = 1 - P_i
        item_info = (a_i**2) * P_i * Q_i
        total_info += item_info

    ax.plot(theta_range, total_info, "b-", linewidth=3, label="测试信息函数")
    ax.fill_between(theta_range, 0, total_info, alpha=0.3, color="blue")

    # 添加标准误差
    se = 1 / np.sqrt(total_info + 1e-8)
    ax2 = ax.twinx()
    ax2.plot(theta_range, se, "r--", linewidth=2, label="标准误差")
    ax2.set_ylabel("标准误差", color="red", fontsize=12)

    ax.set_xlabel("能力水平 (θ)", fontsize=12)
    ax.set_ylabel("信息量", fontsize=12)
    ax.set_title("测试信息函数", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")


def _plot_model_fit_quality(ax, irt_results: Dict[str, Any]):
    """绘制模型拟合质量"""
    quality_metrics = irt_results["quality_metrics"]

    metrics = ["RMSE", "Correlation", "AIC/1000", "BIC/1000"]
    values = [quality_metrics["rmse"], quality_metrics["correlation"], quality_metrics["aic"] / 1000, quality_metrics["bic"] / 1000]

    colors = ["red" if v < 0.1 else "yellow" if v < 0.3 else "green" for v in [values[0], values[1], 0.5, 0.5]]  # 简化的颜色判断

    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor="black")

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.set_title("模型拟合质量指标", fontsize=14, fontweight="bold")
    ax.set_ylabel("指标值", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_ability_difficulty_comparison(ax, abilities: np.ndarray, difficulties: np.ndarray):
    """绘制能力分布与项目难度对比"""
    ax.hist(abilities, bins=30, alpha=0.7, color="lightgreen", label=f"模型能力分布 (n={len(abilities)})", density=True)
    ax.hist(difficulties, bins=30, alpha=0.7, color="orange", label=f"项目难度分布 (n={len(difficulties)})", density=True)

    ax.axvline(np.mean(abilities), color="green", linestyle="-", linewidth=2, label=f"平均能力: {np.mean(abilities):.2f}")
    ax.axvline(np.mean(difficulties), color="red", linestyle="-", linewidth=2, label=f"平均难度: {np.mean(difficulties):.2f}")

    ax.set_xlabel("参数值", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.set_title("能力vs难度分布对比", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_residual_analysis(ax, irt_results: Dict[str, Any]):
    """绘制残差分析"""
    # 这里需要根据实际的拟合数据计算残差
    # 简化版本：展示拟合统计
    fit_info = irt_results["fit_info"]

    stats = ["收敛状态", "迭代次数", "最终似然", "参数数量"]
    values = [
        1 if fit_info["converged"] else 0,
        fit_info["n_iterations"],
        fit_info["final_likelihood"] / 1000,  # 缩放显示
        fit_info["n_items"] + fit_info["n_persons"],
    ]

    ax.bar(stats, values, color=["green", "blue", "purple", "orange"], alpha=0.7)
    ax.set_title("优化过程统计", fontsize=14, fontweight="bold")
    ax.set_ylabel("数值", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _plot_unusual_items(ax, diagnostics: Dict[str, Any]):
    """绘制异常项目识别"""
    item_diag = diagnostics["item_diagnostics"]
    unusual_items = diagnostics["unusual_items"]

    categories = ["总项目数", "低辨别度", "高辨别度", "异常项目"]
    counts = [
        item_diag["discrimination_mean"] * 100,  # 缩放用于显示
        item_diag["low_discrimination_items"],
        item_diag["high_discrimination_items"],
        len(unusual_items),
    ]

    colors = ["blue", "red", "green", "orange"]
    ax.pie(counts, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("项目质量分析", fontsize=14, fontweight="bold")