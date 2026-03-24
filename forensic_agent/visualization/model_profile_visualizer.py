from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


class ModelPerformanceVisualizer:
    """模型性能可视化类 - 用于展示模型在不同聚类上的表现"""

    def __init__(self, figsize=(16, 10)):
        """初始化可视化类

        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.logger = logger

        # 设置美观的配色方案
        self.color_palette = {
            "primary": ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"],
            "light": ["#5dade2", "#ec7063", "#58d68d", "#f5b041", "#af7ac5", "#48c9b0"],
            "dark": ["#2874a6", "#cb4335", "#239b56", "#d68910", "#7d3c98", "#148f77"],
        }

        # 设置绘图风格
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    def visualize_model_profiles(self, profiles: Dict, save_dir: str = None, columns: List[str] = None):
        """可视化模型画像 - 拆分为两张图

        Args:
            profiles: 模型画像数据，格式为 {"overall": {...}, "cluster": {...}}
            save_path: 保存路径（可选）
        """
        # 准备数据
        cluster_data = []
        overall_data = []

        # 处理overall数据（作为baseline参考）
        for model_name, metrics in profiles["overall"].items():
            overall_data.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "F1-Score": metrics["f1_score"],
                    "Total Samples": metrics["total_samples"],
                    "Positive Samples": metrics["positive_samples"],
                    "Negative Samples": metrics["negative_samples"],
                }
            )

        # 处理各个聚类数据
        for cluster_id, cluster_profiles in profiles["cluster"].items():
            for model_name, metrics in cluster_profiles.items():
                cluster_data.append(
                    {
                        "Model": model_name,
                        "Cluster": metrics["display_name"],
                        "Accuracy": metrics["accuracy"],
                        "F1-Score": metrics["f1_score"],
                        "Total Samples": metrics["total_samples"],
                        "Positive Samples": metrics["positive_samples"],
                        "Negative Samples": metrics["negative_samples"],
                        "Samples Ratio": metrics["total_samples"] / profiles["overall"][model_name]["total_samples"],
                    }
                )

        cluster_df = pd.DataFrame(cluster_data)
        overall_df = pd.DataFrame(overall_data)

        cluster = self._create_cluster_chart(cluster_df, overall_df, str(columns))
        table = self._create_table_chart(cluster_df)
        if save_dir:
            save_dir = Path(save_dir)
            self._save_figure(cluster, save_dir / "model_performance_clusters.png")
            self._save_figure(table, save_dir / "model_performance_table.png")

    def _create_table_chart(self, cluster_df: pd.DataFrame):
        """创建基准性能对比图"""
        fig = plt.figure(figsize=self.figsize)

        # 创建网格布局
        gs = fig.add_gridspec(1, 1)
        ax_baseline = fig.add_subplot(gs[0, 0])

        # 下方: 样本信息
        self._plot_sample_info(cluster_df, ax_baseline)

        # 添加总标题
        fig.suptitle("Overall Performance Baseline Analysis", fontsize=18, fontweight="bold", y=0.98)
        return fig

    def _create_cluster_chart(self, cluster_df: pd.DataFrame, overall_df, cluster_names: str):
        """创建聚类性能对比图"""
        fig = plt.figure(figsize=self.figsize)

        # 创建网格布局 - 添加底部区域用于显示overall性能
        # gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[2, 1], hspace=0.15, wspace=0.2)  # 主图占4，overall信息占1
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.15)  # 主图占4，overall信息占1

        ax1 = fig.add_subplot(gs[0, 0])
        # ax2 = fig.add_subplot(gs[0, 1])
        ax_overall = fig.add_subplot(gs[1, :])  # overall信息横跨整个底部

        # 图1: Accuracy比较
        self._plot_metric(cluster_df, "Accuracy", ax1, f"Model Accuracy")

        # 图2: F1-Score比较
        # self._plot_metric(cluster_df, "F1-Score", ax2, f"Model F1-Score")

        # 添加总标题
        fig.suptitle(f"Model Performance Analysis Across {cluster_names}", fontsize=18, fontweight="bold", y=0.95)

        # 在底部专门区域添加overall性能标签
        self._plot_overall_info(overall_df, ax_overall)

        return fig

    def _plot_overall_info(self, overall_df, ax):
        """在专门的区域绘制overall性能信息"""
        ax.axis("off")

        # 计算布局参数
        n_models = len(overall_df)
        box_width = 0.8 / n_models  # 留出边距
        start_x = 0.1
        y_center = 0.5

        for index, row in overall_df.iterrows():
            model_name = row["Model"]
            acc = row["Accuracy"]
            f1 = row["F1-Score"]

            # 获取对应模型的颜色
            color = self.color_palette["primary"][index % len(self.color_palette["primary"])]

            # 计算当前模型的x位置
            x_pos = start_x + index * box_width

            # 添加背景框
            rect = Rectangle(
                (x_pos - 0.02, 0.2), box_width - 0.01, 0.6, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, transform=ax.transAxes
            )
            ax.add_patch(rect)

            # 添加模型名称
            ax.text(
                x_pos + box_width / 2 - 0.02,
                0.7,
                model_name,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                transform=ax.transAxes,
            )

            # 添加性能指标
            ax.text(
                x_pos + box_width / 2 - 0.02,
                0.3,
                f"Acc: {acc:.4f}\nF1: {f1:.4f}",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.transAxes,
            )

        # 添加标题
        ax.text(
            0.02,
            0.5,
            "Overall\nPerformance:",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            style="italic",
            transform=ax.transAxes,
        )

    def _plot_metric(self, df: pd.DataFrame, metric: str, ax, title: str):
        """绘制单个指标的比较图（横置柱状图）

        Args:
            df: 数据框
            metric: 指标名称 ('Accuracy' 或 'F1-Score')
            ax: matplotlib轴对象
            title: 图标题
        """
        # 获取所有模型和聚类
        models = df["Model"].unique()
        clusters = df["Cluster"].unique()

        # 设置bar的高度和位置
        bar_height = 0.7 / len(models)
        y = np.arange(len(clusters))

        # 为每个模型绘制bars
        for i, model in enumerate(models):
            model_data = df[df["Model"] == model]

            # 准备数据
            values = []
            for cluster in clusters:
                cluster_data = model_data[model_data["Cluster"] == cluster]
                if not cluster_data.empty:
                    values.append(cluster_data[metric].values[0])
                else:
                    values.append(0)

            # 使用美观的颜色
            color = self.color_palette["primary"][i % len(self.color_palette["primary"])]
            edge_color = self.color_palette["dark"][i % len(self.color_palette["dark"])]

            # 绘制横向bars
            bars = ax.barh(
                y + i * bar_height, values, bar_height, label=model, color=color, edgecolor=edge_color, linewidth=1.5, alpha=0.85
            )

            # 在bar上添加数值标签 - 改进版本
            for bar, value in zip(bars, values):
                # 始终显示数值，即使很小
                bar_width = bar.get_width()

                # 判断文本放置位置的阈值
                threshold = 0.1  # 可根据需要调整

                # 值太小，文本放在bar外面
                text_x = bar_width + 0.01
                ha = "left"
                text_color = "black"

                ax.text(
                    text_x,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.4f}",
                    ha=ha,
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=text_color,
                )

        # 设置图形属性
        ax.set_xlabel(metric, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_yticks(y + bar_height * (len(models) - 1) / 2)
        ax.set_yticklabels(clusters, fontsize=10)

        ax.set_xlim(0, 1)

        # 添加图例
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

        # 美化网格
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    def _plot_sample_info(self, df: pd.DataFrame, ax):
        """绘制样本信息

        Args:
            df: 数据框
            ax: matplotlib轴对象
        """
        ax.axis("off")

        # 准备样本统计信息
        clusters = df["Cluster"].unique()

        # 创建表格数据
        table_data = []
        for cluster in clusters:
            cluster_data = df[df["Cluster"] == cluster].iloc[0]
            table_data.append(
                [
                    cluster,
                    f"{cluster_data['Total Samples']:,}",
                    f"{cluster_data['Positive Samples']:,}",
                    f"{cluster_data['Negative Samples']:,}",
                    f"{cluster_data['Samples Ratio']:.1%}",
                ]
            )

        # 创建表格
        table = ax.table(
            cellText=table_data,
            colLabels=["Cluster", "Total", "Positive", "Negative", "Sample Ratio"],
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.15, 0.15, 0.15, 0.15],
        )

        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        # 设置表头样式
        for i in range(5):
            table[(0, i)].set_facecolor("#3498db")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # 设置行颜色交替
        for i in range(1, len(clusters) + 1):
            color = "#ecf0f1" if i % 2 == 0 else "white"
            for j in range(5):
                table[(i, j)].set_facecolor(color)

        # 添加标题
        ax.text(0.5, 0.95, "Sample Distribution Information", transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold")

    def _save_figure(self, fig: plt.Figure, save_path: Path) -> None:
        """Save figure to specified path"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        self.logger.info(f"Model Performance Figure saved to {save_path}")
