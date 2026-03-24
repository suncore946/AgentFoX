"""
CSV数据分析工具 - 优化版本
支持多指标分析、数据可视化和统计分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


# 设置字体支持和样式
def setup_fonts():
    """设置matplotlib字体配置"""
    # 中文字体设置（按优先级排列）
    chinese_fonts = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial Unicode MS", "WenQuanYi Micro Hei"]

    # 英文字体设置
    english_fonts = ["Times New Roman", "Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]

    # 尝试设置中文字体
    for font in chinese_fonts:
        try:
            plt.rcParams["font.sans-serif"] = [font] + english_fonts
            break
        except:
            continue

    # # 设置字体属性
    # plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    # plt.rcParams["font.size"] = 10  # 默认字体大小
    # plt.rcParams["axes.titlesize"] = 14  # 标题字体大小
    # plt.rcParams["axes.labelsize"] = 12  # 坐标轴标签字体大小
    # plt.rcParams["xtick.labelsize"] = 10  # x轴刻度标签字体大小
    # plt.rcParams["ytick.labelsize"] = 10  # y轴刻度标签字体大小
    # plt.rcParams["legend.fontsize"] = 10  # 图例字体大小
    # plt.rcParams["figure.titlesize"] = 16  # 图形标题字体大小

    # # 设置图形样式
    # plt.style.use("seaborn-v0_8")

    print("✅ 字体配置完成")


# 初始化字体设置
setup_fonts()
warnings.filterwarnings("ignore")


class DataAnalyzer:
    """数据分析器类，封装所有分析功能"""

    def __init__(self, target_columns: List[str] = None):
        """初始化分析器"""
        self.target_columns = target_columns or ["overall_accuracy", "overall_f1"]
        self.required_columns = ["method", "dataset", "sub_name"] + self.target_columns
        self.metric_labels = {"overall_accuracy": "Overall Accuracy", "overall_f1": "Overall F1 Score", "overall_auc": "Overall AUC"}
        self.data = None
        self.result_tables = {}

    def read_csv_files(self, directory: str) -> Optional[pd.DataFrame]:
        """读取目标目录下的所有CSV文件"""
        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"目录 {directory} 不存在")
            return None

        csv_files = list(directory_path.glob("*.csv"))
        if not csv_files:
            print(f"目录 {directory} 中未找到CSV文件")
            return None

        dataframes = []

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)

                # 检查必要列
                missing_columns = [col for col in self.required_columns if col not in df.columns]
                if missing_columns:
                    print(f"文件 {file_path.name} 缺少列: {missing_columns}，跳过")
                    continue

                # 数据预处理
                df_filtered = self._preprocess_dataframe(df)

                if not df_filtered.empty:
                    dataframes.append(df_filtered)
                    print(f"✓ 成功读取: {file_path.name} ({len(df_filtered)} 行)")
                else:
                    print(f"✗ 无有效数据: {file_path.name}")

            except Exception as e:
                print(f"✗ 读取失败 {file_path.name}: {e}")

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"\n总计读取有效数据: {len(combined_df)} 行")
            self.data = combined_df
            return combined_df
        else:
            print("未找到任何有效数据")
            return None

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理单个数据框"""
        # 选择必要列
        df_filtered = df[self.required_columns].copy()

        # 清理数据
        df_filtered = df_filtered.dropna()

        # 数值列转换
        for col in self.target_columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

        # 再次清理NaN
        df_filtered = df_filtered.dropna()

        # 数据验证
        for col in self.target_columns:
            # 确保指标值在合理范围内
            df_filtered = df_filtered[(df_filtered[col] >= 0) & (df_filtered[col] <= 1)]

        return df_filtered

    def process_data(self, use_weighted_average: bool = True) -> Dict[str, pd.DataFrame]:
        """处理数据并创建透视表"""
        if self.data is None:
            print("请先读取数据")
            return {}

        print("开始处理数据...")
        print(f"数据集数量: {self.data['dataset'].nunique()}")
        print(f"方法数量: {self.data['method'].nunique()}")
        print(f"子数据集数量: {self.data['sub_name'].nunique()}")

        # 如果原始数据有total_samples列，使用它作为权重
        if "total_samples" in self.data.columns and use_weighted_average:
            self.data["weight"] = pd.to_numeric(self.data["total_samples"], errors="coerce").fillna(1)
        else:
            self.data["weight"] = 1

        result_tables = {}

        for target_col in self.target_columns:
            # 按dataset和method分组，计算加权平均
            grouped_data = []

            for (dataset, method), group in self.data.groupby(["dataset", "method"]):
                weighted_value = np.average(group[target_col], weights=group["weight"])
                grouped_data.append({"dataset": dataset, "method": method, f"weighted_{target_col}": weighted_value})

            grouped_df = pd.DataFrame(grouped_data)

            # 创建透视表
            pivot_table = grouped_df.pivot(index="dataset", columns="method", values=f"weighted_{target_col}")

            result_tables[target_col] = pivot_table
            print(f"{target_col} 透视表形状: {pivot_table.shape}")

        self.result_tables = result_tables
        return result_tables

    def create_heatmap(self, metric_name: str, save_path: Optional[str] = None, figsize: Optional[Tuple[int, int]] = None) -> None:
        """创建优化的热力图"""
        if metric_name not in self.result_tables:
            print(f"指标 {metric_name} 不存在")
            return

        pivot_table = self.result_tables[metric_name]

        if figsize is None:
            figsize = (max(12, len(pivot_table.columns) * 1.2), max(8, len(pivot_table) * 0.6))

        fig, ax = plt.subplots(figsize=figsize)

        # 创建热力图
        mask = pivot_table.isnull()
        heatmap = sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            center=pivot_table.mean().mean(),  # 以均值为中心
            cbar_kws={"label": self.metric_labels.get(metric_name, metric_name)},
            cbar=False,  # 不显示右侧标签
            linewidths=0.5,
            linecolor="white",
            mask=mask,
            square=False,
            ax=ax,
        )

        # 美化
        ax.set_title(f"{self.metric_labels.get(metric_name, metric_name)}热力图", fontsize=16, pad=20, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # 旋转标签
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # # 添加最佳值标注
        # max_val = pivot_table.max().max()
        # max_pos = np.where(pivot_table.values == max_val)
        # if len(max_pos[0]) > 0:
        #     ax.add_patch(plt.Rectangle((max_pos[1][0], max_pos[0][0]), 1, 1, fill=False, edgecolor="red", lw=3))

        # plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"热力图已保存: {save_path}")


    def create_comprehensive_analysis(self, save_path: Optional[str] = None) -> None:
        """创建综合分析图表"""
        if len(self.result_tables) < 2:
            print("需要至少2个指标进行综合分析")
            return

        metrics = list(self.result_tables.keys())
        metric1, metric2 = metrics[0], metrics[1]

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(2, 1, hspace=0.3, wspace=0.3)

        # # 1. 相关性散点图
        # ax1 = fig.add_subplot(gs[0, 0])
        # self._plot_correlation_scatter(ax1, metric1, metric2)

        # # 2. 方法性能雷达图
        # ax2 = fig.add_subplot(gs[0, 1], projection="polar")
        # self._plot_radar_chart(ax2, [metric1, metric2])

        # # 3. 性能分布箱线图
        # ax3 = fig.add_subplot(gs[0, 2])
        # self._plot_performance_boxplot(ax3, [metric1, metric2])

        # 4. 排名对比图
        ax4 = fig.add_subplot(gs[0, 0])
        self._plot_ranking_comparison(ax4, [metric1, metric2])

        # # 5. 数据集性能热力图
        # ax5 = fig.add_subplot(gs[0, 1])
        # self._plot_dataset_performance_heatmap(ax5, metric1)

        # # 6. 方法稳定性分析
        # ax6 = fig.add_subplot(gs[1, 2:])
        # self._plot_method_stability(ax6, [metric1, metric2])

        # 7. 综合排名表
        ax7 = fig.add_subplot(gs[1, :])
        self._plot_comprehensive_ranking_table(ax7, [metric1, metric2])

        plt.suptitle("数据分析综合报告", fontsize=20, weight="bold", y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"综合分析图已保存: {save_path}")

    def _plot_correlation_scatter(self, ax, metric1: str, metric2: str) -> None:
        """绘制相关性散点图"""
        methods = self.result_tables[metric1].columns
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        for i, method in enumerate(methods):
            vals1 = self.result_tables[metric1][method].dropna()
            vals2 = self.result_tables[metric2][method].dropna()
            common_datasets = vals1.index.intersection(vals2.index)

            if len(common_datasets) > 0:
                x_vals = vals1[common_datasets]
                y_vals = vals2[common_datasets]
                ax.scatter(x_vals, y_vals, label=method, alpha=0.7, s=60, color=colors[i], edgecolors="black", linewidth=0.5)

        # 添加对角线
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k-", alpha=0.3, zorder=0)

        ax.set_xlabel(self.metric_labels.get(metric1, metric1))
        ax.set_ylabel(self.metric_labels.get(metric2, metric2))
        ax.set_title("指标相关性分析")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_radar_chart(self, ax, metrics: List[str]) -> None:
        """绘制雷达图"""
        methods = self.result_tables[metrics[0]].columns
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for i, method in enumerate(methods):
            values = []
            for metric in metrics:
                values.append(self.result_tables[metric][method].mean())
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, label=method, color=colors[i], alpha=0.8)
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.metric_labels.get(m, m) for m in metrics])
        ax.set_title("方法性能雷达图")
        ax.legend(bbox_to_anchor=(1.3, 1), loc="upper left", fontsize=8)

    def _plot_performance_boxplot(self, ax, metrics: List[str]) -> None:
        """绘制性能分布箱线图"""
        methods = self.result_tables[metrics[0]].columns

        plot_data = []
        labels = []
        colors = []

        for metric in metrics:
            for method in methods:
                data = self.result_tables[metric][method].dropna().values
                if len(data) > 0:
                    plot_data.append(data)
                    labels.append(f"{method}\n({self.metric_labels.get(metric, metric)})")
                    colors.append("lightblue" if metric == metrics[0] else "lightcoral")

        bp = ax.boxplot(plot_data, patch_artist=True, labels=labels)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title("性能分布对比")
        ax.set_ylabel("性能值")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_ranking_comparison(self, ax, metrics: List[str]) -> None:
        """绘制排名对比"""
        methods = self.result_tables[metrics[0]].columns

        ranks = {}
        for metric in metrics:
            ranks[metric] = self.result_tables[metric].mean().rank(ascending=False)

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width / 2, ranks[metrics[0]], width, label=self.metric_labels.get(metrics[0], metrics[0]), alpha=0.8)
        bars2 = ax.bar(x + width / 2, ranks[metrics[1]], width, label=self.metric_labels.get(metrics[1], metrics[1]), alpha=0.8)

        ax.set_xlabel("方法")
        ax.set_ylabel("排名 (越小越好)")
        ax.set_title("方法排名对比")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    def _plot_dataset_performance_heatmap(self, ax, metric: str) -> None:
        """绘制数据集性能热力图"""
        data = self.result_tables[metric]
        sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(f"{self.metric_labels.get(metric, metric)} - 数据集 vs 方法")

    def _plot_method_stability(self, ax, metrics: List[str]) -> None:
        """绘制方法稳定性分析"""
        methods = self.result_tables[metrics[0]].columns

        stability_data = []
        for method in methods:
            stds = []
            for metric in metrics:
                std = self.result_tables[metric][method].std()
                stds.append(std)
            stability_data.append(stds)

        x = np.arange(len(methods))
        width = 0.35

        for i, metric in enumerate(metrics):
            values = [row[i] for row in stability_data]
            ax.bar(x + i * width - width / 2, values, width, label=f"{self.metric_labels.get(metric, metric)} 标准差", alpha=0.8)

        ax.set_xlabel("方法")
        ax.set_ylabel("标准差 (越小越稳定)")
        ax.set_title("方法稳定性分析")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_comprehensive_ranking_table(self, ax, metrics: List[str]) -> None:
        """绘制综合排名表"""
        methods = self.result_tables[metrics[0]].columns

        # 计算综合得分
        scores = {}
        for method in methods:
            score = 0
            for metric in metrics:
                # 归一化分数
                mean_val = self.result_tables[metric][method].mean()
                max_val = self.result_tables[metric].mean().max()
                min_val = self.result_tables[metric].mean().min()
                normalized = (mean_val - min_val) / (max_val - min_val) if max_val > min_val else 0
                score += normalized
            scores[method] = score / len(metrics)

        # 排序
        sorted_methods = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 创建表格数据
        table_data = []
        for i, method in enumerate(sorted_methods):
            row = [i + 1, method]
            for metric in metrics:
                row.append(f"{self.result_tables[metric][method].mean():.3f}")
            row.append(f"{scores[method]:.3f}")
            table_data.append(row)

        columns = ["排名", "方法"] + [self.metric_labels.get(m, m) for m in metrics] + ["综合得分"]

        ax.axis("tight")
        ax.axis("off")
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # 设置表格样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title("综合排名表", fontsize=14, weight="bold", pad=20)

    def print_statistical_summary(self) -> None:
        """打印统计摘要"""
        if not self.result_tables:
            print("请先处理数据")
            return

        print("\n" + "=" * 60)
        print("📊 数据分析统计摘要")
        print("=" * 60)

        for metric_name, pivot_table in self.result_tables.items():
            metric_display = self.metric_labels.get(metric_name, metric_name)

            print(f"\n📈 {metric_display} 统计:")
            print("-" * 40)
            print(f"数据集数量: {len(pivot_table)}")
            print(f"方法数量: {len(pivot_table.columns)}")
            print(f"平均值: {pivot_table.mean().mean():.4f}")
            print(f"标准差: {pivot_table.std().mean():.4f}")

            # 最佳方法
            method_means = pivot_table.mean().sort_values(ascending=False)
            print(f"\n🏆 最佳方法 TOP 3:")
            for i, (method, score) in enumerate(method_means.head(3).items()):
                print(f"  {i+1}. {method}: {score:.4f}")

            # 最佳数据集组合
            max_value = pivot_table.max().max()
            max_pos = np.where(pivot_table.values == max_value)
            if len(max_pos[0]) > 0:
                best_dataset = pivot_table.index[max_pos[0][0]]
                best_method = pivot_table.columns[max_pos[1][0]]
                print(f"\n🎯 最佳组合: {best_dataset} + {best_method} = {max_value:.4f}")

    def export_results(self, output_dir: str = ".") -> None:
        """导出结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for metric_name, pivot_table in self.result_tables.items():
            # 导出CSV
            csv_path = output_path / f"processed_{metric_name}_data.csv"
            pivot_table.to_csv(csv_path)

            # 生成热力图
            img_path = output_path / f"{metric_name}_heatmap.png"
            self.create_heatmap(metric_name, str(img_path))

        # # 生成综合分析图
        # if len(self.result_tables) >= 2:
        #     comprehensive_path = output_path / "comprehensive_analysis.png"
        #     self.create_comprehensive_analysis(str(comprehensive_path))
        # print(f"\n✅ 所有结果已导出到: {output_path}")


def main():
    """主函数"""
    # 配置
    target_directory = "./data"
    target_columns = ["overall_accuracy", "overall_f1"]

    print("🚀 CSV数据分析工具 - 优化版本")
    print("=" * 50)

    # 初始化分析器
    analyzer = DataAnalyzer(target_columns)

    # 读取数据
    print("📂 读取CSV文件...")
    df = analyzer.read_csv_files(target_directory)

    if df is not None:
        print(f"\n✅ 数据读取完成! 总行数: {len(df)}")

        # 数据预览
        print("\n📋 数据预览:")
        print(df.head())
        print(f"\n📊 数据形状: {df.shape}")

        # 处理数据
        print("\n🔄 处理数据...")
        result_tables = analyzer.process_data(use_weighted_average=True)

        if result_tables:
            # 打印统计摘要
            analyzer.print_statistical_summary()

            # 导出所有结果
            print("\n💾 导出结果...")
            analyzer.export_results("./output")

            print("\n🎉 分析完成!")
        else:
            print("❌ 数据处理失败")
    else:
        print("❌ 未能读取到有效数据")


if __name__ == "__main__":
    main()
