"""
1. model: ["CLIPC2P", "DeeCLIP", "DRCT", "Patch_Shuffle", "SPAI"]
2. 训练数据集: ["GenImage"]
3. 测试数据集: ["AIGIBench", "Chameleon", "CO-SPYBench", "WildRF", "WIRE", "synthbuster", "Community-Forensics-eval"]

流程:
1. 读取数据集
2. 根据检测结果进行单热编码
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns
import upsetplot

from scipy import stats
from forensic_agent.data_operation.dataset_sampler import DatasetSampler, SamplingMethod
from forensic_agent.data_operation.dataset_loader import load_project_data
from cfg import CONFIG


class DrawInfo:
    def __init__(self, save_dir="./outputs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def draw_upset(self, model_results: pd.DataFrame, save_name="upset_plot.png", figsize=(10, 6)):
        """绘制UpSet图

        参数:
            model_results: DataFrame, 包含布尔值的数据框
            save_path: str, 保存路径
            figsize: tuple, 图表尺寸
        """
        print("正在绘制UpSet图...")
        save_path = self.save_dir / save_name

        # 创建成员关系列表
        memberships = model_results.apply(lambda row: model_results.columns[row.values].tolist(), axis=1).tolist()
        upset_data = upsetplot.from_memberships(memberships, data=model_results.index)

        # 设置图表
        plt.figure(figsize=figsize)
        upsetplot.plot(
            upset_data,
            sort_by="cardinality",
            # show_counts=True,
            show_percentages="{:.2%}",
            facecolor="darkblue",
            with_lines=True,
            include_empty_subsets=True,
            totals_plot_elements=0,
            element_size=40,
            intersection_plot_elements=8,
        )
        # 去掉白边
        fig = plt.gcf()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(save_path.with_suffix(".pdf"), format="pdf", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"UpSet图已保存为 '{save_path}'")

    def draw_heat_map(self, scores_df: pd.DataFrame, save_name="heatmap.svg", metrics=None):
        """
        绘制并排的热力图，支持多个指标
        参数:
            scores_df: DataFrame，包含必要列
            save_path: 输出文件路径
            metrics: 要绘制的指标列表，默认为['f1', 'acc']
        """
        save_path = self.save_dir / save_name
        if metrics is None:
            metrics = ["f1", "acc"]

        required_cols = {"dataset", "model"} | set(metrics)
        if not required_cols.issubset(scores_df.columns):
            raise ValueError(f"scores_df 必须包含列: {required_cols}")

        # 1) 数据聚合
        df = scores_df.groupby(["dataset", "model"], as_index=False).mean()

        # 2) 创建所有指标的透视表
        pivots = {metric: df.pivot(index="dataset", columns="model", values=metric) for metric in metrics}

        # 3) 统一行列（确保所有图表完全对齐）
        all_datasets = sorted(set().union(*[pvt.index for pvt in pivots.values()]))
        all_models = sorted(set().union(*[pvt.columns for pvt in pivots.values()]))

        # 4) 排序数据集（按第一个指标的均值降序），但强制将 Overall 放到最后
        first_metric = metrics[0]
        ds_means = pivots[first_metric].reindex(index=all_datasets).mean(axis=1, skipna=True)
        ds_order = ds_means.sort_values(ascending=False).index.tolist()
        if "Overall" in ds_order:
            ds_order.remove("Overall")
            ds_order.append("Overall")

        # 重新索引并排序所有透视表
        for metric in metrics:
            pivots[metric] = pivots[metric].reindex(index=all_datasets, columns=all_models)
            pivots[metric] = pivots[metric].loc[ds_order, all_models]

        # 5) 颜色范围统一处理
        all_values = np.concatenate([pvt.values[np.isfinite(pvt.values)] for pvt in pivots.values() if np.isfinite(pvt.values).any()])

        if len(all_values) == 0:
            vmin, vmax = 0.0, 1.0
        elif all_values.min() >= 0.0 and all_values.max() <= 1.0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
            if vmin == vmax:
                vmin, vmax = vmin - 1e-6, vmax + 1e-6

        # 6) 计算画布大小
        n_rows, n_cols = len(ds_order), len(all_models)
        cell_h, cell_w = 0.9, 0.9  # 增大单元格高度和宽度以适应更大字号
        fig_h = max(5, n_rows * cell_h + 2)
        fig_w_single = max(5, n_cols * cell_w + 2)
        fig_w = fig_w_single * len(metrics) + 2

        # 8) 绘图
        # 统一字号设置（较大字号）
        FONT_TITLE = 18
        FONT_TICKS = 16
        FONT_ANNOT = 14

        sns.set(style="white")
        fig, axes = plt.subplots(ncols=len(metrics), figsize=(fig_w, fig_h), constrained_layout=True)
        cmap = sns.color_palette("Reds", as_cmap=True)

        # 如果只有一个指标，确保axes是数组
        if len(metrics) == 1:
            axes = [axes]

        # 绘制每个指标的热图
        title_name = {
            "f1": "F1-Score Heatmap",
            "acc": "Accuracy Heatmap",
        }
        for i, (metric, ax) in enumerate(zip(metrics, axes)):
            # 以百分比显示
            data_percent = pivots[metric] * 100

            # 格式化为字符串注释（保留两位小数，个位数前补0）
            annot_data = data_percent.copy().applymap(
                lambda x: (f"0{x:.2f}%" if np.isfinite(x) and x < 10 else (f"{x:.2f}%" if np.isfinite(x) else ""))
            )

            sns.heatmap(
                data_percent,
                ax=ax,
                cmap=cmap,
                vmin=vmin * 100,
                vmax=vmax * 100,
                annot=annot_data,
                fmt="",  # 直接用字符串，不用fmt
                linewidths=0.4,
                linecolor="white",
                cbar=False,
                annot_kws={"fontsize": FONT_ANNOT, "fontweight": "bold"},
            )

            # 设置标题（统一较大字号）
            ax.set_title(
                title_name.get(metric, metric) + " (%)",
                fontsize=FONT_TITLE,
                fontweight="bold",
            )

            # 隐藏坐标轴名称
            ax.set_xlabel("")
            ax.set_ylabel("")

            # 设置x坐标轴旋转与字号
            ax.tick_params(axis="x", rotation=30)
            labels = ax.get_xticklabels()
            for label in labels:
                label.set_horizontalalignment("center")
                label.set_position((label.get_position()[0] + 0.1, label.get_position()[1]))
            plt.setp(ax.get_xticklabels(), fontsize=FONT_TICKS)

            # y轴标签处理：只在第一个子图显示，字号较大
            if i > 0:
                ax.set_yticklabels([])
            else:
                # 对过长的标签进行换行
                new_labels = []
                for label in ax.get_yticklabels():
                    text = label.get_text()
                    if len(text) > 15:
                        parts = text.split("-")
                        if len(parts) >= 2:
                            new_label = "-\n".join(parts)
                        else:
                            new_label = text
                        new_labels.append(new_label)
                    else:
                        new_labels.append(text)
                ax.set_yticklabels(new_labels)
                ax.tick_params(axis="y", rotation=0)
                plt.setp(ax.get_yticklabels(), ha="right", fontsize=FONT_TICKS)

        # 9) 保存
        # 去掉白边
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(save_path.with_suffix(".pdf"), format="pdf", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"热力图已保存为 '{save_path.with_suffix('.pdf')}'")
        return fig

    def draw_stacked_bar_chart(self, scores_df: pd.DataFrame, save_name="stacked_bar_chart.svg"):
        """
        绘制不同数据集下gt_label为1和0的南丁格尔图
        scores_df: DataFrame，包含必要列['dataset_name', 'gt_label']
        gt_label: 0的记为real, 1的记为fake
        结果保存为可以缩放的svg格式
        """
        grouped = scores_df.groupby(["dataset_name", "gt_label"]).size().unstack(fill_value=0)

        # 确保有0和1两列，如果没有则添加
        if 0 not in grouped.columns:
            grouped[0] = 0
        if 1 not in grouped.columns:
            grouped[1] = 0

        # 重命名列
        grouped = grouped.rename(columns={0: "Real", 1: "Fake"})

        # 打印检查
        print("各数据集Real和Fake样本分布:")
        print(grouped)

        print("\n正在绘制南丁格尔图...")
        # 设置图表参数
        fig = plt.figure(figsize=(12, 10), facecolor="#f8f9fa")
        ax = fig.add_subplot(111, polar=True)

        # 获取数据集名称和对应的角度
        datasets = grouped.index.tolist()
        n_datasets = len(datasets)
        angles = np.linspace(0, 2 * np.pi, n_datasets, endpoint=False).tolist()

        # 确保首尾相连以闭合图形
        datasets.append(datasets[0])
        angles.append(angles[0])

        # 设置角度标签位置
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(datasets[:-1], fontsize=11, fontweight="bold")

        # 设置径向网格线样式
        ax.grid(True, linestyle="--", alpha=0.7, color="gray")

        # 美化坐标轴
        ax.spines["polar"].set_visible(False)

        # 设置颜色
        real_color = "#2E86C1"  # 蓝色
        fake_color = "#E74C3C"  # 红色

        # 绘制Real数据
        real_values = grouped["Real"].tolist()
        real_values.append(real_values[0])
        ax.fill(angles, real_values, color=real_color, alpha=0.6, label="Real")
        ax.plot(angles, real_values, color=real_color, linewidth=2)

        # 绘制Fake数据
        fake_values = grouped["Fake"].tolist()
        fake_values.append(fake_values[0])
        ax.fill(angles, fake_values, color=fake_color, alpha=0.6, label="Fake")
        ax.plot(angles, fake_values, color=fake_color, linewidth=2)

        # 添加数据标签
        for i, angle in enumerate(angles[:-1]):
            real_val = real_values[i]
            fake_val = fake_values[i]

            # 只在有数据的地方添加标签
            if real_val > 0:
                ax.annotate(
                    f"{real_val}",
                    xy=(angle, real_val),
                    xytext=(angle, real_val + max(grouped.max()) * 0.05),
                    fontsize=9,
                    color=real_color,
                    fontweight="bold",
                    ha="center",
                )

            if fake_val > 0:
                ax.annotate(
                    f"{fake_val}",
                    xy=(angle, fake_val),
                    xytext=(angle, fake_val + max(grouped.max()) * 0.05),
                    fontsize=9,
                    color=fake_color,
                    fontweight="bold",
                    ha="center",
                )

        # 添加标题和图例
        plt.title("Distribution of Real vs Fake Samples Across Datasets", fontsize=16, fontweight="bold", pad=20)
        plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=12)

        # 去掉白边并保存为PDF
        fig = plt.gcf()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        save_path = self.save_dir / save_name
        fig.savefig(save_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
        print(f"Chart saved to {save_path.with_suffix('.pdf')}")
        plt.close(fig)

    def draw_neg_pos_bar(self, data: pd.DataFrame, save_name="neg_pos_bar.svg"):
        """
        绘制正负条形图，展示每个模型组合的样本数量分布
        正条形图表示gt_label为1(Fake)的数量
        负条形图表示gt_label为0(Real)的数量
        """
        save_path = self.save_dir / save_name

        # 按照模型组合和gt_label分组统计
        group_cols = CONFIG["dataset"]["model_names"] + ["gt_label"]
        if "Patch_Shuffle" in group_cols:
            group_cols.remove("Patch_Shuffle")
            group_cols.append("PatchShuffle")
        # 确保所有分组列都存在
        missing_cols = [col for col in group_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")

        # 按模型组合分组（不包含gt_label）
        model_cols = CONFIG["dataset"]["model_names"]
        if "Patch_Shuffle" in model_cols:
            model_cols.remove("Patch_Shuffle")
            model_cols.append("PatchShuffle")

        # 统计每个模型组合下的Real和Fake数量
        grouped_stats = []
        val_map = {True: "√", False: "×"}  # 布尔值映射为字符串

        for name, group in data.groupby(model_cols):
            # 去重，确保每个image_path只计算一次
            group_dedup = group.drop_duplicates(subset=["image_path"])

            # 统计Real(gt_label=0)和Fake(gt_label=1)的数量
            real_count = len(group_dedup[group_dedup["gt_label"] == 0])
            fake_count = len(group_dedup[group_dedup["gt_label"] == 1])

            # 创建组合标签
            # 只有预测正确的模型才会被记录
            combination_label = " ".join([f"{model}:{val_map[val]}" for model, val in zip(model_cols, name)])

            # 转为0000的二进制字符串, 0表示错误, 1表示正确
            binary_str = "".join(["1" if val else "0" for val in name])
            grouped_stats.append(
                {
                    "combination": combination_label,
                    "real_count": real_count,
                    "fake_count": fake_count,
                    "total_count": real_count + fake_count,
                    "binary_str": binary_str,  # 保存原始的布尔值元组用于排序
                }
            )

        # 转换为DataFrame并按总数排序
        stats_df = pd.DataFrame(grouped_stats)
        stats_df = stats_df.sort_values(["binary_str"], ascending=[True])
        # print(f"模型组合统计（共{len(stats_df)}个组合）:")
        # print(stats_df)

        # ---- 参数：缩短柱子（通过缩放），放大 y 轴标签字号 ----
        SCALE_FACTOR = 0.45  # 小于1会压缩柱子长度；根据需要调整（0.2~0.7）
        BAR_HEIGHT = 0.45
        YTICK_FONT = 20  # 放大 y 轴标签字号
        VALUE_FONT = 18  # 条形上数值的字号

        # 动态计算画布高度，保证放大字号仍不拥挤
        n_items = max(1, len(stats_df))
        fig_h = max(6, n_items * (BAR_HEIGHT * 2.0))  # 经验值，保证行间距
        fig_w = 24
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # 设置颜色
        real_color = "#2E86C1"  # 蓝色 - Real
        fake_color = "#E74C3C"  # 红色 - Fake

        # 计算用于绘图的压缩值（视觉上缩短柱子）
        real_plot = -stats_df["real_count"].astype(float) * SCALE_FACTOR
        fake_plot = stats_df["fake_count"].astype(float) * SCALE_FACTOR

        # 获取y轴位置
        y_pos = np.arange(len(stats_df))

        # 绘制正负条形图
        # 绘制条形（使用压缩后的值）
        bars_real = ax.barh(y_pos, real_plot, color=real_color, alpha=0.8, label="Real", height=BAR_HEIGHT)
        bars_fake = ax.barh(y_pos, fake_plot, color=fake_color, alpha=0.8, label="Fake", height=BAR_HEIGHT)

        # 添加较细边框以保持美观
        for bar in list(bars_real) + list(bars_fake):
            bar.set_edgecolor("black")
            bar.set_linewidth(0.3)

        # 在条形上添加原始绝对数值注释（不受 SCALE_FACTOR 影响）
        TEXT_Y_OFFSET = BAR_HEIGHT * 0.12  # 根据需要调节（例如 0.08 ~ 0.18）
        for i, (r_cnt, f_cnt) in enumerate(zip(stats_df["real_count"], stats_df["fake_count"])):
            # Real（负侧），位置使用压缩后的中心位置
            if r_cnt > 0:
                x_pos = real_plot.iat[i] / 2.0
                ax.text(
                    x_pos, i - TEXT_Y_OFFSET, str(r_cnt), ha="center", va="center", fontsize=VALUE_FONT, color="white", fontweight="bold"
                )
            # Fake（正侧）
            if f_cnt > 0:
                x_pos = fake_plot.iat[i] / 2.0
                ax.text(
                    x_pos, i - TEXT_Y_OFFSET, str(f_cnt), ha="center", va="center", fontsize=VALUE_FONT, color="white", fontweight="bold"
                )

        # 设置y轴标签（组合名称）
        # 简化标签显示，只显示关键信息
        # 直接使用完整的组合标签，不进行简化
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stats_df["combination"], fontsize=YTICK_FONT)

        # 设置x轴
        # ax.set_xlabel("Sample Count", fontsize=12, fontweight="bold")
        # ax.set_ylabel("Model Combinations (Correct Predictions)", fontsize=12, fontweight="bold")

        # 在x=0处添加垂直线
        ax.axvline(x=0, color="black", linewidth=1, alpha=0.8)

        # 设置 x 轴刻度：刻度位置以“压缩后”的单位设置，但刻度标签显示为原始绝对数值
        # 计算最大显示（压缩后）
        max_plot_val = max(real_plot.abs().max(), fake_plot.max(), 1.0)
        # 构造刻度位置数组（包含负到正）
        # ticks_plot = np.linspace(-max_plot_val, max_plot_val, num=11)  # 使用对称刻度，更多刻度保证可读
        # 生成刻度标签（恢复为原始绝对值并取整）
        # ax.set_xticks(ticks_plot)
        # tick_labels = [str(int(round(abs(t) / SCALE_FACTOR))) for t in ticks_plot]
        # ax.set_xticklabels(tick_labels, fontsize=16)

        # 设置 x 轴显示范围（略留边距）
        ax.set_xlim(-max_plot_val * 1.05, max_plot_val * 1.05)
        ax.tick_params(axis="x", labelbottom=False)
        # # 图例
        # ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=11)

        # 网格线
        ax.grid(True, axis="x", alpha=0.25, linestyle="--")

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),  # 放在图上方，超出轴区域
                ncol=len(labels),  # 所有图例项一行显示
                frameon=True,
                framealpha=0.9,
                fontsize=26,
            )
        # 去掉白边并保存
        fig = plt.gcf()
        fig.subplots_adjust(left=0.32, right=0.8, top=0.9, bottom=0.1)
        fig.savefig(save_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
        plt.close(fig)
        print(f"正负条形图已保存为 '{save_path.with_suffix('.pdf')}'")
        return stats_df


def get_data():
    """加载数据并进行基本验证"""
    print("正在从数据库加载数据...")
    # if CONFIG.get("test_dataset", None):
    #     data = pd.read_csv(CONFIG["test_dataset"])
    # else:
    #     data, _ = load_project_data(**CONFIG["dataset"])

    data, _ = load_project_data(**CONFIG["dataset"])
    data["dataset_name"] = data["dataset_name"].str.replace("AIGCDetect-testset", "AIGCDetect")
    data["dataset_name"] = data["dataset_name"].str.replace("Community-Forensics-eval", "Community-Forensics")
    data["dataset_name"] = data["dataset_name"].str.replace("GenImage-Val", "GenImage")
    data["dataset_name"] = data["dataset_name"].str.replace("anime-testset", "anime")
    data["model_name"] = data["model_name"].str.replace("Patch_Shuffle", "PatchShuffle")

    # 加载大模型预测结果
    llm_pred_res = CONFIG.get("llm_pred_res", None)
    if llm_pred_res:
        if llm_pred_res.endswith(".jsonl"):
            llm_data: dict = {}
            with open(llm_pred_res, "r") as f:
                for line in f:
                    item = json.loads(line)
                    llm_data[item["image_path"]] = {"pred_label": item["pred_prob"], "gt_label": item["gt_label"]}
        else:
            with open(llm_pred_res, "r") as f:
                llm_data: dict = json.load(f)
        # 转为DataFrame
        llm_data_df = pd.DataFrame.from_dict(llm_data, orient="index").reset_index()
        llm_data_df = llm_data_df.rename(columns={"index": "image_path", "pred_label": "pred_prob"})
        llm_data_df = llm_data_df[["image_path", "pred_prob"]]

        # 根据 image_path 从data中获取 gt_label
        llm_data_df = llm_data_df.merge(data[["image_path", "gt_label", "dataset_name"]], on="image_path", how="left")

        llm_data_df["model_name"] = "VLLM"

        # 合并数据
        # 断言, 要求 llm_data_df 中的 image_path 和 data 中的 image_path 一一对应
        missing_images = set(llm_data_df["image_path"]) - set(data["image_path"])
        if missing_images:
            raise ValueError(f"LLM 预测结果中的部分 image_path 在主数据集中找不到: {missing_images}")
        data = pd.concat([data, llm_data_df], ignore_index=True)

    # 按照 "model_name", "dataset_name", "image_path" 去重
    data = data.drop_duplicates(subset=["model_name", "dataset_name", "image_path"])

    print(f"数据加载完成, 共 {len(data)} 条记录")

    # 取 "gt_label"和"pred_prob",阈值为0.5进行比较, 相同则对应的pred_result为1, 否则为0
    data["acc_count"] = ((data["pred_prob"] > 0.5) == data["gt_label"]).astype(bool)
    # if "calibration_prob" in data:
    #     data["pred_label"] = ((data["calibration_prob"] > 0.5) == data["gt_label"]).astype(bool)

    # 根据model_name进行分组, 对pred_result进行独热编码:
    # 例如: CLIPC2P pred_result 和 gt_label, 预测正确则为1, 否则为0
    # 最终结果为: [1, 0, 0, 0, 0], 表示CLIPC2P预测正确, 其他模型预测错误
    # 创建模型预测结果的透视表
    model_results = data.pivot_table(index="image_path", columns="model_name", values="acc_count", fill_value=False).astype(bool)
    return data, model_results


def computer_model_metrics(data_result: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    计算各模型在各数据集上的F1和ACC指标，并在最后一行追加加权平均(Overall)

    Args:
        data_result: 包含预测结果的DataFrame,必须包含列['model_name', 'dataset_name', 'gt_label', 'pred_prob']
        threshold: 分类阈值,默认0.5
        include_overall: 是否包含所有数据集的整体指标,默认True

    Returns:
        DataFrame: 含['model', 'dataset', 'f1', 'acc']，如include_overall为True则含Overall行
    """
    required_cols = {"model_name", "dataset_name", "gt_label", "pred_prob"}
    missing = required_cols - set(data_result.columns)
    if missing:
        raise ValueError(f"data_result 缺少必要列: {missing}")

    yvals = data_result["gt_label"].unique()
    if not set(np.unique(yvals)).issubset({0, 1}):
        raise ValueError("gt_label 应为二值 {0,1}。如不是,请先转换。")

    print("正在计算各模型在各数据集上的F1和ACC...")

    def calculate_metrics_group(group):
        y_true = group["gt_label"]
        y_pred = (group["pred_prob"] > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0, average="binary")
        acc = accuracy_score(y_true, y_pred)
        sample_count = len(y_true)
        dataset_name = group["dataset_name"].iloc[0] if "dataset_name" in group.columns else "Unknown"
        if dataset_name != "Overall":
            model_name = group["model_name"].iloc[0]
            print(f"模型: {model_name}, 数据集: {dataset_name}, F1: {f1:.4f}, ACC: {acc:.4f}")
        return pd.Series({"f1": f1, "acc": acc, "sample_count": sample_count})

    scores_df = (
        data_result.groupby(["model_name", "dataset_name"], as_index=False)
        .apply(calculate_metrics_group)
        .reset_index()
        .rename(columns={"model_name": "model", "dataset_name": "dataset"})
    )

    print("\n正在计算所有数据集的整体指标(加权平均)...")
    # 正确的做法：对每个模型在所有数据集上的样本合并后直接计算整体指标（避免对 F1 进行不准确的加权平均）
    overall_rows = []

    for model_name in data_result["model_name"].unique():
        model_all = data_result[data_result["model_name"] == model_name].drop_duplicates(subset=["image_path"])
        if len(model_all) == 0:
            continue
        y_true_all = model_all["gt_label"]
        y_pred_all = (model_all["pred_prob"] > threshold).astype(int)
        total_samples = len(model_all)
        overall_f1 = f1_score(y_true_all, y_pred_all, zero_division=0)  # 修正拼写
        overall_acc = accuracy_score(y_true_all, y_pred_all)
        overall_rows.append(
            {"model": model_name, "dataset": "Overall", "f1": overall_f1, "acc": overall_acc, "sample_count": total_samples}
        )
        print(f"模型: {model_name}, 数据集: Overall, F1: {overall_f1:.4f}, ACC: {overall_acc:.4f}, 总样本量: {total_samples}")
    overall_df = pd.DataFrame(overall_rows)
    scores_df = pd.concat([scores_df, overall_df], ignore_index=True)
    # 将f1和acc保留4位小数, 然后合并成"f1/acc"列
    scores_df["f1"] = scores_df["f1"].round(4)
    scores_df["acc"] = scores_df["acc"].round(4)
    scores_df["f1/acc"] = scores_df["f1"].astype(str) + "/" + scores_df["acc"].astype(str)
    # model_name按照下述顺序排序, 其他的放在最后按字母顺序排序
    model_order = CONFIG["dataset"]["model_names"] + ["VLLM"]
    scores_df = scores_df.reset_index(drop=True)
    scores_df = scores_df[["model", "dataset", "f1", "acc", "f1/acc"]]
    return scores_df


def computer_logits_average_metrics(data_result: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:

    for dataset_name, group in data_result.groupby("dataset_name"):
        gt_labels = []
        pred_labels = []
        for image_path, img_group in group.groupby("image_path"):
            gt_labels.append(img_group["gt_label"].iloc[0])
            # 计算平均预测概率
            pred_labels.append(img_group["pred_prob"].mean() > threshold)
        f1 = f1_score(gt_labels, pred_labels, zero_division=0, average="binary")
        acc = accuracy_score(gt_labels, pred_labels)
        print(f"数据集: {dataset_name}, 平均Logits F1/ACC: {f1:.4f}/{acc:.4f}, 样本量: {len(gt_labels)}")

    # 计算所有数据集的整体指标
    gt_labels_all = []
    pred_labels_all = []
    for image_path, img_group in data_result.groupby("image_path"):
        gt_labels_all.append(img_group["gt_label"].iloc[0])
        pred_labels_all.append(img_group["pred_prob"].mean() > threshold)
    overall_f1 = f1_score(gt_labels_all, pred_labels_all, zero_division=0, average="binary")
    overall_acc = accuracy_score(gt_labels_all, pred_labels_all)
    print(f"\n所有数据集整体平均Logits结果, F1/ACC: {overall_f1:.4f}/{overall_acc:.4f}, 总样本量: {len(gt_labels_all)}")


def computer_best_metrics(data_result: pd.DataFrame, threshold: float = 0.5, num=None) -> pd.DataFrame:
    # 验证必要列
    required_cols = {"image_path", "dataset_name", "gt_label", "pred_prob", "model_name"}
    missing = required_cols - set(data_result.columns)
    if missing:
        raise ValueError(f"data_result 缺少必要列: {missing}")

    # 验证标签值
    if not set(data_result["gt_label"].unique()).issubset({0, 1}):
        raise ValueError("gt_label 应为二值 {0,1}。如不是,请先转换。")

    # 将预测概率转换为二值预测
    if num is None:
        num = math.ceil(len(set(data_result["model_name"].tolist())) / 2)

    print(f"正在计算最优结果(多数投票: 大于{num}个及以上模型预测正确)...")

    data_result = data_result.copy()
    data_result["pred_label"] = (data_result["pred_prob"] > threshold).astype(int)
    data_result["is_correct"] = data_result["pred_label"] == data_result["gt_label"]

    # 将data_result按dataset_name和image_path分组，统计每组下每张图像预测正确的模型数量

    grouped = (
        data_result.groupby(["dataset_name", "image_path"])
        .agg(
            gt_label=("gt_label", "first"),  # ✅ 直接命名
            correct_count=("is_correct", "sum"),  # ✅ 正确的模型数
            total_models=("model_name", "count"),  # ✅ 总模型数
        )
        .reset_index()
    )

    # 重命名列
    grouped.columns = ["dataset_name", "image_path", "gt_label", "correct_models_count", "total_models_count"]

    # 多数投票判断

    grouped["majority_vote_pass"] = grouped["correct_models_count"] >= num

    # 根据多数投票结果确定最终预测
    grouped["best_prediction"] = np.where(grouped["majority_vote_pass"], grouped["gt_label"], 1 - grouped["gt_label"])

    # 计算各数据集的指标
    best_results = []
    for dataset_name, dataset_group in grouped.groupby("dataset_name"):
        gt_labels = dataset_group["gt_label"].values
        best_predictions = dataset_group["best_prediction"].values

        # 计算F1和ACC
        f1 = f1_score(gt_labels, best_predictions, zero_division=0, average="binary")
        acc = accuracy_score(gt_labels, best_predictions)
        sample_count = len(gt_labels)

        best_results.append({"model": "Majority Vote", "dataset": dataset_name, "f1": f1, "acc": acc, "sample_count": sample_count})

        print(f"数据集: {dataset_name}, 多数投票 F1/ACC: {f1:.4f}/{acc:.4f}, 样本量: {sample_count}")

    best_df = pd.DataFrame(best_results)

    # 计算加权平均的整体指标
    total_samples = best_df["sample_count"].sum()
    weighted_f1 = (best_df["f1"] * best_df["sample_count"]).sum() / total_samples
    weighted_acc = (best_df["acc"] * best_df["sample_count"]).sum() / total_samples

    overall_best = pd.DataFrame([{"model": "Majority Vote", "dataset": "Overall", "f1": weighted_f1, "acc": weighted_acc}])

    print(f"\n所有数据集整体多数投票结果 (加权平均):")
    print(f"  F1: {weighted_f1:.4f}")
    print(f"  ACC: {weighted_acc:.4f}")
    print(f"  总样本量: {total_samples}")

    merged_df = pd.concat([best_df.drop(columns=["sample_count"]), overall_best], ignore_index=True)
    return merged_df


def computer_friedman(scores_df: pd.DataFrame):
    """
    scores_df: DataFrame, 包含列 'dataset', 'model', 'f1'
    进行Friedman检验以评估不同模型在多个数据集上的性能差异是否显著
    """
    # 透视为 (数据集 × 模型) 矩阵
    pivot = scores_df.pivot(index="dataset", columns="model", values="f1")

    # 丢弃不完整的行
    before = len(pivot)
    pivot = pivot.dropna(how="any")
    after = len(pivot)
    if after < before:
        raise ValueError(f"有 {before-after} 个数据集缺少部分模型的结果，已被丢弃。请检查数据完整性。")

    # Friedman 检验
    stat, p = stats.friedmanchisquare(*[pivot[m].values for m in pivot.columns])
    """
    stat：这是 Friedman 检验的统计量（chi-square 值），用于衡量不同模型在各数据集上的性能差异是否显著。
    p：这是对应的 p-value，表示这些差异出现的概率。如果 p 值很小（如 < 0.05），说明模型间的性能差异具有统计学显著性。
    """
    print(f"Friedman 检验结果: chi2={stat:.3f}, p-value={p:.3g}")
    return stat, p


def computer_number(data_result: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个数据集的样本数量，分别统计gt_label为0和1的数量
    """
    # 根据image_path去重
    data_result = data_result.drop_duplicates(subset=["image_path"])
    # 统计每个数据集的总样本数
    total_counts = data_result.groupby("dataset_name").size().reset_index(name="total_samples")

    # 统计每个数据集中gt_label为0和1的数量
    label_counts = data_result.groupby(["dataset_name", "gt_label"]).size().unstack(fill_value=0)

    # 重命名列，使其更清晰
    if 0 in label_counts.columns:
        label_counts = label_counts.rename(columns={0: "real_samples"})
    else:
        label_counts["real_samples"] = 0

    if 1 in label_counts.columns:
        label_counts = label_counts.rename(columns={1: "fake_samples"})
    else:
        label_counts["fake_samples"] = 0

    # 重置索引以便合并
    label_counts = label_counts.reset_index()

    # 合并总数和分类数据
    detailed_counts = pd.merge(total_counts, label_counts, on="dataset_name")

    # 计算比例
    detailed_counts["real_ratio"] = detailed_counts["real_samples"] / detailed_counts["total_samples"]
    detailed_counts["fake_ratio"] = detailed_counts["fake_samples"] / detailed_counts["total_samples"]

    # 格式化百分比列为更易读的格式
    detailed_counts_csv = detailed_counts.copy()
    detailed_counts_csv["real_ratio_percent"] = (detailed_counts_csv["real_ratio"] * 100).round(2)
    detailed_counts_csv["fake_ratio_percent"] = (detailed_counts_csv["fake_ratio"] * 100).round(2)

    # 重新排列列顺序
    columns_order = [
        "dataset_name",
        "total_samples",
        "real_samples",
        "fake_samples",
        "real_ratio",
        "fake_ratio",
        "real_ratio_percent",
        "fake_ratio_percent",
    ]
    detailed_counts_csv = detailed_counts_csv[columns_order]
    return detailed_counts


def main(sampling_method: SamplingMethod):
    print(f"开始采样:{sampling_method}")

    data_df, model_results = get_data()
    src_data = data_df.copy()
    save_dir = Path(CONFIG["dataset"]["sampling_save_dir"]) / "_".join(CONFIG["dataset"]["dataset_names"])
    sampler = DatasetSampler(config=CONFIG["dataset"], data=data_df, model_results=model_results, save_dir=save_dir)
    if sampling_method == SamplingMethod.ALL:
        sampling_name = "sample_by_all"
    elif sampling_method == SamplingMethod.DATASET:
        sampling_name = "sample_by_dataset"
        data_df, model_results = sampler.sample_by_dataset(force_reload=False)
    elif sampling_method == SamplingMethod.LABEL:
        sampling_name = "sample_by_label"
        exist_sampled = CONFIG.get("exist_sampled", None)
        exist_sampled_data = None
        if exist_sampled:
            print(f"使用已存在的采样数据: {exist_sampled}")
            data_df = pd.read_csv(exist_sampled)
            # 根据 image_path 从原始数据中获取 model_results
            exist_sampled_data = data_df["image_path"].drop_duplicates().tolist()
        per_sample = CONFIG["dataset"]["per_sample"]
        data_df, model_results = sampler.sample_by_label(per_sample=per_sample, exist_content=exist_sampled_data, force_reload=True)

    save_dir = save_dir / sampling_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 采样数据
    data_df.to_csv(save_dir / "sampled_data.csv", index=False)
    number = computer_number(data_df)
    number.to_csv(save_dir / "dataset_numbers.csv", index=False)

    scores_df = computer_model_metrics(src_data)

    # 将 f1/acc 列拆分为两个单独的列用于透视
    scores_pivot = scores_df.copy()
    # 横坐标为 dataset,纵坐标为 model
    pivot_table = scores_pivot.pivot(index="model", columns="dataset", values="f1/acc")
    # 重置索引使 model 成为一列
    pivot_table = pivot_table.reset_index()
    # 保存为 CSV,横坐标为 dataset,纵坐标为 model
    pivot_table.to_csv(save_dir / "model_metrics.csv", index=False)

    print(f"模型指标已保存至 {save_dir / 'model_metrics.csv'}")

    computer_logits_average_metrics(src_data)
    print("平均Logits指标计算完成")

    best_df = computer_best_metrics(src_data, num=None)
    best_df.to_csv(save_dir / "best_metrics.csv", index=False)

    # 绘图与保存绘图
    draw_info = DrawInfo(save_dir)
    # 将stats_df的gt_label列信息合并进model_results, 按照image_path合并
    model_results_with_labels = model_results.reset_index().merge(
        data_df[["image_path", "gt_label"]].drop_duplicates(subset=["image_path"]),
        on="image_path",
        how="left",
    )
    draw_info.draw_neg_pos_bar(model_results_with_labels)
    draw_info.draw_heat_map(scores_df)
    # draw_info.draw_stacked_bar_chart(data_df)
    draw_info.draw_upset(model_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据分析流程控制")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "dataset", "label"], help="采样方式: all/dataset/label，默认all")
    args = parser.parse_args()
    # 根据命令行参数选择采样方式
    if args.mode == "all":
        sampling_method = SamplingMethod.ALL
    elif args.mode == "dataset":
        sampling_method = SamplingMethod.DATASET
    elif args.mode == "label":
        sampling_method = SamplingMethod.LABEL
    else:
        raise ValueError("未知采样方式")
    print(f"选择的采样方式: {sampling_method}")
    main(sampling_method)
