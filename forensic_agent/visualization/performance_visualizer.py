import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Radar, HeatMap, Scatter, Pie, Boxplot, Tab
from pyecharts.commons.utils import JsCode
from pyecharts import options as opts
from pyecharts.charts import Line, Page
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
import pyecharts.options as opts


class PerformanceVisualizer:
    def __init__(self, config):
        self.config = config
        self.save_dir = self.config.save_path / "calibration_analysis"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _format_brier_score(self, brier_score: float) -> str:
        """智能格式化Brier Score显示，根据值的大小自动调整精度"""
        if brier_score >= 0.01:
            return f"{brier_score:.3f}"
        elif brier_score >= 0.001:
            return f"{brier_score:.4f}"
        elif brier_score >= 0.0001:
            return f"{brier_score:.5f}"
        elif brier_score >= 0.00001:
            return f"{brier_score:.6f}"
        else:
            # 对于极小值使用科学计数法
            return f"{brier_score:.2e}"

    def _safe_execute(self, func, operation_name="绘图操作"):
        """统一的错误处理装饰器"""
        try:
            return func()
        except ImportError:
            logger.warning(f"缺少依赖库，无法执行{operation_name}")
        except Exception as e:
            logger.error(f"{operation_name}失败: {e}")

    def _save_chart(self, chart, save_name: Path | str):
        """统一的图表保存逻辑"""
        # 兼容 Path 和 str，确保最终文件名以 .html 结尾
        if isinstance(save_name, Path):
            save_name = str(save_name)
        # 去除多余的目录前缀，只保留文件名
        save_name = os.path.basename(save_name)
        ext = os.path.splitext(save_name)[1]
        if ext.lower() != ".html":
            save_name = os.path.splitext(save_name)[0] + ".html"
        save_path = self.save_dir / save_name
        chart.render(str(save_path))

    def _get_unified_chart_options(
        self,
        title: str,
        xaxis_name: str = None,
        yaxis_name: str = None,
        xaxis_rotate: int = 0,
        enable_toolbox: bool = True,
        enable_legend: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """获取统一的图表样式配置"""
        options = {
            "title_opts": opts.TitleOpts(
                title=title, pos_left="center", pos_top="top", title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold")
            ),
            "xaxis_opts": opts.AxisOpts(
                name=xaxis_name,
                name_location="middle",
                name_gap=35,
                axislabel_opts=opts.LabelOpts(rotate=xaxis_rotate) if xaxis_rotate else None,
            ),
            "yaxis_opts": opts.AxisOpts(name=yaxis_name, name_location="middle", name_gap=50),
        }

        if enable_legend:
            options["legend_opts"] = opts.LegendOpts(pos_bottom="bottom", pos_left="center")
        if enable_toolbox:
            options["toolbox_opts"] = opts.ToolboxOpts(
                is_show=True, pos_right="top", feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts())
            )

        options.update(kwargs)
        return options

    def plot_radar_chart(
        self,
        metrics: Dict,
        output_path: str = "radar_chart.html",
        title: str = "Model Performance Radar Chart",
    ) -> None:
        """
        绘制模型性能雷达图

        参数:
        - metrics: 已读取的JSON数据字典（calibration_result.json的内容），键为模型名，值为模型数据
        - save_path: 保存路径（可选，默认为 'radar_chart.html'）
        - title: 图表标题（可选，默认为 'Model Performance Radar Chart'）
        """

        def _plot():
            # 从JSON提取数据，创建DataFrame
            data = []
            model_names = list(metrics.keys())
            original_metrics = ["accuracy", "f1_score", "false_positive_rate"]  # 基础指标（可配置）
            radar_metrics = ["ece", "brier_score"]  # 雷达图轴（可配置）
            for model_name in model_names:
                original_res = metrics[model_name].get("original_result", {}).get("basic_metrics", {})
                calibrated_res = metrics[model_name].get("calibrated_result", {}).get("basic_metrics", {})
                row = {"model": model_name}
                for m in radar_metrics:
                    row[m] = calibrated_res.get(m, 0.0)  # 默认0.0
                for m in original_metrics:
                    row[m] = original_res.get(m, 0.0)
                data.append(row)
            print(f"雷达图数据: {data}")

            df = pd.DataFrame(data)
            if df.empty:
                logger.warning("没有可用数据绘制雷达图")
                return

            # 数据归一化
            for col in radar_metrics:
                if col in ["ece", "false_positive_rate", "brier_score"]:  # 低值好的指标，反转
                    df[col] = 1 - ((df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6))
                else:  # 高值好的指标，归一化
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)
            df[radar_metrics] = df[radar_metrics] * 100  # 缩放到[0,100]

            # 保留4位小数
            for col in radar_metrics:
                df[col] = df[col].apply(lambda x: round(float(x), 4))

            # 创建雷达图schema（每个metric一个轴，max=100）
            schema = [opts.RadarIndicatorItem(name=m.capitalize(), max_=100) for m in radar_metrics]

            # 创建雷达图
            radar = Radar()
            radar.add_schema(schema=schema)

            # 为每个模型添加series
            for _, row in df.iterrows():
                model_name = row["model"]
                values = [[row[m] for m in radar_metrics]]  # Radar需要[[v1,v2,...]]
                radar.add(series_name=model_name, data=values, linestyle_opts=opts.LineStyleOpts(width=2))

            # 设置选项
            unified_opts = self._get_unified_chart_options(title=title, enable_legend=True, enable_toolbox=True)
            radar.set_global_opts(**unified_opts)
            radar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

            # 保存
            self._save_chart(radar, output_path)
            logger.info(f"雷达图已保存到: {self.save_dir / output_path}")

        self._safe_execute(_plot, "雷达图绘制")

    def plot_heatmap(
        self,
        model_df: pd.DataFrame,
        row_column: str = "dataset",
        col_column: str = "model",
        target_column: str = "acc",
        save_path: str = None,
        title: str = None,
    ) -> None:
        def _plot():
            # 数据验证
            if model_df.empty or not all(col in model_df.columns for col in [row_column, col_column, target_column]):
                logger.warning(f"数据无效或缺少必要列: {[row_column, col_column, target_column]}")
                return

            # 创建透视表并转换为数值
            pivot_table = model_df.pivot_table(index=row_column, columns=col_column, values=target_column, aggfunc="mean").apply(
                pd.to_numeric, errors="coerce"
            )

            if pivot_table.isnull().all().all():
                logger.warning(f"'{target_column}'列没有有效数值数据")
                return

            # 准备热力图数据
            x_data, y_data = list(pivot_table.columns), list(pivot_table.index)
            heatmap_data = [
                [j, i, round(float(pivot_table.loc[y_data[i], x_data[j]]), 4)]
                for i in range(len(y_data))
                for j in range(len(x_data))
                if not pd.isna(pivot_table.loc[y_data[i], x_data[j]])
            ]

            # 创建热力图
            chart_options = self._get_unified_chart_options(
                title=title or f"{target_column}热力图",
                enable_legend=False,
                visualmap_opts=opts.VisualMapOpts(
                    min_=float(pivot_table.min().min()),
                    max_=float(pivot_table.max().max()),
                    is_calculable=True,
                    orient="vertical",
                    pos_left="right",
                    pos_top="center",
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    name=col_column.replace("_", " ").title(),
                    axislabel_opts=opts.LabelOpts(rotate=45),
                    name_location="middle",
                    name_gap=35,
                ),
                yaxis_opts=opts.AxisOpts(type_="category", name=row_column.replace("_", " ").title(), name_location="middle", name_gap=50),
            )

            heatmap = (
                HeatMap()
                .add_xaxis(x_data)
                .add_yaxis("", y_data, heatmap_data, label_opts=opts.LabelOpts(is_show=True, position="inside"))
                .set_global_opts(**chart_options)
            )

            self._save_chart(heatmap, save_path)

            # 统计信息
            min_val, max_val = pivot_table.min().min(), pivot_table.max().max()
            completeness = (1 - pivot_table.isnull().sum().sum() / pivot_table.size) * 100
            logger.info(
                f"热力图统计 - 大小: {pivot_table.shape}, 完整度: {completeness:.1f}%, {target_column}范围: [{min_val:.4f}, {max_val:.4f}]"
            )

        self._safe_execute(_plot, "热力图生成")

    def plot_box_plot(
        self,
        model_df: pd.DataFrame = None,
        calibration_results: Dict = None,
        metrics: List[str] = None,
        group_column: str = "model_name",
        target_column: str = "pred_prob",
        save_path: str = None,
        figsize: Tuple[int, int] = None,
        title: str = None,
    ) -> None:
        """绘制校准前后性能指标的箱型折线图"""

        def _plot():
            if calibration_results:
                # 校准结果对比图
                original_results = calibration_results.get("test_original_result", {})
                calibrated_results = calibration_results.get("test_calibrated_result", {})

                if not (original_results or calibrated_results):
                    logger.warning("校准结果数据为空")
                    return

                metrics = metrics or ["ece", "accuracy", "f1_score"]
                if isinstance(metrics, str):
                    metrics = [metrics]

                # 构建对比数据
                plot_data = []
                for results, type_name in [(original_results, "校准前"), (calibrated_results, "校准后")]:
                    for model_name, model_data in results.items():
                        basic_metrics = model_data.get("basic_metrics", {})
                        for metric in metrics:
                            if metric in basic_metrics:
                                plot_data.append(
                                    {"model_name": model_name, "metric": metric, "value": basic_metrics[metric], "type": type_name}
                                )

                if not plot_data:
                    logger.warning("没有可用的指标数据")
                    return

                plot_df = pd.DataFrame(plot_data)
                models = list(plot_df["model_name"].unique())
                metric_name_map = {"ece": "期望校准误差 (ECE)", "accuracy": "准确率", "f1_score": "F1分数"}

                # 为每个指标创建对比图
                for metric in metrics:
                    metric_data = plot_df[plot_df["metric"] == metric]
                    if metric_data.empty:
                        continue

                    original_values = [
                        (
                            float(metric_data[(metric_data["model_name"] == model) & (metric_data["type"] == "校准前")]["value"].values[0])
                            if len(metric_data[(metric_data["model_name"] == model) & (metric_data["type"] == "校准前")]["value"].values)
                            > 0
                            else 0
                        )
                        for model in models
                    ]

                    calibrated_values = [
                        (
                            float(metric_data[(metric_data["model_name"] == model) & (metric_data["type"] == "校准后")]["value"].values[0])
                            if len(metric_data[(metric_data["model_name"] == model) & (metric_data["type"] == "校准后")]["value"].values)
                            > 0
                            else 0
                        )
                        for model in models
                    ]

                    chart_options = self._get_unified_chart_options(
                        title=f"{metric_name_map.get(metric, metric.upper())} - 校准前后对比",
                        xaxis_name="模型",
                        yaxis_name=metric_name_map.get(metric, metric),
                        xaxis_rotate=45,
                    )

                    bar_chart = (
                        Bar()
                        .add_xaxis(models)
                        .add_yaxis("校准前", original_values, color="#FF6B6B")
                        .add_yaxis("校准后", calibrated_values, color="#4ECDC4")
                        .set_global_opts(**chart_options)
                    )

                    chart_path = (
                        save_path.replace(".html", f"_{metric}.html")
                        if save_path and save_path.endswith(".html")
                        else save_path.replace(os.path.splitext(save_path)[1], f"_{metric}.html") if save_path else f"boxplot_{metric}.html"
                    )

                    self._save_chart(bar_chart, chart_path)

            else:
                # DataFrame箱型图
                if model_df is None or model_df.empty or group_column not in model_df.columns or target_column not in model_df.columns:
                    logger.warning("DataFrame数据无效或缺少必要列")
                    return

                groups = model_df[group_column].unique()
                box_data = []
                for group in groups:
                    group_data = model_df[model_df[group_column] == group][target_column].dropna()
                    if len(group_data) > 0:
                        q1, q2, q3 = group_data.quantile([0.25, 0.5, 0.75])
                        iqr = q3 - q1
                        lower, upper = max(group_data.min(), q1 - 1.5 * iqr), min(group_data.max(), q3 + 1.5 * iqr)
                        box_data.append([float(lower), float(q1), float(q2), float(q3), float(upper)])

                if not box_data:
                    logger.warning("没有有效数据用于绘制箱型图")
                    return

                chart_options = self._get_unified_chart_options(
                    title=title or f"{target_column}分布箱型图",
                    xaxis_name=group_column.replace("_", " ").title(),
                    yaxis_name=target_column.replace("_", " ").title(),
                    xaxis_rotate=45,
                    enable_legend=False,
                )

                boxplot = Boxplot().add_xaxis(list(groups)).add_yaxis("", box_data).set_global_opts(**chart_options)
                self._save_chart(boxplot, save_path)

        self._safe_execute(_plot, "箱型图生成")

    def plot_probability_distributions(self, models_data: List[Dict], save_name: str = "模型归一化概率分布直方图对比.html") -> None:
        """绘制每个模型校准前后的归一化概率分布直方图到同一个HTML文件，一个页面下一个模型一张图"""

        def _plot():
            # 创建固定的概率分箱：0-1，步长0.1
            # 确保精确的区间边界，避免浮点数精度问题
            bin_edges = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0] 精确11个点，10个区间

            def calc_normalized_histogram(data):
                """计算归一化直方图"""
                counts, _ = np.histogram(data, bins=bin_edges, density=False)
                # 归一化：每个分箱的计数除以总样本数，保留四位小数
                normalized_counts = counts / len(data)
                return [round(count, 4) for count in normalized_counts.tolist()]

            # 创建x轴标签：直接显示区间范围，确保从0.0开始到1.0结束
            x_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]  # [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]

            # 收集所有模型的数据并验证
            valid_models_data = []
            all_counts = []

            for model_data in models_data:
                model_name = model_data["model_name"]
                original_probs = model_data["original_probs"]
                calibrated_probs = model_data["calibrated_probs"]

                if len(original_probs) != len(calibrated_probs):
                    logger.warning(f"模型 {model_name} 的输入数组长度不一致，跳过")
                    continue

                # 计算归一化直方图数据
                original_counts = calc_normalized_histogram(original_probs)
                calibrated_counts = calc_normalized_histogram(calibrated_probs)

                all_counts.extend(original_counts)
                all_counts.extend(calibrated_counts)

                valid_models_data.append(
                    {
                        "model_name": model_name,
                        "original_probs": original_probs,
                        "calibrated_probs": calibrated_probs,
                        "original_counts": original_counts,
                        "calibrated_counts": calibrated_counts,
                    }
                )

            if not valid_models_data:
                logger.warning("没有有效的模型数据")
                return

            # 让坐标轴自动调整以包含所有数据
            # 只记录统计信息用于日志

            # 创建Tab容器来容纳每个模型的图表
            tab = Tab()

            # 为每个模型创建单独的柱状图
            for i, model_data in enumerate(valid_models_data):
                model_name = model_data["model_name"]
                original_counts = model_data["original_counts"]
                calibrated_counts = model_data["calibrated_counts"]
                original_probs = model_data["original_probs"]
                calibrated_probs = model_data["calibrated_probs"]

                bar_chart = Bar()
                bar_chart.add_xaxis(x_labels)

                # 添加原始概率数据（柱状图）
                bar_chart.add_yaxis(
                    "校准前概率分布",
                    original_counts,
                    color="#FF6B6B",
                    gap="20%",
                    label_opts=opts.LabelOpts(
                        is_show=True, position="top", formatter=JsCode("function(params) { return (params.value * 100).toFixed(1) + '%'; }")
                    ),
                    itemstyle_opts=opts.ItemStyleOpts(opacity=0.7),
                    tooltip_opts=opts.TooltipOpts(value_formatter=JsCode("function(value) { return (value * 100).toFixed(1) + '%'; }")),
                )

                # 添加校准后概率数据（柱状图）
                bar_chart.add_yaxis(
                    "校准后概率分布",
                    calibrated_counts,
                    color="#4ECDC4",
                    gap="20%",
                    label_opts=opts.LabelOpts(
                        is_show=True, position="top", formatter=JsCode("function(params) { return (params.value * 100).toFixed(1) + '%'; }")
                    ),
                    itemstyle_opts=opts.ItemStyleOpts(opacity=0.7),
                    tooltip_opts=opts.TooltipOpts(value_formatter=JsCode("function(value) { return (value * 100).toFixed(1) + '%'; }")),
                )

                # 创建折线图显示趋势
                line_chart = Line()
                line_chart.add_xaxis(x_labels)

                # 添加原始概率趋势线
                line_chart.add_yaxis(
                    "校准前趋势",
                    original_counts,
                    color="#FF4444",
                    is_smooth=True,
                    symbol="circle",
                    symbol_size=6,
                    linestyle_opts=opts.LineStyleOpts(width=3),
                    label_opts=opts.LabelOpts(is_show=False),
                    tooltip_opts=opts.TooltipOpts(value_formatter=JsCode("function(value) { return (value * 100).toFixed(1) + '%'; }")),
                )

                # 添加校准后概率趋势线
                line_chart.add_yaxis(
                    "校准后趋势",
                    calibrated_counts,
                    color="#2ECC71",
                    is_smooth=True,
                    symbol="diamond",
                    symbol_size=6,
                    linestyle_opts=opts.LineStyleOpts(width=3),
                    label_opts=opts.LabelOpts(is_show=False),
                    tooltip_opts=opts.TooltipOpts(value_formatter=JsCode("function(value) { return (value * 100).toFixed(1) + '%'; }")),
                )

                # 将折线图覆盖到柱状图上
                bar_chart = bar_chart.overlap(line_chart)

                # 配置组合图表选项
                bar_chart.set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{model_name} - 概率分布对比（柱状图+趋势线）",
                        subtitle=f"样本数: {len(original_probs):,} | "
                        f"原始均值: {np.mean(original_probs):.3f} | "
                        f"校准后均值: {np.mean(calibrated_probs):.3f}",
                        pos_left="center",
                        pos_top="top",
                        title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold"),
                        subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12, color="#666666"),
                    ),
                    toolbox_opts=opts.ToolboxOpts(
                        is_show=True,
                        pos_right="top",
                        feature=opts.ToolBoxFeatureOpts(
                            data_view=opts.ToolBoxFeatureDataViewOpts(is_show=True),
                            magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=True, type_=["line", "bar"]),
                            restore=opts.ToolBoxFeatureRestoreOpts(is_show=True),
                            save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(is_show=True),
                        ),
                    ),
                    legend_opts=opts.LegendOpts(pos_bottom="bottom", pos_left="center", orient="horizontal", selected_mode="multiple"),
                    xaxis_opts=opts.AxisOpts(
                        name="概率区间",
                        name_location="middle",
                        name_gap=35,
                        type_="category",
                        boundary_gap=True,  # 设置为True使起始点和终止点不放置数据
                        axispointer_opts=opts.AxisPointerOpts(type_="shadow"),
                        axislabel_opts=opts.LabelOpts(rotate=45),  # 倾斜显示区间标签
                    ),
                    yaxis_opts=opts.AxisOpts(
                        name="归一化频率",
                        name_location="middle",
                        name_gap=50,
                        type_="value",
                        min_=0,  # 固定最小值为0
                        max_=1,
                        axislabel_opts=opts.LabelOpts(formatter=JsCode("function(value) { return (value * 100).toFixed(0) + '%'; }")),
                    ),
                )

                # 将图表添加到Tab
                tab.add(bar_chart, f"{model_name}")

                # 记录单个模型的统计信息
                # 确保转换为numpy数组进行计算
                original_probs_array = np.array(original_probs)
                calibrated_probs_array = np.array(calibrated_probs)

                logger.info(
                    f"模型 {model_name} 归一化概率分布统计:\n"
                    f"    样本总数: {len(original_probs):,}\n"
                    f"    原始概率均值: {np.mean(original_probs_array):.3f} (标准差: {np.std(original_probs_array):.3f})\n"
                    f"    校准后概率均值: {np.mean(calibrated_probs_array):.3f} (标准差: {np.std(calibrated_probs_array):.3f})\n"
                    f"    平均变化量: {np.mean(calibrated_probs_array - original_probs_array):.3f}\n"
                    f"    该模型最大频率: {max(max(original_counts), max(calibrated_counts)):.4f}\n"
                )

            # 保存组合图表
            self._save_chart(tab, save_name)

        self._safe_execute(_plot, "各模型概率分布组合图表（柱状图+折线图）生成")

    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob_original: np.ndarray,
        y_prob_calibrated: np.ndarray = None,
        n_bins: int = 100,
        model_name: str = None,
        save_path: str = "reliability_diagram.html",
    ) -> None:
        """绘制可靠性图（校准图）

        所以每个点 (x, y) 表示“模型在预测为 x 概率的样本中，实际正例的比例为 y”。这正是校准曲线的核心含义。

        Args:
            y_true: 真实标签 (0 或 1)
            y_prob_original: 原始预测概率
            y_prob_calibrated: 校准后预测概率（可选）
            n_bins: 分箱数量
            model_name: 模型名称
            save_path: 保存路径
        """
        if model_name:
            save_path = f"{model_name}_{save_path}"

        # 数据验证
        if len(y_true) != len(y_prob_original):
            raise ValueError("y_true 和 y_prob_original 长度不一致")

        if y_prob_calibrated is not None and len(y_true) != len(y_prob_calibrated):
            raise ValueError("y_true 和 y_prob_calibrated 长度不一致")

        # 确保数据类型正确
        y_true = np.asarray(y_true)
        y_prob_original = np.asarray(y_prob_original)
        if y_prob_calibrated is not None:
            y_prob_calibrated = np.asarray(y_prob_calibrated)

        # 计算校准曲线数据
        def compute_calibration_data(y_true, y_prob, n_bins):
            """计算校准曲线的数据点
            fraction_of_positives：每个分箱（bin）中真实为正例（标签为1）的样本比例（即 y_true 的均值），用于校准曲线的 y 轴。
            mean_predicted_value：每个分箱中预测概率的均值（即 y_prob 的均值），用于校准曲线的 x 轴。
            bin_counts：每个分箱中样本的数量（即该 bin 内有多少个样本）。
            bin_lowers：每个分箱的左边界（包含）。
            bin_uppers：每个分箱的右边界（不包含，最后一个bin右边界为1包含）。
            """
            # 使用 sklearn 的 calibration_curve 函数
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

            # 计算每个分箱的样本数量，修正边界包含逻辑
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            bin_counts = []
            for i in range(n_bins):
                if i == 0:
                    in_bin = (y_prob >= bin_lowers[i]) & (y_prob <= bin_uppers[i])
                else:
                    in_bin = (y_prob > bin_lowers[i]) & (y_prob <= bin_uppers[i])
                # 最后一个bin包含右端点1.0
                if i == n_bins - 1:
                    in_bin = (y_prob > bin_lowers[i]) & (y_prob <= bin_uppers[i] + 1e-8)
                bin_counts.append(np.sum(in_bin))

            return fraction_of_positives, mean_predicted_value, bin_counts, bin_lowers, bin_uppers

        # 计算原始模型的校准数据
        # fop_orig：fraction_of_positives，分箱后每个bin中真实为正例（y_true=1）的样本比例（即y轴坐标），用于校准曲线的y值。
        # mpv_orig：mean_predicted_value，分箱后每个bin中预测概率（y_prob_original）的均值（即x轴坐标），用于校准曲线的x值。
        # counts_orig：每个bin中包含的样本数量（即该分箱内有多少个样本）。
        # bin_lowers：每个分箱的左边界（包含），如[0.0, 0.01, ...]。
        # bin_uppers：每个分箱的右边界（不包含，最后一个bin右边界为1包含），如[0.01, 0.02, ..., 1.0]。
        fop_orig, mpv_orig, counts_orig, bin_lowers, bin_uppers = compute_calibration_data(y_true, y_prob_original, n_bins)

        # 计算校准后模型的数据（如果提供）
        calibrated_data = None
        if y_prob_calibrated is not None:
            fop_cal, mpv_cal, counts_cal, _, _ = compute_calibration_data(y_true, y_prob_calibrated, n_bins)
            calibrated_data = (fop_cal, mpv_cal, counts_cal)

        # 创建折线图（仅线，无点）
        line = Line()

        # 理想校准线（对角线）
        perfect_line_x = [0, 1]
        perfect_line_y = [0, 1]
        line.add_xaxis([str(x) for x in perfect_line_x])
        line.add_yaxis(
            "理想校准线",
            perfect_line_y,
            color="#999999",
            linestyle_opts=opts.LineStyleOpts(width=2, type_="dashed"),
            symbol="none",
            label_opts=opts.LabelOpts(is_show=False),
        )

        # 原始模型校准曲线（仅线）
        original_points = [(round(float(x), 4), round(float(y), 4)) for x, y in zip(mpv_orig.tolist(), fop_orig.tolist())]
        line.add_xaxis([round(float(x), 4) for x in mpv_orig])
        line.add_yaxis(
            "原始模型曲线" if model_name is None else f"{model_name} (原始) 曲线",
            [round(float(y), 4) for y in fop_orig],
            color="#FF6B6B",
            linestyle_opts=opts.LineStyleOpts(width=2, type_="solid"),
            symbol="none",
            label_opts=opts.LabelOpts(is_show=False),
        )

        # 校准后模型校准曲线（仅线）
        if calibrated_data is not None:
            fop_cal, mpv_cal, counts_cal = calibrated_data
            line.add_xaxis([round(float(x), 4) for x in mpv_cal])
            line.add_yaxis(
                "校准后模型曲线" if model_name is None else f"{model_name} (校准后) 曲线",
                [round(float(y), 4) for y in fop_cal],
                color="#4ECDC4",
                linestyle_opts=opts.LineStyleOpts(width=2, type_="solid"),
                symbol="none",
                label_opts=opts.LabelOpts(is_show=False),
            )

        # 计算统计信息
        brier_score_orig = brier_score_loss(y_true, y_prob_original)
        f1_orig = f1_score(y_true, (y_prob_original >= 0.5).astype(int))
        stats_text = f"原始 Brier Score: {self._format_brier_score(brier_score_orig)} | 原始 F1: {f1_orig:.3f}"

        if y_prob_calibrated is not None:
            brier_score_cal = brier_score_loss(y_true, y_prob_calibrated)
            f1_cal = f1_score(y_true, (y_prob_calibrated >= 0.5).astype(int))
            stats_text += f" | 校准后 Brier Score: {self._format_brier_score(brier_score_cal)} | 校准后 F1: {f1_cal:.3f}"
            improvement = ((brier_score_orig - brier_score_cal) / brier_score_orig) * 100
            stats_text += f" | Brier改善: {improvement:.1f}%"

        # 设置图表选项
        title = f"可靠性图 - {model_name}" if model_name else "可靠性图"

        chart_options = self._get_unified_chart_options(
            title=title,
            xaxis_name="平均预测概率",
            yaxis_name="正例比例",
            enable_legend=True,
        )

        # 添加副标题显示统计信息
        chart_options["title_opts"] = opts.TitleOpts(
            title=title,
            subtitle=stats_text,
            pos_left="center",
            pos_top="top",
            title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold"),
            subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12, color="#666666"),
        )

        # 设置坐标轴范围
        chart_options["xaxis_opts"] = opts.AxisOpts(
            name="平均预测概率",
            name_location="middle",
            name_gap=35,
            type_="value",
            min_=0,
            max_=1,
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
        )

        chart_options["yaxis_opts"] = opts.AxisOpts(
            name="正例比例",
            name_location="middle",
            name_gap=50,
            type_="value",
            min_=0,
            max_=1,
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
        )

        line.set_global_opts(**chart_options)

        # 保存图表
        self._save_chart(line, save_path)

        # 输出统计信息
        logger.info(f"可靠性图生成完成:")
        logger.info(f"  样本数量: {len(y_true):,}")
        logger.info(f"  分箱数量: {n_bins}")
        logger.info(f"  {stats_text}")

        return line

    def plot_accuracy_and_f1_charts(
        self,
        csv_data,
        output_path,
        target_title="Community-Forensics-eval",
    ):
        """
        生成准确率和F1分数的折线图，保存在同一个页面文件中，带有自定义间距。
        """
        # 提取所有唯一的subset_name和model_name
        subset_names = sorted(list(set([item["subset_name"] for item in csv_data])), key=lambda x: int(str(x).split("*")[0]))
        model_names = sorted(list(set([item["model_name"] for item in csv_data])))

        # 创建准确率折线图
        for item in csv_data:
            pass
        acc_line = Line(init_opts=opts.InitOpts(width="1700px", height="500px"))
        acc_line.add_xaxis(subset_names)
        for model_name in model_names:
            model_data = [item for item in csv_data if item["model_name"] == model_name]
            accuracy_dict = {item["subset_name"]: item["accuracy"] for item in model_data}
            accuracy_values = [round(accuracy_dict.get(subset, 0) * 100, 2) for subset in subset_names]
            acc_line.add_yaxis(model_name, accuracy_values, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))

        acc_line.set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{target_title}上不同子集准确率比较",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=18, font_weight="bold"),
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                name="数据集",
                boundary_gap=False,
                axislabel_opts=opts.LabelOpts(interval=0),
            ),
            yaxis_opts=opts.AxisOpts(name="准确率", min_=0, max_=100, interval=10, axislabel_opts=opts.LabelOpts(formatter="{value}%")),
            # legend_opts=opts.LegendOpts(
            #     pos_right="0%",
            #     pos_top="10%",
            #     orient="vertical",
            #     item_width=25,
            #     item_height=14,
            # ),
            legend_opts=opts.LegendOpts(pos_top="5%", item_width=25, item_height=14),
        )

        # 创建一个空白间隔组件（使用空表格）
        spacer = Table().add([""], [[""]]).set_global_opts(title_opts=ComponentTitleOpts(title="", subtitle=""))

        # 创建F1折线图
        f1_line = Line(init_opts=opts.InitOpts(width="1700px", height="500px"))
        f1_line.add_xaxis(subset_names)
        for model_name in model_names:
            model_data = [item for item in csv_data if item["model_name"] == model_name]
            f1_dict = {item["subset_name"]: item["f1_score"] for item in model_data}
            f1_values = [round(f1_dict.get(subset, 0) * 100, 2) for subset in subset_names]
            f1_line.add_yaxis(model_name, f1_values, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))

        f1_line.set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{target_title}上不同子集F1分数比较",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=18, font_weight="bold"),
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                name="数据集",
                boundary_gap=False,
                axislabel_opts=opts.LabelOpts(interval=0),
            ),
            yaxis_opts=opts.AxisOpts(name="F1分数", min_=0, max_=100, interval=10, axislabel_opts=opts.LabelOpts(formatter="{value}%")),
            # legend_opts=opts.LegendOpts(pos_right="0%", pos_top="10%", orient="vertical", item_width=25, item_height=14),
            legend_opts=opts.LegendOpts(pos_top="5%", item_width=25, item_height=14),
        )

        # 使用Page将图表组合，添加间隔
        page = Page(layout=Page.SimplePageLayout)
        page.add(acc_line)
        # 添加多个空白间隔来增加两图之间的距离
        for _ in range(3):  # 添加3个空白间隔，可以调整数量来控制间距
            page.add(spacer)
        page.add(f1_line)

        # 覆盖保存页面（如果已存在则覆盖）
        import os

        if os.path.exists(output_path):
            os.remove(output_path)
        page.render(output_path)
        print(f"合并图表已保存到: {output_path}")
