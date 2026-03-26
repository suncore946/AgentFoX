import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import scipy.stats as stats

from .base_visualizer import BaseVisualizer, VisualizationConfig


class ChartType(Enum):
    """图表类型枚举"""
    BAR_CHART = "bar_chart"                    # 柱状图
    VIOLIN_PLOT = "violin_plot"                # 小提琴图
    BOX_PLOT = "box_plot"                      # 箱线图
    FOREST_PLOT = "forest_plot"                # 森林图
    SIGNIFICANCE_PLOT = "significance_plot"    # 显著性图
    EFFECT_SIZE_PLOT = "effect_size_plot"      # 效应量图
    CONFIDENCE_INTERVAL = "confidence_interval" # 置信区间图
    POWER_ANALYSIS = "power_analysis"          # 功效分析图
    COMPARISON_MATRIX = "comparison_matrix"    # 比较矩阵图


@dataclass
class ChartConfig:
    """图表配置"""
    chart_type: ChartType = ChartType.BAR_CHART
    show_error_bars: bool = True
    show_significance: bool = True
    show_effect_size: bool = True
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    effect_size_thresholds: Tuple[float, float, float] = (0.2, 0.5, 0.8)  # 小、中、大效应
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'chart_type': self.chart_type.value,
            'show_error_bars': self.show_error_bars,
            'show_significance': self.show_significance,
            'show_effect_size': self.show_effect_size,
            'confidence_level': self.confidence_level,
            'significance_threshold': self.significance_threshold,
            'effect_size_thresholds': self.effect_size_thresholds
        }


class StatisticalChartGenerator(BaseVisualizer):
    """统计图表生成器
    
    生成各种统计分析结果的可视化图表
    """
    
    def __init__(self,
                 output_dir: str = "output",
                 vis_config: Optional[VisualizationConfig] = None,
                 chart_config: Optional[ChartConfig] = None):
        """初始化统计图表生成器
        
        Args:
            output_dir: 输出目录
            vis_config: 可视化配置
            chart_config: 图表配置
        """
        super().__init__(output_dir, vis_config)
        self.chart_config = chart_config or ChartConfig()
        
        # 创建图表子目录
        self.charts_dir = os.path.join(self.output_dir, "statistical_charts")
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def generate_significance_plot(self,
                                 test_results: Dict[str, Dict[str, Any]],
                                 title: str = "统计显著性检验结果",
                                 filename: str = "significance_plot") -> str:
        """生成显著性检验结果图表
        
        Args:
            test_results: 检验结果 {test_name: {p_value, statistic, ...}}
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成统计显著性检验结果图表")
        
        if not test_results:
            self.logger.warning("检验结果为空，跳过图表生成")
            return ""
        
        # 提取数据
        test_names = []
        p_values = []
        statistics = []
        
        for test_name, result in test_results.items():
            if isinstance(result, dict) and 'p_value' in result:
                test_names.append(test_name)
                p_values.append(result['p_value'])
                statistics.append(result.get('statistic', 0))
        
        if not p_values:
            self.logger.warning("没有找到有效的p值，跳过图表生成")
            return ""
        
        # 创建图表
        fig, (ax1, ax2) = self.create_subplots(2, 1, figsize=(12, 10))
        
        # 子图1: p值柱状图
        self._plot_p_values(ax1, test_names, p_values)
        
        # 子图2: 统计量图
        self._plot_statistics(ax2, test_names, statistics)
        
        # 设置整体标题
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "statistical_charts")
        plt.close(fig)
        
        return filepath
    
    def generate_effect_size_plot(self,
                                effect_sizes: Dict[str, Dict[str, float]],
                                title: str = "效应量分析",
                                filename: str = "effect_size_plot") -> str:
        """生成效应量图表
        
        Args:
            effect_sizes: 效应量数据 {comparison: {effect_size, confidence_interval, ...}}
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成效应量分析图表")
        
        if not effect_sizes:
            self.logger.warning("效应量数据为空，跳过图表生成")
            return ""
        
        # 创建图表
        fig, ax = self.create_figure()
        
        # 提取数据
        comparisons = []
        effects = []
        lower_ci = []
        upper_ci = []
        
        for comparison, data in effect_sizes.items():
            if isinstance(data, dict) and 'effect_size' in data:
                comparisons.append(comparison)
                effects.append(data['effect_size'])
                
                # 置信区间
                ci = data.get('confidence_interval', [data['effect_size'], data['effect_size']])
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    lower_ci.append(ci[0])
                    upper_ci.append(ci[1])
                else:
                    lower_ci.append(data['effect_size'])
                    upper_ci.append(data['effect_size'])
        
        # 绘制效应量图
        self._plot_effect_sizes(ax, comparisons, effects, lower_ci, upper_ci, title)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "statistical_charts")
        plt.close(fig)
        
        return filepath
    
    def generate_forest_plot(self,
                           studies_data: Dict[str, Dict[str, float]],
                           title: str = "森林图",
                           filename: str = "forest_plot") -> str:
        """生成森林图
        
        Args:
            studies_data: 研究数据 {study_name: {effect_size, lower_ci, upper_ci, weight}}
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成森林图")
        
        if not studies_data:
            self.logger.warning("研究数据为空，跳过森林图生成")
            return ""
        
        # 创建图表
        fig, ax = self.create_figure(figsize=(12, 8))
        
        # 提取数据
        study_names = []
        effect_sizes = []
        lower_cis = []
        upper_cis = []
        weights = []
        
        for study, data in studies_data.items():
            if isinstance(data, dict) and all(key in data for key in ['effect_size', 'lower_ci', 'upper_ci']):
                study_names.append(study)
                effect_sizes.append(data['effect_size'])
                lower_cis.append(data['lower_ci'])
                upper_cis.append(data['upper_ci'])
                weights.append(data.get('weight', 1.0))
        
        # 绘制森林图
        self._plot_forest_chart(ax, study_names, effect_sizes, lower_cis, upper_cis, weights, title)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "statistical_charts")
        plt.close(fig)
        
        return filepath
    
    def generate_confidence_interval_plot(self,
                                        ci_data: Dict[str, Dict[str, Union[float, List[float]]]],
                                        title: str = "置信区间图",
                                        filename: str = "confidence_interval_plot") -> str:
        """生成置信区间图
        
        Args:
            ci_data: 置信区间数据 {parameter: {estimate, confidence_interval}}
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成置信区间图")
        
        if not ci_data:
            self.logger.warning("置信区间数据为空，跳过图表生成")
            return ""
        
        # 创建图表
        fig, ax = self.create_figure()
        
        # 提取数据
        parameters = []
        estimates = []
        lower_bounds = []
        upper_bounds = []
        
        for param, data in ci_data.items():
            if isinstance(data, dict) and 'estimate' in data:
                parameters.append(param)
                estimates.append(data['estimate'])
                
                ci = data.get('confidence_interval', [data['estimate'], data['estimate']])
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    lower_bounds.append(ci[0])
                    upper_bounds.append(ci[1])
                else:
                    lower_bounds.append(data['estimate'])
                    upper_bounds.append(data['estimate'])
        
        # 绘制置信区间图
        self._plot_confidence_intervals(ax, parameters, estimates, lower_bounds, upper_bounds, title)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "statistical_charts")
        plt.close(fig)
        
        return filepath
    
    def generate_power_analysis_plot(self,
                                   power_data: Dict[str, Any],
                                   title: str = "统计功效分析",
                                   filename: str = "power_analysis_plot") -> str:
        """生成统计功效分析图
        
        Args:
            power_data: 功效分析数据
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成统计功效分析图")
        
        if not power_data:
            self.logger.warning("功效分析数据为空，跳过图表生成")
            return ""
        
        # 创建图表
        fig, axes = self.create_subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # 功效与效应量关系
        if 'effect_sizes' in power_data and 'powers' in power_data:
            self._plot_power_vs_effect_size(axes[0], power_data['effect_sizes'], power_data['powers'])
        
        # 功效与样本量关系
        if 'sample_sizes' in power_data and 'powers_sample' in power_data:
            self._plot_power_vs_sample_size(axes[1], power_data['sample_sizes'], power_data['powers_sample'])
        
        # 功效与显著性水平关系
        if 'alpha_levels' in power_data and 'powers_alpha' in power_data:
            self._plot_power_vs_alpha(axes[2], power_data['alpha_levels'], power_data['powers_alpha'])
        
        # 当前研究的功效摘要
        if 'current_power' in power_data:
            self._plot_power_summary(axes[3], power_data)
        
        # 设置整体标题
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "statistical_charts")
        plt.close(fig)
        
        return filepath
    
    def generate_comparison_matrix(self,
                                 comparison_data: Dict[str, Dict[str, Dict[str, Any]]],
                                 title: str = "多重比较矩阵",
                                 filename: str = "comparison_matrix") -> str:
        """生成多重比较矩阵图
        
        Args:
            comparison_data: 比较数据 {group1: {group2: {p_value, effect_size, ...}}}
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成多重比较矩阵图")
        
        if not comparison_data:
            self.logger.warning("比较数据为空，跳过矩阵图生成")
            return ""
        
        # 创建图表
        fig, (ax1, ax2) = self.create_subplots(1, 2, figsize=(16, 7))
        
        # 创建p值矩阵
        groups = list(comparison_data.keys())
        n_groups = len(groups)
        
        p_matrix = np.ones((n_groups, n_groups))
        effect_matrix = np.zeros((n_groups, n_groups))
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if group1 in comparison_data and group2 in comparison_data[group1]:
                    data = comparison_data[group1][group2]
                    p_matrix[i, j] = data.get('p_value', 1.0)
                    effect_matrix[i, j] = data.get('effect_size', 0.0)
        
        # 绘制p值矩阵
        self._plot_comparison_p_values(ax1, p_matrix, groups, "p值矩阵")
        
        # 绘制效应量矩阵
        self._plot_comparison_effect_sizes(ax2, effect_matrix, groups, "效应量矩阵")
        
        # 设置整体标题
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "statistical_charts")
        plt.close(fig)
        
        return filepath
    
    def _plot_p_values(self, ax: plt.Axes, test_names: List[str], p_values: List[float]):
        """绘制p值柱状图"""
        colors = []
        for p in p_values:
            if p < 0.001:
                colors.append('#E74C3C')  # 红色：高度显著
            elif p < 0.01:
                colors.append('#F39C12')  # 橙色：很显著
            elif p < self.chart_config.significance_threshold:
                colors.append('#F1C40F')  # 黄色：显著
            else:
                colors.append('#BDC3C7')  # 灰色：不显著
        
        bars = ax.bar(test_names, p_values, color=colors, alpha=0.7)
        
        # 添加显著性阈值线
        ax.axhline(y=self.chart_config.significance_threshold, color='red', 
                  linestyle='--', alpha=0.7, label=f'α = {self.chart_config.significance_threshold}')
        
        # 设置y轴为对数尺度
        ax.set_yscale('log')
        ax.set_ylabel('p值 (对数尺度)', fontsize=self.config.label_size)
        ax.set_title('统计显著性检验p值', fontsize=self.config.title_size)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加图例
        ax.legend()
        
        # 在每个柱上添加数值标签
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{p_val:.4f}', ha='center', va='bottom', 
                   fontsize=self.config.legend_size - 1)
    
    def _plot_statistics(self, ax: plt.Axes, test_names: List[str], statistics: List[float]):
        """绘制统计量图"""
        colors = self.get_colors(len(test_names))
        
        bars = ax.bar(test_names, statistics, color=colors, alpha=0.7)
        
        ax.set_ylabel('统计量', fontsize=self.config.label_size)
        ax.set_title('检验统计量', fontsize=self.config.title_size)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 在每个柱上添加数值标签
        for bar, stat in zip(bars, statistics):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                   f'{stat:.3f}', ha='center', va='bottom',
                   fontsize=self.config.legend_size - 1)
    
    def _plot_effect_sizes(self, ax: plt.Axes, comparisons: List[str], effects: List[float],
                          lower_ci: List[float], upper_ci: List[float], title: str):
        """绘制效应量图"""
        y_positions = np.arange(len(comparisons))
        
        # 颜色编码效应量大小
        colors = []
        small, medium, large = self.chart_config.effect_size_thresholds
        
        for effect in effects:
            abs_effect = abs(effect)
            if abs_effect < small:
                colors.append('#BDC3C7')  # 灰色：很小
            elif abs_effect < medium:
                colors.append('#F1C40F')  # 黄色：小
            elif abs_effect < large:
                colors.append('#F39C12')  # 橙色：中等
            else:
                colors.append('#E74C3C')  # 红色：大
        
        # 绘制效应量点和置信区间
        ax.errorbar(effects, y_positions, xerr=[np.array(effects) - np.array(lower_ci),
                                               np.array(upper_ci) - np.array(effects)],
                   fmt='o', capsize=5, capthick=2, markersize=8)
        
        # 为每个点着色
        for i, (x, y, color) in enumerate(zip(effects, y_positions, colors)):
            ax.scatter(x, y, color=color, s=100, zorder=3, edgecolors='black')
        
        # 添加零效应参考线
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加效应量阈值线
        for threshold, label, style in [(small, '小效应', ':'), (medium, '中效应', '--'), (large, '大效应', '-.')]:
            ax.axvline(x=threshold, color='gray', linestyle=style, alpha=0.5, label=f'{label} = {threshold}')
            ax.axvline(x=-threshold, color='gray', linestyle=style, alpha=0.5)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(comparisons)
        ax.set_xlabel('效应量', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size)
        ax.legend()
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_forest_chart(self, ax: plt.Axes, study_names: List[str], effect_sizes: List[float],
                          lower_cis: List[float], upper_cis: List[float], weights: List[float], title: str):
        """绘制森林图"""
        y_positions = np.arange(len(study_names))
        
        # 根据权重调整点的大小
        max_weight = max(weights) if weights else 1
        point_sizes = [100 + 200 * (w / max_weight) for w in weights]
        
        # 绘制置信区间线
        for i, (lower, upper, y_pos) in enumerate(zip(lower_cis, upper_cis, y_positions)):
            ax.plot([lower, upper], [y_pos, y_pos], 'k-', linewidth=2, alpha=0.6)
            # 添加置信区间端点
            ax.plot(lower, y_pos, '|', markersize=8, color='black')
            ax.plot(upper, y_pos, '|', markersize=8, color='black')
        
        # 绘制效应量点
        colors = self.get_colors(len(study_names))
        for i, (effect, y_pos, size, color) in enumerate(zip(effect_sizes, y_positions, point_sizes, colors)):
            ax.scatter(effect, y_pos, s=size, color=color, alpha=0.7, 
                      edgecolors='black', linewidth=1, zorder=3)
        
        # 添加零效应参考线
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='无效应')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(study_names)
        ax.set_xlabel('效应量', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size)
        ax.legend()
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_confidence_intervals(self, ax: plt.Axes, parameters: List[str], 
                                 estimates: List[float], lower_bounds: List[float], 
                                 upper_bounds: List[float], title: str):
        """绘制置信区间图"""
        y_positions = np.arange(len(parameters))
        
        # 绘制置信区间
        ax.errorbar(estimates, y_positions, 
                   xerr=[np.array(estimates) - np.array(lower_bounds),
                         np.array(upper_bounds) - np.array(estimates)],
                   fmt='o', capsize=5, capthick=2, markersize=8,
                   color=self.get_colors(1)[0], alpha=0.7)
        
        # 添加零参考线（如果适用）
        if min(lower_bounds) < 0 < max(upper_bounds):
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='零值参考')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(parameters)
        ax.set_xlabel(f'{int(self.chart_config.confidence_level*100)}% 置信区间', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size)
        
        if min(lower_bounds) < 0 < max(upper_bounds):
            ax.legend()
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_power_vs_effect_size(self, ax: plt.Axes, effect_sizes: List[float], powers: List[float]):
        """绘制功效vs效应量图"""
        ax.plot(effect_sizes, powers, 'o-', color=self.get_colors(1)[0], linewidth=2, markersize=6)
        
        # 添加功效阈值线
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='功效 = 0.8')
        
        ax.set_xlabel('效应量', fontsize=self.config.label_size)
        ax.set_ylabel('统计功效', fontsize=self.config.label_size)
        ax.set_title('功效 vs 效应量', fontsize=self.config.title_size)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_power_vs_sample_size(self, ax: plt.Axes, sample_sizes: List[int], powers: List[float]):
        """绘制功效vs样本量图"""
        ax.plot(sample_sizes, powers, 'o-', color=self.get_colors(2)[1], linewidth=2, markersize=6)
        
        # 添加功效阈值线
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='功效 = 0.8')
        
        ax.set_xlabel('样本量', fontsize=self.config.label_size)
        ax.set_ylabel('统计功效', fontsize=self.config.label_size)
        ax.set_title('功效 vs 样本量', fontsize=self.config.title_size)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_power_vs_alpha(self, ax: plt.Axes, alpha_levels: List[float], powers: List[float]):
        """绘制功效vs显著性水平图"""
        ax.plot(alpha_levels, powers, 'o-', color=self.get_colors(3)[2], linewidth=2, markersize=6)
        
        # 添加常用显著性水平线
        ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        
        ax.set_xlabel('显著性水平 (α)', fontsize=self.config.label_size)
        ax.set_ylabel('统计功效', fontsize=self.config.label_size)
        ax.set_title('功效 vs 显著性水平', fontsize=self.config.title_size)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_power_summary(self, ax: plt.Axes, power_data: Dict[str, Any]):
        """绘制功效摘要"""
        current_power = power_data.get('current_power', 0.5)
        
        # 绘制功效条
        ax.barh(['当前功效'], [current_power], color=self.get_colors(1)[0], alpha=0.7)
        
        # 添加阈值线
        ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='理想功效 = 0.8')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('统计功效', fontsize=self.config.label_size)
        ax.set_title('当前研究功效', fontsize=self.config.title_size)
        ax.legend()
        
        # 添加数值标签
        ax.text(current_power + 0.05, 0, f'{current_power:.3f}', 
               va='center', fontsize=self.config.label_size, fontweight='bold')
    
    def _plot_comparison_p_values(self, ax: plt.Axes, p_matrix: np.ndarray, 
                                groups: List[str], title: str):
        """绘制比较p值矩阵"""
        # 对p值取负对数以便更好地可视化
        log_p_matrix = -np.log10(np.clip(p_matrix, 1e-10, 1))
        
        im = ax.imshow(log_p_matrix, cmap='viridis', aspect='equal')
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(groups)))
        ax.set_yticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_yticklabels(groups)
        
        # 添加数值标签
        for i in range(len(groups)):
            for j in range(len(groups)):
                text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(title, fontsize=self.config.title_size)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('-log10(p值)', fontsize=self.config.label_size)
    
    def _plot_comparison_effect_sizes(self, ax: plt.Axes, effect_matrix: np.ndarray, 
                                    groups: List[str], title: str):
        """绘制比较效应量矩阵"""
        im = ax.imshow(effect_matrix, cmap='RdBu_r', aspect='equal', 
                      vmin=-1, vmax=1, center=0)
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(groups)))
        ax.set_yticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_yticklabels(groups)
        
        # 添加数值标签
        for i in range(len(groups)):
            for j in range(len(groups)):
                color = "white" if abs(effect_matrix[i, j]) > 0.5 else "black"
                text = ax.text(j, i, f'{effect_matrix[i, j]:.3f}',
                             ha="center", va="center", color=color, fontsize=8)
        
        ax.set_title(title, fontsize=self.config.title_size)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('效应量', fontsize=self.config.label_size)
    
    def generate(self, data: Dict[str, Any], **kwargs) -> str:
        """生成统计图表（实现抽象方法）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文件路径
        """
        chart_type = kwargs.get('chart_type', self.chart_config.chart_type)
        
        if chart_type == ChartType.SIGNIFICANCE_PLOT:
            return self.generate_significance_plot(
                data,
                title=kwargs.get('title', '显著性检验结果'),
                filename=kwargs.get('filename', 'significance_plot')
            )
        elif chart_type == ChartType.EFFECT_SIZE_PLOT:
            return self.generate_effect_size_plot(
                data,
                title=kwargs.get('title', '效应量分析'),
                filename=kwargs.get('filename', 'effect_size_plot')
            )
        elif chart_type == ChartType.FOREST_PLOT:
            return self.generate_forest_plot(
                data,
                title=kwargs.get('title', '森林图'),
                filename=kwargs.get('filename', 'forest_plot')
            )
        elif chart_type == ChartType.CONFIDENCE_INTERVAL:
            return self.generate_confidence_interval_plot(
                data,
                title=kwargs.get('title', '置信区间图'),
                filename=kwargs.get('filename', 'confidence_interval_plot')
            )
        elif chart_type == ChartType.POWER_ANALYSIS:
            return self.generate_power_analysis_plot(
                data,
                title=kwargs.get('title', '统计功效分析'),
                filename=kwargs.get('filename', 'power_analysis_plot')
            )
        elif chart_type == ChartType.COMPARISON_MATRIX:
            return self.generate_comparison_matrix(
                data,
                title=kwargs.get('title', '多重比较矩阵'),
                filename=kwargs.get('filename', 'comparison_matrix')
            )
        else:
            self.logger.warning(f"不支持的图表类型: {chart_type}")
            return ""


# 便捷函数
def create_statistical_chart_generator(output_dir: str = "output",
                                     vis_config: Optional[VisualizationConfig] = None,
                                     chart_config: Optional[ChartConfig] = None,
                                     **kwargs) -> StatisticalChartGenerator:
    """创建统计图表生成器的便捷函数"""
    return StatisticalChartGenerator(
        output_dir=output_dir,
        vis_config=vis_config,
        chart_config=chart_config
    )