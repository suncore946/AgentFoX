import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

from .base_visualizer import BaseVisualizer, VisualizationConfig


class HeatmapType(Enum):
    """热力图类型枚举"""
    PERFORMANCE = "performance"           # 性能指标热力图
    SIGNIFICANCE = "significance"         # 统计显著性热力图
    FEATURE_IMPORTANCE = "feature_importance"  # 特征重要性热力图
    CORRELATION = "correlation"           # 相关性热力图
    DIFFERENCE = "difference"             # 差异热力图


@dataclass
class HeatmapConfig:
    """热力图配置"""
    heatmap_type: HeatmapType = HeatmapType.PERFORMANCE
    colormap: str = "RdYlBu_r"
    show_values: bool = True
    value_format: str = ".3f"
    show_colorbar: bool = True
    square_cells: bool = False
    cluster_rows: bool = False
    cluster_cols: bool = False
    mask_insignificant: bool = True
    significance_threshold: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'heatmap_type': self.heatmap_type.value,
            'colormap': self.colormap,
            'show_values': self.show_values,
            'value_format': self.value_format,
            'show_colorbar': self.show_colorbar,
            'square_cells': self.square_cells,
            'cluster_rows': self.cluster_rows,
            'cluster_cols': self.cluster_cols,
            'mask_insignificant': self.mask_insignificant,
            'significance_threshold': self.significance_threshold
        }


class HeatmapGenerator(BaseVisualizer):
    """热力图生成器
    
    生成各种类型的热力图来可视化模型性能和统计结果
    """
    
    def __init__(self,
                 output_dir: str = "output",
                 vis_config: Optional[VisualizationConfig] = None,
                 heatmap_config: Optional[HeatmapConfig] = None):
        """初始化热力图生成器
        
        Args:
            output_dir: 输出目录
            vis_config: 可视化配置
            heatmap_config: 热力图配置
        """
        super().__init__(output_dir, vis_config)
        self.heatmap_config = heatmap_config or HeatmapConfig()
        
        # 创建热力图子目录
        self.heatmap_dir = os.path.join(self.output_dir, "heatmaps")
        os.makedirs(self.heatmap_dir, exist_ok=True)
    
    def generate_performance_heatmap(self,
                                   performance_data: Dict[str, Dict[str, float]],
                                   title: str = "模型性能热力图",
                                   filename: str = "performance_heatmap",
                                   statistical_results: Optional[Dict[str, Any]] = None) -> str:
        """生成性能指标热力图
        
        Args:
            performance_data: 性能数据 {content_type: {metric: value}}
            title: 图表标题
            filename: 文件名
            statistical_results: 统计结果（用于显著性标注）
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成性能指标热力图")
        
        # 转换数据格式
        df = self._dict_to_dataframe(performance_data)
        
        if df.empty:
            self.logger.warning("性能数据为空，跳过热力图生成")
            return ""
        
        # 创建图表
        fig, ax = self.create_figure()
        
        # 生成热力图
        self._create_heatmap(
            ax, df, title, 
            colormap=self.heatmap_config.colormap,
            show_values=self.heatmap_config.show_values,
            value_format=self.heatmap_config.value_format
        )
        
        # 添加统计显著性标注
        if statistical_results:
            self._add_significance_markers(ax, df, statistical_results)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "heatmaps")
        plt.close(fig)
        
        return filepath
    
    def generate_significance_heatmap(self,
                                    p_values: Dict[str, Dict[str, float]],
                                    title: str = "统计显著性热力图",
                                    filename: str = "significance_heatmap") -> str:
        """生成统计显著性热力图
        
        Args:
            p_values: p值数据 {test: {comparison: p_value}}
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成统计显著性热力图")
        
        # 转换数据格式
        df = self._dict_to_dataframe(p_values)
        
        if df.empty:
            self.logger.warning("p值数据为空，跳过热力图生成")
            return ""
        
        # 创建图表
        fig, ax = self.create_figure()
        
        # 计算负对数p值用于更好的可视化
        log_p_df = -np.log10(df.clip(lower=1e-10))  # 避免log(0)
        
        # 创建显著性热力图
        self._create_heatmap(
            ax, log_p_df, title,
            colormap="viridis",
            show_values=True,
            value_format=".2f",
            cbar_label="-log10(p-value)"
        )
        
        # 添加显著性阈值线
        self._add_significance_threshold_line(ax, log_p_df, self.heatmap_config.significance_threshold)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "heatmaps")
        plt.close(fig)
        
        return filepath
    
    def generate_feature_importance_heatmap(self,
                                          feature_importance: Dict[str, Dict[str, float]],
                                          title: str = "特征重要性热力图",
                                          filename: str = "feature_importance_heatmap") -> str:
        """生成特征重要性热力图
        
        Args:
            feature_importance: 特征重要性数据 {content_type: {feature: importance}}
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成特征重要性热力图")
        
        # 转换数据格式
        df = self._dict_to_dataframe(feature_importance)
        
        if df.empty:
            self.logger.warning("特征重要性数据为空，跳过热力图生成")
            return ""
        
        # 创建图表
        fig, ax = self.create_figure()
        
        # 创建特征重要性热力图
        self._create_heatmap(
            ax, df, title,
            colormap="plasma",
            show_values=True,
            value_format=".3f",
            cbar_label="重要性分数"
        )
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "heatmaps")
        plt.close(fig)
        
        return filepath
    
    def generate_correlation_heatmap(self,
                                   correlation_matrix: Union[pd.DataFrame, Dict[str, Dict[str, float]]],
                                   title: str = "特征相关性热力图",
                                   filename: str = "correlation_heatmap") -> str:
        """生成相关性热力图
        
        Args:
            correlation_matrix: 相关性矩阵
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成相关性热力图")
        
        # 转换数据格式
        if isinstance(correlation_matrix, dict):
            df = self._dict_to_dataframe(correlation_matrix)
        else:
            df = correlation_matrix
        
        if df.empty:
            self.logger.warning("相关性数据为空，跳过热力图生成")
            return ""
        
        # 创建图表
        fig, ax = self.create_figure()
        
        # 创建相关性热力图（使用对称的colormap）
        self._create_heatmap(
            ax, df, title,
            colormap="coolwarm",
            show_values=True,
            value_format=".2f",
            center=0,  # 相关性以0为中心
            vmin=-1,
            vmax=1,
            cbar_label="相关系数"
        )
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "heatmaps")
        plt.close(fig)
        
        return filepath
    
    def generate_difference_heatmap(self,
                                  baseline_data: Dict[str, Dict[str, float]],
                                  comparison_data: Dict[str, Dict[str, float]],
                                  title: str = "性能差异热力图",
                                  filename: str = "difference_heatmap") -> str:
        """生成差异热力图
        
        Args:
            baseline_data: 基线数据
            comparison_data: 比较数据
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成性能差异热力图")
        
        # 转换数据格式
        baseline_df = self._dict_to_dataframe(baseline_data)
        comparison_df = self._dict_to_dataframe(comparison_data)
        
        if baseline_df.empty or comparison_df.empty:
            self.logger.warning("差异数据为空，跳过热力图生成")
            return ""
        
        # 计算差异
        # 确保数据框具有相同的结构
        common_index = baseline_df.index.intersection(comparison_df.index)
        common_columns = baseline_df.columns.intersection(comparison_df.columns)
        
        baseline_aligned = baseline_df.loc[common_index, common_columns]
        comparison_aligned = comparison_df.loc[common_index, common_columns]
        
        difference_df = comparison_aligned - baseline_aligned
        
        # 创建图表
        fig, ax = self.create_figure()
        
        # 创建差异热力图
        self._create_heatmap(
            ax, difference_df, title,
            colormap="RdBu_r",
            show_values=True,
            value_format=".3f",
            center=0,  # 差异以0为中心
            cbar_label="性能差异"
        )
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "heatmaps")
        plt.close(fig)
        
        return filepath
    
    def generate_comprehensive_heatmap(self,
                                     data: Dict[str, Any],
                                     title: str = "综合性能分析热力图",
                                     filename: str = "comprehensive_heatmap") -> str:
        """生成综合热力图（包含多个子图）
        
        Args:
            data: 包含多种数据的字典
            title: 图表标题
            filename: 文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成综合性能分析热力图")
        
        # 确定子图数量和布局
        available_plots = []
        if 'performance' in data:
            available_plots.append(('performance', '性能指标'))
        if 'significance' in data:
            available_plots.append(('significance', '统计显著性'))
        if 'feature_importance' in data:
            available_plots.append(('feature_importance', '特征重要性'))
        if 'correlation' in data:
            available_plots.append(('correlation', '特征相关性'))
        
        if not available_plots:
            self.logger.warning("没有可用的数据进行综合热力图生成")
            return ""
        
        # 计算子图布局
        n_plots = len(available_plots)
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots == 2:
            rows, cols = 1, 2
        elif n_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        # 创建子图
        fig, axes = self.create_subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # 生成各个热力图
        for i, (data_key, subplot_title) in enumerate(available_plots):
            ax = axes[i]
            
            if data_key == 'performance':
                df = self._dict_to_dataframe(data[data_key])
                self._create_heatmap(ax, df, subplot_title, 
                                   colormap="RdYlBu_r", show_values=True)
            
            elif data_key == 'significance':
                df = self._dict_to_dataframe(data[data_key])
                log_p_df = -np.log10(df.clip(lower=1e-10))
                self._create_heatmap(ax, log_p_df, subplot_title,
                                   colormap="viridis", show_values=True)
            
            elif data_key == 'feature_importance':
                df = self._dict_to_dataframe(data[data_key])
                self._create_heatmap(ax, df, subplot_title,
                                   colormap="plasma", show_values=True)
            
            elif data_key == 'correlation':
                if isinstance(data[data_key], dict):
                    df = self._dict_to_dataframe(data[data_key])
                else:
                    df = data[data_key]
                self._create_heatmap(ax, df, subplot_title,
                                   colormap="coolwarm", show_values=True, center=0)
        
        # 隐藏多余的子图
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # 设置整体标题
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold', y=0.98)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "heatmaps")
        plt.close(fig)
        
        return filepath
    
    def _dict_to_dataframe(self, data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """将嵌套字典转换为DataFrame
        
        Args:
            data: 嵌套字典数据
            
        Returns:
            pd.DataFrame: 转换后的数据框
        """
        if not data:
            return pd.DataFrame()
        
        try:
            # 转换为DataFrame，行为外层key，列为内层key
            df = pd.DataFrame.from_dict(data, orient='index')
            return df.fillna(0)  # 填充缺失值
        except Exception as e:
            self.logger.error(f"数据转换失败: {e}")
            return pd.DataFrame()
    
    def _create_heatmap(self,
                       ax: plt.Axes,
                       data: pd.DataFrame,
                       title: str,
                       colormap: str = "RdYlBu_r",
                       show_values: bool = True,
                       value_format: str = ".3f",
                       center: Optional[float] = None,
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       cbar_label: str = "值") -> None:
        """创建热力图
        
        Args:
            ax: 坐标轴对象
            data: 数据框
            title: 标题
            colormap: 颜色映射
            show_values: 是否显示数值
            value_format: 数值格式
            center: 颜色中心值
            vmin: 最小值
            vmax: 最大值
            cbar_label: 颜色条标签
        """
        # 创建热力图
        sns.heatmap(
            data,
            ax=ax,
            cmap=colormap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            annot=show_values,
            fmt=value_format,
            square=self.heatmap_config.square_cells,
            cbar=self.heatmap_config.show_colorbar,
            cbar_kws={'label': cbar_label} if self.heatmap_config.show_colorbar else None,
            annot_kws={'fontsize': self.config.legend_size}
        )
        
        # 设置标题和标签
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold', pad=20)
        ax.set_xlabel("指标", fontsize=self.config.label_size)
        ax.set_ylabel("内容类型", fontsize=self.config.label_size)
        
        # 旋转标签以提高可读性
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    
    def _add_significance_markers(self,
                                ax: plt.Axes,
                                data: pd.DataFrame,
                                statistical_results: Dict[str, Any]) -> None:
        """添加统计显著性标记
        
        Args:
            ax: 坐标轴对象
            data: 数据框
            statistical_results: 统计结果
        """
        # 遍历数据框中的每个单元格
        for i, row_name in enumerate(data.index):
            for j, col_name in enumerate(data.columns):
                # 查找对应的统计结果
                key = f"{row_name}_{col_name}"
                if key in statistical_results:
                    result = statistical_results[key]
                    if isinstance(result, dict) and 'p_value' in result:
                        p_value = result['p_value']
                        
                        # 确定显著性符号
                        if p_value < 0.001:
                            marker = "***"
                        elif p_value < 0.01:
                            marker = "**"
                        elif p_value < 0.05:
                            marker = "*"
                        else:
                            marker = ""
                        
                        # 添加标记
                        if marker:
                            ax.text(j + 0.8, i + 0.2, marker, 
                                   fontsize=8, fontweight='bold', color='red')
    
    def _add_significance_threshold_line(self,
                                       ax: plt.Axes,
                                       data: pd.DataFrame,
                                       threshold: float = 0.05) -> None:
        """添加显著性阈值线
        
        Args:
            ax: 坐标轴对象
            data: 数据框
            threshold: 显著性阈值
        """
        threshold_log = -np.log10(threshold)
        
        # 添加水平参考线
        for i in range(len(data.index)):
            ax.axhline(y=i + 0.5, xmin=0, xmax=1, 
                      color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        # 添加图例说明
        ax.text(0.02, 0.98, f'阈值: p < {threshold}\n(-log10 = {threshold_log:.2f})', 
               transform=ax.transAxes, fontsize=self.config.legend_size,
               verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def generate(self, data: Dict[str, Any], **kwargs) -> str:
        """生成热力图（实现抽象方法）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文件路径
        """
        heatmap_type = kwargs.get('heatmap_type', self.heatmap_config.heatmap_type)
        
        if heatmap_type == HeatmapType.PERFORMANCE:
            return self.generate_performance_heatmap(
                data, 
                title=kwargs.get('title', '性能热力图'),
                filename=kwargs.get('filename', 'performance_heatmap')
            )
        elif heatmap_type == HeatmapType.SIGNIFICANCE:
            return self.generate_significance_heatmap(
                data,
                title=kwargs.get('title', '显著性热力图'),
                filename=kwargs.get('filename', 'significance_heatmap')
            )
        elif heatmap_type == HeatmapType.FEATURE_IMPORTANCE:
            return self.generate_feature_importance_heatmap(
                data,
                title=kwargs.get('title', '特征重要性热力图'),
                filename=kwargs.get('filename', 'feature_importance_heatmap')
            )
        elif heatmap_type == HeatmapType.CORRELATION:
            return self.generate_correlation_heatmap(
                data,
                title=kwargs.get('title', '相关性热力图'),
                filename=kwargs.get('filename', 'correlation_heatmap')
            )
        else:
            return self.generate_comprehensive_heatmap(
                data,
                title=kwargs.get('title', '综合热力图'),
                filename=kwargs.get('filename', 'comprehensive_heatmap')
            )


# 便捷函数
def create_heatmap_generator(output_dir: str = "output",
                           vis_config: Optional[VisualizationConfig] = None,
                           heatmap_config: Optional[HeatmapConfig] = None,
                           **kwargs) -> HeatmapGenerator:
    """创建热力图生成器的便捷函数"""
    return HeatmapGenerator(
        output_dir=output_dir,
        vis_config=vis_config,
        heatmap_config=heatmap_config
    )