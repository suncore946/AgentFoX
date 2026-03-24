import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 忽略matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class ColorPalette(Enum):
    """调色板枚举"""
    PROFESSIONAL = "professional"  # 专业商务风格
    SCIENTIFIC = "scientific"      # 科学研究风格
    MODERN = "modern"              # 现代简约风格
    COLORFUL = "colorful"          # 多彩活泼风格
    MONOCHROME = "monochrome"      # 单色风格


class PlotStyle(Enum):
    """绘图风格枚举"""
    CLEAN = "clean"                # 简洁风格
    DETAILED = "detailed"          # 详细风格
    MINIMAL = "minimal"            # 极简风格
    CLASSIC = "classic"            # 经典风格


@dataclass
class VisualizationConfig:
    """可视化配置类"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    color_palette: ColorPalette = ColorPalette.PROFESSIONAL
    plot_style: PlotStyle = PlotStyle.CLEAN
    font_size: int = 12
    title_size: int = 16
    label_size: int = 11
    legend_size: int = 10
    grid: bool = True
    spines: bool = True
    tight_layout: bool = True
    save_transparent: bool = False
    save_format: str = 'png'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'figure_size': self.figure_size,
            'dpi': self.dpi,
            'color_palette': self.color_palette.value,
            'plot_style': self.plot_style.value,
            'font_size': self.font_size,
            'title_size': self.title_size,
            'label_size': self.label_size,
            'legend_size': self.legend_size,
            'grid': self.grid,
            'spines': self.spines,
            'tight_layout': self.tight_layout,
            'save_transparent': self.save_transparent,
            'save_format': self.save_format
        }


class BaseVisualizer(ABC):
    """基础可视化器抽象类
    
    定义所有可视化器的通用接口和功能
    """
    
    def __init__(self,
                 output_dir: str = "output",
                 config: Optional[VisualizationConfig] = None):
        """初始化基础可视化器
        
        Args:
            output_dir: 输出目录
            config: 可视化配置
        """
        self.output_dir = output_dir
        self.config = config or VisualizationConfig()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化matplotlib和seaborn设置
        self._setup_matplotlib()
        self._setup_seaborn()
    
    def _setup_matplotlib(self):
        """设置matplotlib参数"""
        plt.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'legend.fontsize': self.config.legend_size,
            'xtick.labelsize': self.config.label_size,
            'ytick.labelsize': self.config.label_size,
            'grid.alpha': 0.3,
            'axes.grid': self.config.grid,
            'axes.spines.left': self.config.spines,
            'axes.spines.bottom': self.config.spines,
            'axes.spines.top': self.config.spines,
            'axes.spines.right': self.config.spines,
            'figure.autolayout': self.config.tight_layout
        })
    
    def _setup_seaborn(self):
        """设置seaborn样式"""
        if self.config.plot_style == PlotStyle.CLEAN:
            sns.set_style("whitegrid")
        elif self.config.plot_style == PlotStyle.MINIMAL:
            sns.set_style("white")
        elif self.config.plot_style == PlotStyle.DETAILED:
            sns.set_style("darkgrid")
        else:  # CLASSIC
            sns.set_style("ticks")
        
        # 设置调色板
        colors = self._get_color_palette()
        sns.set_palette(colors)
    
    def _get_color_palette(self) -> List[str]:
        """获取调色板颜色
        
        Returns:
            List[str]: 颜色列表
        """
        palettes = {
            ColorPalette.PROFESSIONAL: [
                '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4A5D23',
                '#7209B7', '#560BAD', '#480CA8', '#3A0CA3', '#3F37C9'
            ],
            ColorPalette.SCIENTIFIC: [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ],
            ColorPalette.MODERN: [
                '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#6C5CE7',
                '#00B894', '#00CEC9', '#0984E3', '#6C5CE7', '#A29BFE'
            ],
            ColorPalette.COLORFUL: [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
            ],
            ColorPalette.MONOCHROME: [
                '#2C3E50', '#34495E', '#5D6D7E', '#85929E', '#AEB6BF',
                '#D5DBDB', '#EAEDED', '#F8F9F9', '#FFFFFF', '#000000'
            ]
        }
        
        return palettes.get(self.config.color_palette, palettes[ColorPalette.PROFESSIONAL])
    
    def get_colors(self, n: int) -> List[str]:
        """获取指定数量的颜色
        
        Args:
            n: 需要的颜色数量
            
        Returns:
            List[str]: 颜色列表
        """
        base_colors = self._get_color_palette()
        
        if n <= len(base_colors):
            return base_colors[:n]
        
        # 如果需要更多颜色，生成渐变色
        extended_colors = base_colors[:]
        while len(extended_colors) < n:
            # 添加颜色的变体
            for base_color in base_colors:
                if len(extended_colors) >= n:
                    break
                # 生成较浅的变体
                lighter = self._lighten_color(base_color, 0.3)
                extended_colors.append(lighter)
        
        return extended_colors[:n]
    
    def _lighten_color(self, color: str, amount: float) -> str:
        """使颜色变浅
        
        Args:
            color: 原始颜色
            amount: 变浅程度 (0-1)
            
        Returns:
            str: 变浅后的颜色
        """
        try:
            c = mcolors.to_rgb(color)
            c = [(1 - amount) * x + amount for x in c]
            return mcolors.to_hex(c)
        except:
            return color
    
    def save_figure(self,
                   fig: plt.Figure,
                   filename: str,
                   subfolder: str = "",
                   **kwargs) -> str:
        """保存图表
        
        Args:
            fig: matplotlib图表对象
            filename: 文件名
            subfolder: 子文件夹名
            **kwargs: 其他保存参数
            
        Returns:
            str: 保存的文件路径
        """
        # 构建完整路径
        if subfolder:
            save_dir = os.path.join(self.output_dir, subfolder)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        # 确保文件扩展名
        if not filename.endswith(f'.{self.config.save_format}'):
            filename = f"{filename}.{self.config.save_format}"
        
        filepath = os.path.join(save_dir, filename)
        
        # 设置保存参数
        save_kwargs = {
            'dpi': self.config.dpi,
            'bbox_inches': 'tight',
            'transparent': self.config.save_transparent,
            'format': self.config.save_format
        }
        save_kwargs.update(kwargs)
        
        # 保存图表
        try:
            fig.savefig(filepath, **save_kwargs)
            self.logger.info(f"图表已保存: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"保存图表失败: {e}")
            raise
    
    def create_figure(self,
                     figsize: Optional[Tuple[int, int]] = None,
                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """创建图表和坐标轴
        
        Args:
            figsize: 图表尺寸
            **kwargs: 其他参数
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: 图表和坐标轴对象
        """
        if figsize is None:
            figsize = self.config.figure_size
        
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        return fig, ax
    
    def create_subplots(self,
                       nrows: int = 1,
                       ncols: int = 1,
                       figsize: Optional[Tuple[int, int]] = None,
                       **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """创建子图
        
        Args:
            nrows: 行数
            ncols: 列数
            figsize: 图表尺寸
            **kwargs: 其他参数
            
        Returns:
            Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]: 图表和坐标轴对象
        """
        if figsize is None:
            # 根据子图数量调整尺寸
            base_width, base_height = self.config.figure_size
            figsize = (base_width * ncols, base_height * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes
    
    def apply_theme(self, ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""):
        """应用主题样式
        
        Args:
            ax: 坐标轴对象
            title: 标题
            xlabel: X轴标签
            ylabel: Y轴标签
        """
        if title:
            ax.set_title(title, fontsize=self.config.title_size, fontweight='bold', pad=20)
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config.label_size)
            
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config.label_size)
        
        # 设置网格
        if self.config.grid:
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 设置边框
        if not self.config.spines:
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    def add_statistical_annotations(self,
                                  ax: plt.Axes,
                                  statistical_results: Dict[str, Any],
                                  position: str = "top"):
        """添加统计显著性标注
        
        Args:
            ax: 坐标轴对象
            statistical_results: 统计结果
            position: 标注位置
        """
        if not statistical_results:
            return
        
        # 提取p值和显著性
        annotations = []
        for test_name, result in statistical_results.items():
            if isinstance(result, dict) and 'p_value' in result:
                p_value = result['p_value']
                if p_value < 0.001:
                    sig_symbol = "***"
                elif p_value < 0.01:
                    sig_symbol = "**"
                elif p_value < 0.05:
                    sig_symbol = "*"
                else:
                    sig_symbol = "ns"
                
                annotations.append(f"{test_name}: {sig_symbol}")
        
        # 添加标注
        if annotations:
            annotation_text = ", ".join(annotations)
            if position == "top":
                y_pos = 0.95
            else:  # bottom
                y_pos = 0.05
            
            ax.text(0.02, y_pos, annotation_text, transform=ax.transAxes,
                   fontsize=self.config.legend_size, style='italic',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    @abstractmethod
    def generate(self, data: Dict[str, Any], **kwargs) -> str:
        """生成可视化图表（抽象方法）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文件路径
        """
        pass
    
    def cleanup(self):
        """清理资源"""
        plt.close('all')


# 便捷函数
def create_base_visualizer(output_dir: str = "output",
                          config: Optional[VisualizationConfig] = None) -> BaseVisualizer:
    """创建基础可视化器的便捷函数"""
    # 由于BaseVisualizer是抽象类，这里返回一个简单的实现
    class SimpleVisualizer(BaseVisualizer):
        def generate(self, data: Dict[str, Any], **kwargs) -> str:
            return ""
    
    return SimpleVisualizer(output_dir, config)


def get_default_config(style: str = "professional") -> VisualizationConfig:
    """获取默认配置
    
    Args:
        style: 风格名称
        
    Returns:
        VisualizationConfig: 配置对象
    """
    style_configs = {
        "professional": VisualizationConfig(
            color_palette=ColorPalette.PROFESSIONAL,
            plot_style=PlotStyle.CLEAN
        ),
        "scientific": VisualizationConfig(
            color_palette=ColorPalette.SCIENTIFIC,
            plot_style=PlotStyle.DETAILED
        ),
        "modern": VisualizationConfig(
            color_palette=ColorPalette.MODERN,
            plot_style=PlotStyle.MINIMAL
        ),
        "colorful": VisualizationConfig(
            color_palette=ColorPalette.COLORFUL,
            plot_style=PlotStyle.CLEAN
        )
    }
    
    return style_configs.get(style, VisualizationConfig())