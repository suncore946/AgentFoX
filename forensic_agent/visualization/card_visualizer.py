import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import textwrap

from .base_visualizer import BaseVisualizer, VisualizationConfig
from ..llm_integration.card_generator import ModelCard, SegmentCard, CardFormat


class CardLayout(Enum):
    """卡片布局枚举"""
    SINGLE = "single"          # 单卡片布局
    GRID = "grid"              # 网格布局
    FLOW = "flow"              # 流式布局
    COMPARISON = "comparison"   # 对比布局


class VisualizationTheme(Enum):
    """可视化主题枚举"""
    PROFESSIONAL = "professional"  # 专业主题
    MODERN = "modern"              # 现代主题
    MINIMAL = "minimal"            # 极简主题
    COLORFUL = "colorful"          # 多彩主题


@dataclass
class CardVisualizationConfig:
    """卡片可视化配置"""
    layout: CardLayout = CardLayout.SINGLE
    theme: VisualizationTheme = VisualizationTheme.PROFESSIONAL
    card_width: int = 800
    card_height: int = 600
    margin: int = 20
    padding: int = 15
    title_size: int = 18
    text_size: int = 12
    background_color: str = "#FFFFFF"
    border_color: str = "#E0E0E0"
    accent_color: str = "#2E86AB"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'layout': self.layout.value,
            'theme': self.theme.value,
            'card_width': self.card_width,
            'card_height': self.card_height,
            'margin': self.margin,
            'padding': self.padding,
            'title_size': self.title_size,
            'text_size': self.text_size,
            'background_color': self.background_color,
            'border_color': self.border_color,
            'accent_color': self.accent_color
        }


class CardVisualizer(BaseVisualizer):
    """卡片可视化器
    
    将LLM生成的卡片转换为可视化形式
    """
    
    def __init__(self,
                 output_dir: str = "output",
                 vis_config: Optional[VisualizationConfig] = None,
                 card_config: Optional[CardVisualizationConfig] = None):
        """初始化卡片可视化器
        
        Args:
            output_dir: 输出目录
            vis_config: 可视化配置
            card_config: 卡片可视化配置
        """
        super().__init__(output_dir, vis_config)
        self.card_config = card_config or CardVisualizationConfig()
        
        # 创建卡片子目录
        self.cards_dir = os.path.join(self.output_dir, "cards")
        os.makedirs(self.cards_dir, exist_ok=True)
        
        # 设置主题颜色
        self._setup_theme_colors()
    
    def _setup_theme_colors(self):
        """设置主题颜色"""
        theme_colors = {
            VisualizationTheme.PROFESSIONAL: {
                'background': '#FFFFFF',
                'border': '#E0E0E0',
                'accent': '#2E86AB',
                'text': '#2C3E50',
                'highlight': '#F8F9FA'
            },
            VisualizationTheme.MODERN: {
                'background': '#FAFBFC',
                'border': '#E1E4E8',
                'accent': '#6C5CE7',
                'text': '#24292E',
                'highlight': '#F6F8FA'
            },
            VisualizationTheme.MINIMAL: {
                'background': '#FFFFFF',
                'border': '#F0F0F0',
                'accent': '#000000',
                'text': '#333333',
                'highlight': '#F9F9F9'
            },
            VisualizationTheme.COLORFUL: {
                'background': '#FFFFFF',
                'border': '#E3F2FD',
                'accent': '#FF6B6B',
                'text': '#2C3E50',
                'highlight': '#E8F5E8'
            }
        }
        
        self.theme_colors = theme_colors.get(
            self.card_config.theme, 
            theme_colors[VisualizationTheme.PROFESSIONAL]
        )
    
    def visualize_model_card(self,
                           model_card: ModelCard,
                           filename: Optional[str] = None) -> str:
        """可视化模型卡片
        
        Args:
            model_card: 模型卡片对象
            filename: 输出文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info(f"可视化模型卡片: {model_card.model_name}")
        
        if filename is None:
            filename = f"model_card_{model_card.model_name.replace(' ', '_')}"
        
        # 创建图表
        fig, ax = self.create_figure(
            figsize=(self.card_config.card_width/100, self.card_config.card_height/100)
        )
        
        # 绘制卡片
        self._draw_model_card(ax, model_card)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "cards")
        plt.close(fig)
        
        return filepath
    
    def visualize_segment_card(self,
                             segment_card: SegmentCard,
                             filename: Optional[str] = None) -> str:
        """可视化片段卡片
        
        Args:
            segment_card: 片段卡片对象
            filename: 输出文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info(f"可视化片段卡片: {segment_card.segment_name}")
        
        if filename is None:
            filename = f"segment_card_{segment_card.segment_id}"
        
        # 创建图表
        fig, ax = self.create_figure(
            figsize=(self.card_config.card_width/100, self.card_config.card_height/100)
        )
        
        # 绘制卡片
        self._draw_segment_card(ax, segment_card)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "cards")
        plt.close(fig)
        
        return filepath
    
    def visualize_multiple_cards(self,
                               cards: List[Union[ModelCard, SegmentCard]],
                               layout: Optional[CardLayout] = None,
                               filename: str = "multiple_cards") -> str:
        """可视化多个卡片
        
        Args:
            cards: 卡片列表
            layout: 布局方式
            filename: 输出文件名
            
        Returns:
            str: 生成的文件路径
        """
        self.logger.info(f"可视化多个卡片，数量: {len(cards)}")
        
        if not cards:
            self.logger.warning("卡片列表为空")
            return ""
        
        layout = layout or self.card_config.layout
        
        if layout == CardLayout.GRID:
            return self._visualize_grid_layout(cards, filename)
        elif layout == CardLayout.FLOW:
            return self._visualize_flow_layout(cards, filename)
        elif layout == CardLayout.COMPARISON:
            return self._visualize_comparison_layout(cards, filename)
        else:  # SINGLE
            return self._visualize_single_layout(cards, filename)
    
    def _draw_model_card(self, ax: plt.Axes, model_card: ModelCard):
        """绘制模型卡片
        
        Args:
            ax: 坐标轴对象
            model_card: 模型卡片
        """
        # 设置坐标轴
        ax.set_xlim(0, self.card_config.card_width)
        ax.set_ylim(0, self.card_config.card_height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制卡片背景
        card_bg = FancyBboxPatch(
            (self.card_config.margin, self.card_config.margin),
            self.card_config.card_width - 2*self.card_config.margin,
            self.card_config.card_height - 2*self.card_config.margin,
            boxstyle="round,pad=10",
            facecolor=self.theme_colors['background'],
            edgecolor=self.theme_colors['border'],
            linewidth=2
        )
        ax.add_patch(card_bg)
        
        # 绘制标题区域
        title_y = self.card_config.card_height - self.card_config.margin - 40
        title_bg = Rectangle(
            (self.card_config.margin + 10, title_y - 15),
            self.card_config.card_width - 2*self.card_config.margin - 20,
            50,
            facecolor=self.theme_colors['accent'],
            alpha=0.1
        )
        ax.add_patch(title_bg)
        
        # 添加模型名称标题
        ax.text(
            self.card_config.card_width // 2, title_y + 10,
            f"模型性能分析卡片: {model_card.model_name}",
            fontsize=self.card_config.title_size,
            fontweight='bold',
            ha='center',
            va='center',
            color=self.theme_colors['text']
        )
        
        # 绘制内容区域
        content_y_start = title_y - 60
        self._draw_model_card_content(ax, model_card, content_y_start)
        
        # 添加生成时间
        ax.text(
            self.card_config.card_width - self.card_config.margin - 10,
            self.card_config.margin + 10,
            f"生成时间: {model_card.generation_timestamp[:19]}",
            fontsize=self.card_config.text_size - 2,
            ha='right',
            va='bottom',
            color=self.theme_colors['text'],
            alpha=0.7
        )
    
    def _draw_model_card_content(self, ax: plt.Axes, model_card: ModelCard, start_y: int):
        """绘制模型卡片内容
        
        Args:
            ax: 坐标轴对象
            model_card: 模型卡片
            start_y: 起始Y坐标
        """
        current_y = start_y
        margin_x = self.card_config.margin + self.card_config.padding
        content_width = self.card_config.card_width - 2 * margin_x
        
        # 解析卡片内容（简化的markdown解析）
        content_lines = model_card.card_content.split('\n')
        
        for line in content_lines:
            if not line.strip():
                current_y -= 15  # 空行间距
                continue
            
            # 检查是否是标题
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title_text = line.lstrip('# ').strip()
                
                if level == 1:  # 主标题
                    current_y -= 30
                    ax.text(
                        margin_x, current_y,
                        title_text,
                        fontsize=self.card_config.text_size + 4,
                        fontweight='bold',
                        ha='left',
                        va='top',
                        color=self.theme_colors['accent']
                    )
                    current_y -= 25
                    
                    # 添加下划线
                    ax.plot(
                        [margin_x, margin_x + len(title_text) * 8],
                        [current_y, current_y],
                        color=self.theme_colors['accent'],
                        linewidth=2
                    )
                    current_y -= 15
                
                elif level == 2:  # 副标题
                    current_y -= 20
                    ax.text(
                        margin_x, current_y,
                        title_text,
                        fontsize=self.card_config.text_size + 2,
                        fontweight='bold',
                        ha='left',
                        va='top',
                        color=self.theme_colors['text']
                    )
                    current_y -= 20
            
            # 列表项
            elif line.strip().startswith('-') or line.strip().startswith('*'):
                list_text = line.strip().lstrip('-* ').strip()
                current_y -= 15
                
                # 添加圆点
                ax.plot(
                    margin_x + 10, current_y - 5,
                    'o',
                    color=self.theme_colors['accent'],
                    markersize=3
                )
                
                # 分行显示长文本
                wrapped_lines = textwrap.wrap(list_text, width=80)
                for wrapped_line in wrapped_lines:
                    ax.text(
                        margin_x + 25, current_y,
                        wrapped_line,
                        fontsize=self.card_config.text_size,
                        ha='left',
                        va='top',
                        color=self.theme_colors['text']
                    )
                    current_y -= 15
            
            # 普通文本
            else:
                current_y -= 15
                wrapped_lines = textwrap.wrap(line.strip(), width=90)
                for wrapped_line in wrapped_lines:
                    ax.text(
                        margin_x, current_y,
                        wrapped_line,
                        fontsize=self.card_config.text_size,
                        ha='left',
                        va='top',
                        color=self.theme_colors['text']
                    )
                    current_y -= 15
            
            # 防止内容超出卡片底部
            if current_y < self.card_config.margin + 50:
                break
    
    def _draw_segment_card(self, ax: plt.Axes, segment_card: SegmentCard):
        """绘制片段卡片
        
        Args:
            ax: 坐标轴对象
            segment_card: 片段卡片
        """
        # 设置坐标轴
        ax.set_xlim(0, self.card_config.card_width)
        ax.set_ylim(0, self.card_config.card_height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制卡片背景
        card_bg = FancyBboxPatch(
            (self.card_config.margin, self.card_config.margin),
            self.card_config.card_width - 2*self.card_config.margin,
            self.card_config.card_height - 2*self.card_config.margin,
            boxstyle="round,pad=10",
            facecolor=self.theme_colors['background'],
            edgecolor=self.theme_colors['border'],
            linewidth=2
        )
        ax.add_patch(card_bg)
        
        # 绘制标题区域
        title_y = self.card_config.card_height - self.card_config.margin - 40
        title_bg = Rectangle(
            (self.card_config.margin + 10, title_y - 15),
            self.card_config.card_width - 2*self.card_config.margin - 20,
            50,
            facecolor=self.theme_colors['highlight'],
            alpha=0.8
        )
        ax.add_patch(title_bg)
        
        # 添加片段名称标题
        ax.text(
            self.card_config.card_width // 2, title_y + 10,
            f"内容类型: {segment_card.segment_name}",
            fontsize=self.card_config.title_size,
            fontweight='bold',
            ha='center',
            va='center',
            color=self.theme_colors['text']
        )
        
        # 绘制性能指标区域
        metrics_y = title_y - 80
        self._draw_performance_metrics(ax, segment_card.performance_metrics, metrics_y)
        
        # 绘制内容区域
        content_y_start = metrics_y - 120
        self._draw_segment_card_content(ax, segment_card, content_y_start)
    
    def _draw_performance_metrics(self, ax: plt.Axes, metrics: Dict[str, float], start_y: int):
        """绘制性能指标
        
        Args:
            ax: 坐标轴对象
            metrics: 性能指标字典
            start_y: 起始Y坐标
        """
        if not metrics:
            return
        
        margin_x = self.card_config.margin + self.card_config.padding
        current_y = start_y
        
        # 绘制性能指标标题
        ax.text(
            margin_x, current_y,
            "性能指标",
            fontsize=self.card_config.text_size + 2,
            fontweight='bold',
            ha='left',
            va='top',
            color=self.theme_colors['accent']
        )
        current_y -= 25
        
        # 绘制指标条形图（简化版）
        bar_width = (self.card_config.card_width - 2 * margin_x - 100) / len(metrics)
        bar_x = margin_x + 100
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            # 绘制指标名称
            ax.text(
                margin_x, current_y,
                f"{metric_name}:",
                fontsize=self.card_config.text_size,
                ha='left',
                va='center',
                color=self.theme_colors['text']
            )
            
            # 绘制数值
            ax.text(
                margin_x + 80, current_y,
                f"{value:.4f}" if isinstance(value, float) else str(value),
                fontsize=self.card_config.text_size,
                ha='left',
                va='center',
                color=self.theme_colors['accent'],
                fontweight='bold'
            )
            
            current_y -= 20
    
    def _draw_segment_card_content(self, ax: plt.Axes, segment_card: SegmentCard, start_y: int):
        """绘制片段卡片内容
        
        Args:
            ax: 坐标轴对象
            segment_card: 片段卡片
            start_y: 起始Y坐标
        """
        # 使用与模型卡片类似的内容绘制逻辑
        self._draw_model_card_content(ax, segment_card, start_y)
    
    def _visualize_grid_layout(self, cards: List[Union[ModelCard, SegmentCard]], filename: str) -> str:
        """网格布局可视化
        
        Args:
            cards: 卡片列表
            filename: 文件名
            
        Returns:
            str: 文件路径
        """
        n_cards = len(cards)
        cols = min(3, n_cards)  # 最多3列
        rows = (n_cards + cols - 1) // cols
        
        # 计算总尺寸
        total_width = cols * (self.card_config.card_width + self.card_config.margin) + self.card_config.margin
        total_height = rows * (self.card_config.card_height + self.card_config.margin) + self.card_config.margin
        
        # 创建大图表
        fig, ax = self.create_figure(figsize=(total_width/100, total_height/100))
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制每个卡片
        for i, card in enumerate(cards):
            row = i // cols
            col = i % cols
            
            # 计算卡片位置
            card_x = col * (self.card_config.card_width + self.card_config.margin) + self.card_config.margin
            card_y = total_height - (row + 1) * (self.card_config.card_height + self.card_config.margin)
            
            # 创建子区域
            card_ax = fig.add_axes([
                card_x / total_width,
                card_y / total_height,
                self.card_config.card_width / total_width,
                self.card_config.card_height / total_height
            ])
            
            # 绘制卡片
            if isinstance(card, ModelCard):
                self._draw_model_card(card_ax, card)
            else:
                self._draw_segment_card(card_ax, card)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "cards")
        plt.close(fig)
        
        return filepath
    
    def _visualize_flow_layout(self, cards: List[Union[ModelCard, SegmentCard]], filename: str) -> str:
        """流式布局可视化
        
        Args:
            cards: 卡片列表
            filename: 文件名
            
        Returns:
            str: 文件路径
        """
        # 简化为垂直流式布局
        total_width = self.card_config.card_width + 2 * self.card_config.margin
        total_height = len(cards) * (self.card_config.card_height + self.card_config.margin) + self.card_config.margin
        
        # 创建大图表
        fig, ax = self.create_figure(figsize=(total_width/100, total_height/100))
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制每个卡片
        for i, card in enumerate(cards):
            card_y = total_height - (i + 1) * (self.card_config.card_height + self.card_config.margin)
            
            # 创建子区域
            card_ax = fig.add_axes([
                self.card_config.margin / total_width,
                card_y / total_height,
                self.card_config.card_width / total_width,
                self.card_config.card_height / total_height
            ])
            
            # 绘制卡片
            if isinstance(card, ModelCard):
                self._draw_model_card(card_ax, card)
            else:
                self._draw_segment_card(card_ax, card)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "cards")
        plt.close(fig)
        
        return filepath
    
    def _visualize_comparison_layout(self, cards: List[Union[ModelCard, SegmentCard]], filename: str) -> str:
        """对比布局可视化
        
        Args:
            cards: 卡片列表
            filename: 文件名
            
        Returns:
            str: 文件路径
        """
        # 简化为并排比较（最多4个）
        n_cards = min(len(cards), 4)
        
        total_width = n_cards * (self.card_config.card_width + self.card_config.margin) + self.card_config.margin
        total_height = self.card_config.card_height + 2 * self.card_config.margin
        
        # 创建大图表
        fig, ax = self.create_figure(figsize=(total_width/100, total_height/100))
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制每个卡片
        for i, card in enumerate(cards[:n_cards]):
            card_x = i * (self.card_config.card_width + self.card_config.margin) + self.card_config.margin
            
            # 创建子区域
            card_ax = fig.add_axes([
                card_x / total_width,
                self.card_config.margin / total_height,
                self.card_config.card_width / total_width,
                self.card_config.card_height / total_height
            ])
            
            # 绘制卡片
            if isinstance(card, ModelCard):
                self._draw_model_card(card_ax, card)
            else:
                self._draw_segment_card(card_ax, card)
        
        # 保存图表
        filepath = self.save_figure(fig, filename, "cards")
        plt.close(fig)
        
        return filepath
    
    def _visualize_single_layout(self, cards: List[Union[ModelCard, SegmentCard]], filename: str) -> str:
        """单卡片布局（选择第一个卡片）
        
        Args:
            cards: 卡片列表
            filename: 文件名
            
        Returns:
            str: 文件路径
        """
        if not cards:
            return ""
        
        card = cards[0]
        if isinstance(card, ModelCard):
            return self.visualize_model_card(card, filename)
        else:
            return self.visualize_segment_card(card, filename)
    
    def generate(self, data: Dict[str, Any], **kwargs) -> str:
        """生成卡片可视化（实现抽象方法）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文件路径
        """
        if 'model_card' in data:
            return self.visualize_model_card(data['model_card'], kwargs.get('filename'))
        elif 'segment_card' in data:
            return self.visualize_segment_card(data['segment_card'], kwargs.get('filename'))
        elif 'cards' in data:
            return self.visualize_multiple_cards(
                data['cards'], 
                kwargs.get('layout'),
                kwargs.get('filename', 'multiple_cards')
            )
        else:
            self.logger.warning("未找到可视化的卡片数据")
            return ""


# 便捷函数
def create_card_visualizer(output_dir: str = "output",
                          theme: str = "professional",
                          vis_config: Optional[VisualizationConfig] = None,
                          card_config: Optional[CardVisualizationConfig] = None,
                          **kwargs) -> CardVisualizer:
    """创建卡片可视化器的便捷函数"""
    if card_config is None:
        theme_enum = VisualizationTheme(theme) if isinstance(theme, str) else theme
        card_config = CardVisualizationConfig(theme=theme_enum)
    
    return CardVisualizer(
        output_dir=output_dir,
        vis_config=vis_config,
        card_config=card_config
    )