from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import warnings
from sklearn.decomposition import PCA

from ...utils.logger import get_logger
from .clustering_dataclass import ClusteringInfo


@dataclass
class VisualizationConfig:
    """Visualization configuration parameters"""

    figsize: Tuple[int, int] = (15, 12)
    dpi: int = 300
    style: str = "whitegrid"
    palette: str = "Set1"
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    alpha: float = 0.7
    marker_size: int = 50
    save_format: str = "png"


class ClusteringVisualizer:
    """Clustering results visualizer

    Renders 2 core charts to comprehensively display clustering performance and results:
    1. Cluster spatial distribution plot - Shows cluster distribution in dimensionally reduced space
    2. Cluster statistical features plot - Shows cluster size distribution and coverage statistics
    """

    # Predefined color schemes
    ELEGANT_COLORS = [
        "#FF6B6B",  # 珊瑚红
        "#4ECDC4",  # 青绿色
        "#45B7D1",  # 天蓝色
        "#96CEB4",  # 薄荷绿
        "#FECA57",  # 金黄色
        "#FF9FF3",  # 粉紫色
        "#54A0FF",  # 亮蓝色
        "#5F27CD",  # 深紫色
        "#00D2D3",  # 青色
        "#FF9F43",  # 橙色
        "#C44569",  # 玫瑰色
        "#F8B500",  # 琥珀色
    ]

    ELEGANT_NOISE_COLOR = "#E8E8E8"  # 优雅的灰色

    NOISE_COLOR = "#bdbdbd"  # Gray color for noise points

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer

        Args:
            config: Visualization configuration, uses default if None
        """
        self.config = config or VisualizationConfig()
        self.logger = get_logger(__name__)

        # 设置matplotlib为非交互式模式，不显示图形
        matplotlib.use("Agg")
        plt.ioff()  # 关闭交互模式

        # Set matplotlib style
        plt.style.use("default")
        sns.set_style(self.config.style)

        # Ignore matplotlib warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

        self.logger.info(f"Clustering visualizer initialized - Figure size: {self.config.figsize}, DPI: {self.config.dpi}")

    def visualize(
        self,
        discovery_result: ClusteringInfo,  # 修正参数名
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generate complete visualization of clustering results

        Args:
            discovery_result: Clustering discovery results
            save_path: Save path, saves image if provided

        Returns:
            matplotlib.Figure: Complete visualization figure with 2 subplots
        """
        # Create layout with 2 subplots
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        gs = fig.add_gridspec(1, 2, height_ratios=[1], width_ratios=[1.2, 1])

        # 确保不显示图形
        plt.ioff()

        try:
            # Subplot 1: Cluster spatial distribution plot
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_cluster_distribution(ax1, discovery_result)

            # Subplot 2: Cluster statistical features plot
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_cluster_statistics(ax2, discovery_result)

            # Set overall title
            n_clusters = len(discovery_result.cluster_types)
            silhouette_score = discovery_result.quality_metrics.get("silhouette_score", 0.0)

            fig.suptitle(
                f"Clustering Results Visualization - {discovery_result.clustering_columns}\n"
                f"Discovered {n_clusters} clusters, Total samples: {len(discovery_result.clustering_data)}, "
                f"Silhouette score: {silhouette_score:.4f}",
                fontsize=self.config.title_fontsize + 1,
                fontweight="bold",
                y=0.98,
            )

            # Adjust layout
            try:
                plt.tight_layout()
            except Exception as e:
                self.logger.warning(f"tight_layout failed: {e}, using subplots_adjust")
                plt.subplots_adjust(top=0.85, wspace=0.4, hspace=0.3)

            # Save figure
            if save_path:
                self._save_figure(fig, save_path)

            self.logger.info(f"Clustering visualization generated - {n_clusters} clusters")
            return fig

        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
            plt.close(fig)
            raise

    def _prepare_plot_data(self, cluster_data: Union[np.ndarray, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare data for plotting"""
        try:
            if isinstance(cluster_data, pd.DataFrame):
                if cluster_data.empty:
                    return None
                # 选择数值列
                numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    self.logger.warning("No numeric columns found in DataFrame")
                    return None
                return cluster_data[numeric_cols].fillna(0)

            elif isinstance(cluster_data, np.ndarray):
                if cluster_data.size == 0:
                    return None
                return pd.DataFrame(cluster_data)
            else:
                # 尝试转换为DataFrame
                return pd.DataFrame(cluster_data)

        except Exception as e:
            self.logger.error(f"Failed to prepare plot data: {e}")
            return None

    def _plot_cluster_distribution(self, ax: plt.Axes, discovery_result: ClusteringInfo) -> None:
        """Plot cluster spatial distribution"""
        feature_name = f"scale of {discovery_result.clustering_columns}"

        # 准备数据
        plot_data = self._prepare_plot_data(discovery_result.clustering_data)

        if plot_data is None or plot_data.empty:
            ax.text(
                0.5,
                0.5,
                "No valid data for visualization",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.config.label_fontsize,
            )
            ax.set_title("Cluster Distribution", fontsize=self.config.title_fontsize)
            return

        # 根据数据维度选择不同的可视化方法
        n_dimensions = plot_data.shape[1]

        if n_dimensions == 1:
            # 1维数据：使用箱线图和密度图
            self._plot_1d_clusters(ax, plot_data, discovery_result, feature_name)
        elif n_dimensions == 2:
            # 2维数据：直接使用散点图
            coords = plot_data.values
            self._plot_2d_clusters(ax, coords, discovery_result, "Direct 2D projection")
        elif n_dimensions > 2:
            # 多维数据：使用PCA降维到2D
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(plot_data)
            variance_info = f"PC1: {pca.explained_variance_ratio_[0]:.2%}, " f"PC2: {pca.explained_variance_ratio_[1]:.2%}"
            self._plot_2d_clusters(ax, coords, discovery_result, variance_info)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data dimensions for visualization",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.config.label_fontsize,
            )
            ax.set_title("Cluster Distribution", fontsize=self.config.title_fontsize)

    def _plot_1d_clusters(self, ax: plt.Axes, data: pd.DataFrame, discovery_result: ClusteringInfo, feature_name) -> None:
        """为1维数据绘制聚类分布图"""
        # 获取数据值
        values = data.iloc[:, 0].values

        # 创建标签数组
        labels = np.full(len(values), -1)  # 默认为噪声
        colors = []

        # 为每个聚类分配标签和颜色
        for i, content_type in enumerate(discovery_result.result):
            cluster_color = self.ELEGANT_COLORS[i % len(self.ELEGANT_COLORS)]
            colors.append(cluster_color)

            for idx in content_type.sample_indices:
                if 0 <= idx < len(labels):
                    labels[idx] = i

        # 准备聚类数据
        unique_labels = np.unique(labels)
        cluster_data = []
        cluster_names = []
        ELEGANT_COLORS = []

        for label in unique_labels:
            mask = labels == label
            cluster_values = values[mask]

            if len(cluster_values) > 0:
                cluster_data.append(cluster_values)

                if label == -1:  # 噪声点
                    cluster_names.append(f"Noise (n={mask.sum()})")
                    ELEGANT_COLORS.append(self.NOISE_COLOR)
                else:  # 聚类点
                    if label < len(discovery_result.result):
                        cluster_name = discovery_result.result[label].type_id
                        cluster_names.append(f"{cluster_name} (n={mask.sum()})")
                        ELEGANT_COLORS.append(colors[label] if label < len(colors) else self.ELEGANT_COLORS[0])

        if not cluster_data:
            ax.text(0.5, 0.5, "No valid cluster data", ha="center", va="center", transform=ax.transAxes)
            return

        try:
            # 使用简化的可视化方法：水平箱线图
            box_positions = list(range(1, len(cluster_data) + 1))
            bp = ax.boxplot(
                cluster_data,
                positions=box_positions,
                patch_artist=True,
                vert=False,
                widths=0.6,
                showfliers=True,
                flierprops=dict(marker="o", markersize=4, alpha=0.6),
            )

            # 设置箱型图颜色
            for patch, color in zip(bp["boxes"], ELEGANT_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # 设置图表属性
            ax.set_yticks(box_positions)
            ax.set_yticklabels(cluster_names, fontsize=self.config.legend_fontsize)
            ax.set_xlabel(feature_name, fontsize=self.config.label_fontsize)
            ax.grid(True, alpha=0.3, axis="x")
            ax.set_title("Cluster Distribution (1D)", fontsize=self.config.title_fontsize)

        except Exception as e:
            self.logger.error(f"Error plotting 1D clusters: {e}")
            ax.text(0.5, 0.5, f"Plot error: {str(e)}", ha="center", va="center", transform=ax.transAxes)

    def _plot_2d_clusters(self, ax: plt.Axes, coords: np.ndarray, discovery_result: ClusteringInfo, variance_info: str) -> None:
        """为2D数据绘制聚类散点图"""
        # 创建标签数组
        labels = np.full(len(coords), -1)
        colors = []

        # 分配颜色和标签
        for i, content_type in enumerate(discovery_result.cluster_types):
            cluster_color = self.ELEGANT_COLORS[i % len(self.ELEGANT_COLORS)]
            colors.append(cluster_color)

            for idx in content_type.sample_indices:
                if 0 <= idx < len(labels):
                    labels[idx] = i
                else:
                    self.logger.warning(f"Sample index {idx} out of bounds for data of length {len(labels)}")

        # 绘制聚类点
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            if not np.any(mask):
                continue

            if label == -1:  # 噪声点
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=self.NOISE_COLOR,
                    alpha=self.config.alpha,
                    s=self.config.marker_size // 2,
                    label="Noise",
                    marker="x",
                )
            else:  # 聚类点
                if label < len(discovery_result.cluster_types):
                    cluster_name = discovery_result.cluster_types[label].type_id
                    cluster_color = colors[label] if label < len(colors) else self.ELEGANT_COLORS[0]

                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        c=cluster_color,
                        alpha=self.config.alpha,
                        s=self.config.marker_size,
                        label=f"{cluster_name} (n={mask.sum()})",
                        edgecolors="black",
                        linewidth=0.5,
                    )

        # 设置图表属性
        ax.set_xlabel("Dimension 1", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Dimension 2", fontsize=self.config.label_fontsize)
        ax.set_title("Cluster Distribution", fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=0.3)

        # 添加图例
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=self.config.legend_fontsize,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
        )
        legend.set_title("Clusters", prop={"size": self.config.legend_fontsize, "weight": "bold"})

        # 添加方差信息
        if variance_info and "PC" in variance_info:
            ax.text(
                0.02, 0.98, variance_info, transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

    def _plot_cluster_statistics(self, ax: plt.Axes, discovery_result: ClusteringInfo) -> None:
        """Plot cluster statistical features"""
        try:
            content_types = discovery_result.cluster_types

            if not content_types:
                ax.text(
                    0.5, 0.5, "No clustering results", ha="center", va="center", transform=ax.transAxes, fontsize=self.config.label_fontsize
                )
                ax.set_title(f"Cluster Statistics of {discovery_result.clustering_columns}", fontsize=self.config.title_fontsize)
                return

            # Extract statistical data
            cluster_names = [ct.type_id for ct in content_types]
            cluster_sizes = [ct.cluster_size for ct in content_types]
            coverage_rates = [ct.coverage_rate * 100 for ct in content_types]  # Convert to percentage

            # Create combination chart: bar chart + line chart
            ax2 = ax.twinx()

            # Plot cluster size bar chart
            x = np.arange(len(cluster_names))
            bars = ax.bar(x, cluster_sizes, alpha=0.7, color="steelblue", label="Cluster Size")

            # Plot coverage rate line chart
            line = ax2.plot(x, coverage_rates, color="orange", marker="o", linewidth=2, markersize=6, label="Coverage Rate (%)")

            # Add value labels
            max_size = max(cluster_sizes) if cluster_sizes else 1
            max_coverage = max(coverage_rates) if coverage_rates else 1

            for i, (bar, size, coverage) in enumerate(zip(bars, cluster_sizes, coverage_rates)):
                # Bar chart labels
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + max_size * 0.01, str(size), ha="center", va="bottom", fontsize=9
                )
                # Line chart labels
                ax2.text(i, coverage + max_coverage * 0.02, f"{coverage:.1f}%", ha="center", va="bottom", fontsize=9, color="orange")

            # Set chart properties
            ax.set_ylabel("Sample Count", fontsize=self.config.label_fontsize, color="steelblue")
            ax2.set_ylabel("Coverage Rate (%)", fontsize=self.config.label_fontsize, color="orange")

            # Set x-axis labels
            ax.set_xticks(x)
            ax.set_xticklabels([name.replace("_", "\n") for name in cluster_names], fontsize=9, rotation=45, ha="right")

            # Set colors
            ax.tick_params(axis="y", labelcolor="steelblue")
            ax2.tick_params(axis="y", labelcolor="orange")

            # Add grid
            ax.grid(True, alpha=0.3, axis="y")

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=self.config.legend_fontsize)

            # Set title
            avg_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)
            total_coverage = sum(coverage_rates)

            ax.set_title(
                f"Cluster Statistics\n" f"Avg Size: {avg_size:.1f}±{std_size:.1f}, Total Coverage: {total_coverage:.1f}%",
                fontsize=self.config.title_fontsize,
                pad=20,
            )

            # Add statistics info box
            stats_text = f"Clusters: {len(content_types)}\n"
            stats_text += f"Max Cluster: {max(cluster_sizes)}\n"
            stats_text += f"Min Cluster: {min(cluster_sizes)}\n"
            stats_text += f"Balance: {1 - (std_size/avg_size if avg_size > 0 else 0):.2f}"

            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
                verticalalignment="top",
            )

        except Exception as e:
            self.logger.error(f"Failed to plot cluster statistics: {e}")
            ax.text(
                0.5, 0.5, f"Plot failed: {str(e)}", ha="center", va="center", transform=ax.transAxes, fontsize=self.config.label_fontsize
            )
            ax.set_title("Cluster Statistics (Error)", fontsize=self.config.title_fontsize)

    def _save_figure(self, fig: plt.Figure, save_path: Path) -> None:
        """Save figure to specified path"""
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure save_path is a Path object
            save_path = save_path.with_suffix(f".{self.config.save_format}")

            fig.savefig(
                save_path,
                format=self.config.save_format,
                dpi=self.config.dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

            self.logger.info(f"Clustering visualization saved: {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to save figure: {e}")
            raise RuntimeError(f"Cannot save figure to {save_path}: {e}") from e
        finally:
            plt.close(fig)

    def create_elegant_thumbnail(
        self,
        discovery_result: ClusteringInfo,
        size: Tuple[int, int] = (400, 400),
        style: str = "elegant",  # elegant, minimal, artistic
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """创建优美的聚类结果缩略图

        Args:
            discovery_result: 聚类结果
            size: 图片尺寸 (宽, 高)
            style: 风格 ('elegant', 'minimal', 'artistic')
            save_path: 保存路径

        Returns:
            matplotlib.Figure: 缩略图
        """
        # 准备数据
        plot_data = self._prepare_plot_data(discovery_result.clustering_data)
        if plot_data is None or plot_data.empty:
            return self._create_empty_thumbnail(size)

        # 降维处理
        coords = self._prepare_2d_coords(plot_data)
        if coords is None:
            return self._create_empty_thumbnail(size)

        # 创建图形
        fig_size = (size[0] / 100, size[1] / 100)  # 转换为英寸
        fig, ax = plt.subplots(figsize=fig_size, dpi=100)

        # 移除所有装饰
        self._remove_all_decorations(ax)

        # 根据风格绘制
        if style == "elegant":
            self._plot_elegant_style(ax, coords, discovery_result)
        elif style == "minimal":
            self._plot_minimal_style(ax, coords, discovery_result)
        elif style == "artistic":
            self._plot_artistic_style(ax, coords, discovery_result)
        else:
            self._plot_elegant_style(ax, coords, discovery_result)

        # 调整布局
        plt.tight_layout(pad=0)

        # 保存图片
        if save_path:
            self._save_thumbnail(fig, save_path, transparent=True)

        return fig

    def _prepare_2d_coords(self, plot_data: pd.DataFrame) -> Optional[np.ndarray]:
        """准备2D坐标数据"""
        try:
            n_dimensions = plot_data.shape[1]

            if n_dimensions == 1:
                # 1维数据转换为2维（添加随机y坐标）
                x = plot_data.iloc[:, 0].values
                y = np.random.normal(0, 0.1, len(x))  # 添加少量随机噪声
                coords = np.column_stack([x, y])
            elif n_dimensions == 2:
                coords = plot_data.values
            else:
                # 使用PCA降维
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(plot_data)

            return coords

        except Exception as e:
            self.logger.error(f"Failed to prepare 2D coordinates: {e}")
            return None

    def _remove_all_decorations(self, ax: plt.Axes):
        """移除所有装饰元素"""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)

    def _plot_elegant_style(self, ax: plt.Axes, coords: np.ndarray, discovery_result: ClusteringInfo):
        """优雅风格绘制"""
        # 创建标签数组
        labels = self._create_labels_array(coords, discovery_result)

        # 绘制聚类点
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            if not np.any(mask):
                continue

            if label == -1:  # 噪声点
                ax.scatter(
                    coords[mask, 0], coords[mask, 1], c=self.ELEGANT_NOISE_COLOR, s=15, alpha=0.4, edgecolors="none", zorder=1  # 较小的点
                )
            else:  # 聚类点
                color = self.ELEGANT_COLORS[label % len(self.ELEGANT_COLORS)]

                # 主要散点
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=25, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)

                # 添加光晕效果
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=50, alpha=0.15, edgecolors="none", zorder=2)

        # 设置等比例和边距
        self._set_equal_aspect_with_margin(ax, coords)

    def _plot_minimal_style(self, ax: plt.Axes, coords: np.ndarray, discovery_result: ClusteringInfo):
        """极简风格绘制"""
        labels = self._create_labels_array(coords, discovery_result)
        unique_labels = np.unique(labels)

        # 使用更简洁的配色
        simple_colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

        for label in unique_labels:
            mask = labels == label
            if not np.any(mask):
                continue

            if label == -1:
                continue  # 跳过噪声点，更简洁
            else:
                color = simple_colors[label % len(simple_colors)]
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=12, alpha=0.9, edgecolors="none", zorder=1)  # 更小的点

        self._set_equal_aspect_with_margin(ax, coords)

    def _plot_artistic_style(self, ax: plt.Axes, coords: np.ndarray, discovery_result: ClusteringInfo):
        """艺术风格绘制"""
        labels = self._create_labels_array(coords, discovery_result)
        unique_labels = np.unique(labels)

        # 渐变背景
        self._add_gradient_background(ax, coords)

        for label in unique_labels:
            mask = labels == label
            if not np.any(mask):
                continue

            if label == -1:
                ax.scatter(coords[mask, 0], coords[mask, 1], c="white", s=8, alpha=0.6, edgecolors="gray", linewidths=0.3, zorder=2)
            else:
                color = self.ELEGANT_COLORS[label % len(self.ELEGANT_COLORS)]

                # 多层叠加效果
                for size, alpha in [(40, 0.1), (25, 0.3), (15, 0.8)]:
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=size, alpha=alpha, edgecolors="none", zorder=3)

        self._set_equal_aspect_with_margin(ax, coords)

    def _create_labels_array(self, coords: np.ndarray, discovery_result: ClusteringInfo) -> np.ndarray:
        """创建标签数组"""
        labels = np.full(len(coords), -1)

        for i, content_type in enumerate(discovery_result.cluster_types):
            for idx in content_type.sample_indices:
                if 0 <= idx < len(labels):
                    labels[idx] = i

        return labels

    def _set_equal_aspect_with_margin(self, ax: plt.Axes, coords: np.ndarray, margin: float = 0.05):
        """设置等比例和边距"""
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        # 添加边距
        x_margin = x_range * margin
        y_margin = y_range * margin

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect("equal", adjustable="box")

    def _add_gradient_background(self, ax: plt.Axes, coords: np.ndarray):
        """添加渐变背景"""
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        # 创建渐变
        gradient = np.linspace(0, 1, 256).reshape(256, -1)
        gradient = np.vstack((gradient, gradient))

        ax.imshow(gradient, extent=[x_min, x_max, y_min, y_max], aspect="auto", cmap="plasma", alpha=0.1, zorder=0)

    def _create_empty_thumbnail(self, size: Tuple[int, int]) -> plt.Figure:
        """创建空缩略图"""
        fig_size = (size[0] / 100, size[1] / 100)
        fig, ax = plt.subplots(figsize=fig_size, dpi=100)

        self._remove_all_decorations(ax)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=16, color="gray", alpha=0.5)

        return fig

    def _save_thumbnail(self, fig: plt.Figure, save_path: Path, transparent: bool = True):
        """保存缩略图"""
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=transparent,
                facecolor="none" if transparent else "white",
                edgecolor="none",
            )

            self.logger.info(f"Elegant thumbnail saved: {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to save thumbnail: {e}")
            raise
        finally:
            plt.close(fig)
