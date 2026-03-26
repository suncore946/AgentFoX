from concurrent.futures import as_completed
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Cluster:
    """内容类型定义"""

    type_id: str  # 类型ID
    name: str  # 类型名称
    sample_indices: List[int]  # 样本索引
    coverage_rate: float  # 覆盖率

    # 聚类信息（用于聚类方法）
    cluster_size: int  # 样本数量
    cluster_center: Optional[np.ndarray] = None
    cluster_radius: Optional[float] = None

    def __post_init__(self):
        """计算派生属性"""
        if self.cluster_size == 0 and self.sample_indices:
            self.cluster_size = len(self.sample_indices)


@dataclass
class ClusteringInfo:
    model: Any  # 聚类模型
    name: str  # 聚类算法名称
    description: str  # 聚类算法描述
    cluster_types: List[Cluster]  # 聚类类型列表
    cluster_labels: np.ndarray  # 聚类标签
    discovery_stats: Dict[str, Any]  # 发现统计信息
    quality_metrics: Dict[str, float]  # 质量指标
    warning_messages: List[str]  # 警告信息列表
    clustering_columns: str = ""  # 聚类名
    clustering_desc: Dict[str, str] = None # 聚类描述
    clustering_ability: Dict[str, Any] = None  # 聚类能力评估结果
    clustering_data: Dict[str, Any] = None  # 聚类数据相关信息


@dataclass
class ClusteringStats:
    """优化的聚类统计信息缓存类"""

    unique_labels: np.ndarray
    label_counts: np.ndarray
    cluster_mask: np.ndarray
    noise_mask: np.ndarray
    n_clusters: int
    n_noise: int
    noise_ratio: float
    valid_labels: np.ndarray
    label_to_count: Dict[int, int]

    @classmethod
    def from_labels(cls, cluster_labels: np.ndarray, total_samples: int) -> "ClusteringStats":
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_mask = cluster_labels != -1
        noise_mask = ~cluster_mask
        n_noise = counts[unique_labels == -1][0] if -1 in unique_labels else 0
        valid_labels = unique_labels[unique_labels != -1]
        n_clusters = len(valid_labels)
        noise_ratio = n_noise / total_samples if total_samples > 0 else 0
        label_to_count = dict(zip(unique_labels, counts))

        return cls(
            unique_labels=unique_labels,
            label_counts=counts,
            cluster_mask=cluster_mask,
            noise_mask=noise_mask,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_ratio=noise_ratio,
            valid_labels=valid_labels,
            label_to_count=label_to_count,
        )


@dataclass
class ClusteringContext:
    """内存优化的聚类上下文缓存"""

    pca_data: np.ndarray
    min_cluster: int = 2

    @classmethod
    def create_optimized(
        cls,
        pca_data: np.ndarray,
        min_cluster: int = 2,
    ) -> "ClusteringContext":
        return cls(
            pca_data=pca_data,
            min_cluster=min_cluster,
        )
