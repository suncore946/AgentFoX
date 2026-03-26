from concurrent.futures import as_completed
from pathlib import Path
import pickle
import sqlite3
import pandas as pd
import numpy as np
import hdbscan
from typing import List, Dict, Any, Optional, Tuple
from scipy.signal import find_peaks


from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from ...utils.logger import get_logger
from ...configs.config_dataclass import ClusteringConfig
from .clustering_ability_assessor import ClusteringAbilityAssessor
from .clustering_dataclass import Cluster, ClusteringInfo, ClusteringStats, ClusteringContext
from .clustering_pca import pca_reduce_dimensions


class OptimalKFinder:
    """优化的K值选择器"""

    # 数据量阈值：超过100万条使用MiniBatchKMeans
    LARGE_DATA_THRESHOLD = 1000000

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._quick_kmeans_template = {"random_state": random_state, "n_init": 3, "max_iter": 50, "tol": 1e-2}

    def _get_kmeans_class_and_params(self, n_samples: int) -> Tuple[type, Dict]:
        """根据数据量选择KMeans算法和参数"""
        if n_samples > self.LARGE_DATA_THRESHOLD:
            return MiniBatchKMeans, {**self._quick_kmeans_template, "batch_size": min(4096, max(100, n_samples // 100)), "max_iter": 30}
        else:
            return KMeans, {**self._quick_kmeans_template, "algorithm": "lloyd"}

    def sample_data_for_kmeans(self, pca_data, sample_size=None):
        # 采样十分之一，最大不超过1000000
        if sample_size is None:
            sample_size = min(max(1, len(pca_data) // 10), 1000000)
        if len(pca_data) > sample_size:
            idx = np.random.RandomState(self.random_state).choice(len(pca_data), sample_size, replace=False)
            return pca_data[idx]
        return pca_data

    def find_optimal_k(self, pca_data: np.ndarray, min_k: int, max_k: int) -> Tuple[int, Dict]:
        sampled_data = self.sample_data_for_kmeans(pca_data)
        k_range = np.arange(min_k, max_k + 1)
        inertias = self._compute_inertias_efficient(sampled_data, k_range)
        optimal_k = self._find_elbow_point(inertias, k_range, min_k)

        return optimal_k, {
            "method": "elbow_optimized",
            "k_range": k_range.tolist(),
            "inertias": inertias.tolist(),
            "optimal_k": optimal_k,
            "algorithm_used": "MiniBatchKMeans" if len(sampled_data) > self.LARGE_DATA_THRESHOLD else "KMeans",
        }

    def _compute_inertias_efficient(self, data: np.ndarray, k_range: np.ndarray) -> np.ndarray:
        inertias = np.zeros(len(k_range))
        data_mean = np.mean(data, axis=0)

        # 根据数据量选择算法
        kmeans_class, base_params = self._get_kmeans_class_and_params(len(data))

        for i, k in enumerate(k_range):
            try:
                kmeans = kmeans_class(n_clusters=k, **base_params)
                kmeans.fit(data)
                inertias[i] = kmeans.inertia_
            except Exception:
                inertias[i] = self._estimate_inertia_heuristic(data, k, data_mean)

        return inertias

    def _estimate_inertia_heuristic(self, data: np.ndarray, k: int, data_mean: np.ndarray) -> float:
        total_variance = np.sum(np.var(data, axis=0))
        return total_variance / k

    def _find_elbow_point(self, inertias: np.ndarray, k_range: np.ndarray, min_k: int) -> int:
        if len(inertias) < 3:
            return min_k

        try:
            second_derivative = np.diff(inertias, 2)
            peaks, _ = find_peaks(-second_derivative)

            if len(peaks) > 0:
                peak_values = -second_derivative[peaks]
                best_peak_idx = peaks[np.argmax(peak_values)]
                return int(k_range[best_peak_idx + 2])
            else:
                return int(k_range[np.argmax(-second_derivative) + 2])

        except Exception:
            return int(k_range[len(k_range) // 2])


class ClusteringAlgorithm:
    """内存和性能优化的聚类算法管理器 - 支持多种聚类方法"""

    CORRELATION_THRESHOLD = 0.9
    DEFAULT_RANDOM_STATE = 42
    LARGE_DATA_THRESHOLD = 1000000  # 100万条数据的阈值
    MEDIUM_DATA_THRESHOLD = 50000  # 5万条数据的阈值

    KMEANS_FINAL_NINT = 20
    KMEANS_FINAL_MAX_ITER = 300

    # 扩展的聚类方法列表
    CLUSTERING_METHODS = [
        ("kmeans_pca", "_cluster_kmeans"),
        ("hdbscan_pca", "_cluster_hdbscan"),
        ("gmm_pca", "_cluster_gmm"),
        ("agglomerative_pca", "_cluster_agglomerative"),
        ("dbscan_pca", "_cluster_dbscan"),
        # ("spectral_pca", "_cluster_spectral"),
    ]

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.random_state = getattr(config, "random_state", self.DEFAULT_RANDOM_STATE)
        np.random.seed(self.random_state)

        # 根据数据特点筛选适用的聚类方法
        self.clustering_methods = self._get_applicable_methods()

        self.clustering_ability = ClusteringAbilityAssessor()
        self.scoring_weights = self.config.scoring_weights
        self.scoring_thresholds = self.config.scoring_thresholds

        self._k_finder = OptimalKFinder(self.random_state)
        self.base_save_path = Path(self.config.save_dir)
        self.base_save_path.mkdir(parents=True, exist_ok=True)

    def _get_applicable_methods(self) -> List[Tuple[str, callable]]:
        """根据配置筛选适用的聚类方法"""
        all_methods = [(name, getattr(self, method_name)) for name, method_name in self.CLUSTERING_METHODS]

        # 可以根据配置或其他条件筛选方法
        # 例如，排除计算量过大的方法
        return all_methods

    def _get_kmeans_class_and_params(self, n_samples: int) -> Tuple[type, Dict]:
        """根据数据量选择KMeans算法和参数"""
        base_params = {
            "random_state": self.random_state,
            "n_init": self.KMEANS_FINAL_NINT,
            "max_iter": self.KMEANS_FINAL_MAX_ITER,
        }

        if n_samples > self.LARGE_DATA_THRESHOLD:
            self.logger.info(f"数据量较大 ({n_samples:,} 条)，使用MiniBatchKMeans")
            return MiniBatchKMeans, {
                **base_params,
                "batch_size": min(4096, max(100, n_samples // 100)),
                "verbose": 1,
            }
        else:
            self.logger.info(f"数据量适中 ({n_samples:,} 条)，使用标准KMeans")
            return KMeans, {
                **base_params,
                "algorithm": "lloyd",
            }

    def run(self, model: Optional[object], processed_data: pd.DataFrame, aggregation_columns, clustering_desc) -> ClusteringInfo:
        """执行聚类方法并返回可视化数据"""
        if processed_data.empty:
            self.logger.warning("输入数据为空，无法执行聚类")
            return [], None

        pca_data = pca_reduce_dimensions(processed_data)[0]

        # 创建共享上下文
        clustering_ability = self.clustering_ability.assess(pca_data)
        context = ClusteringContext.create_optimized(pca_data)

        results: List[ClusteringInfo] = []

        if model:
            # 使用预训练模型
            cluster_labels = model.fit_predict(processed_data)
            result = self.process_clustering_result(context, cluster_labels, "KMeans_PCA", model)
            results.append(result)
        else:
            # 尝试多种聚类方法
            for method_name, method_func in self.clustering_methods:
                try:
                    self.logger.info(f"开始执行聚类方法: {method_name}")
                    result = method_func(context)
                    if result is not None:
                        results.append(result)
                        self.logger.info(f"聚类方法 {method_name} 完成，得分: {result.quality_metrics.get('score', 0):.3f}")
                    else:
                        self.logger.warning(f"聚类方法 {method_name} 返回空结果")
                except Exception as e:
                    self.logger.warning(f"聚类方法 {method_name} 失败: {e}")
                    continue

        if not results:
            self.logger.error("所有聚类方法都失败了")
            return None

        # 选择最佳结果
        best_result = max(results, key=lambda r: r.quality_metrics.get("score", 0))
        best_result.clustering_ability = clustering_ability
        best_result.clustering_data = pca_data
        best_result.clustering_columns = aggregation_columns
        best_result.clustering_desc = clustering_desc
        self.logger.info(f"选择最佳聚类方法: {best_result.name}，得分: {best_result.quality_metrics.get('score', 0):.3f}")
        return best_result

    def _cluster_kmeans(self, context: ClusteringContext, **kwargs):
        """KMeans聚类 - 根据数据量自动选择算法"""
        try:
            n_samples = context.pca_data.shape[0]
            min_k = max(2, context.min_cluster)
            max_k = min(16, max(3, int(np.sqrt(n_samples // 2))))

            if min_k >= max_k:
                optimal_k = min_k
                optimization_info = {"method": "fallback", "reason": "insufficient_samples"}
            else:
                optimal_k, optimization_info = self._k_finder.find_optimal_k(context.pca_data, min_k, max_k)

            # 根据数据量选择KMeans算法
            kmeans_class, kmeans_params = self._get_kmeans_class_and_params(n_samples)

            kmeans = kmeans_class(n_clusters=optimal_k, **kmeans_params)
            cluster_labels = kmeans.fit_predict(context.pca_data)

            extra_info = {
                "optimal_k": optimal_k,
                "optimization_info": optimization_info,
                "inertia": float(kmeans.inertia_),
                "algorithm_used": kmeans_class.__name__,
                "n_samples": n_samples,
            }

            if hasattr(kmeans, "n_iter_"):
                extra_info["n_iter"] = int(kmeans.n_iter_)

            return self.process_clustering_result(context, cluster_labels, "KMeans_PCA", kmeans, extra_info=extra_info)

        except Exception as e:
            self.logger.error(f"KMeans聚类失败: {e}")
            return None

    def _cluster_hdbscan(self, context: ClusteringContext, **kwargs):
        """HDBSCAN聚类 - 基于密度的聚类"""
        try:
            n_samples = context.pca_data.shape[0]

            # 根据数据量调整参数
            if n_samples > self.LARGE_DATA_THRESHOLD:
                min_cluster_size = max(100, n_samples // 1000)
                min_samples = max(10, min_cluster_size // 10)
            elif n_samples > self.MEDIUM_DATA_THRESHOLD:
                min_cluster_size = max(50, n_samples // 500)
                min_samples = max(5, min_cluster_size // 10)
            else:
                min_cluster_size = max(10, n_samples // 100)
                min_samples = max(3, min_cluster_size // 5)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.0,
                metric="euclidean",
                cluster_selection_method="eom",
            )

            cluster_labels = clusterer.fit_predict(context.pca_data)

            extra_info = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters_found": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "noise_points": sum(cluster_labels == -1),
            }

            return self.process_clustering_result(context, cluster_labels, "HDBSCAN_PCA", clusterer, extra_info=extra_info)

        except Exception as e:
            self.logger.error(f"HDBSCAN聚类失败: {e}")
            return None

    def _cluster_gmm(self, context: ClusteringContext, **kwargs):
        """高斯混合模型聚类"""
        try:
            n_samples = context.pca_data.shape[0]
            min_k = max(2, context.min_cluster)
            max_k = min(16, max(3, int(np.sqrt(n_samples // 2))))

            if min_k >= max_k:
                optimal_k = min_k
            else:
                # 使用BIC/AIC选择最优组件数
                optimal_k = self._find_optimal_gmm_components(context.pca_data, min_k, max_k)

            gmm = GaussianMixture(n_components=optimal_k, random_state=self.random_state, covariance_type="full", max_iter=100, n_init=3)

            gmm.fit(context.pca_data)
            cluster_labels = gmm.predict(context.pca_data)

            extra_info = {
                "optimal_k": optimal_k,
                "aic": float(gmm.aic(context.pca_data)),
                "bic": float(gmm.bic(context.pca_data)),
                "converged": bool(gmm.converged_),
            }

            return self.process_clustering_result(context, cluster_labels, "GMM_PCA", gmm, extra_info=extra_info)

        except Exception as e:
            self.logger.error(f"GMM聚类失败: {e}")
            return None

    def _find_optimal_gmm_components(self, data: np.ndarray, min_k: int, max_k: int) -> int:
        """使用BIC选择最优的GMM组件数"""
        k_range = range(min_k, max_k + 1)
        bics = []

        # 对大数据进行采样
        if len(data) > 10000:
            sample_idx = np.random.RandomState(self.random_state).choice(len(data), 10000, replace=False)
            sample_data = data[sample_idx]
        else:
            sample_data = data

        for k in k_range:
            try:
                gmm = GaussianMixture(n_components=k, random_state=self.random_state, covariance_type="full", max_iter=50)
                gmm.fit(sample_data)
                bics.append(gmm.bic(sample_data))
            except:
                bics.append(float("inf"))

        optimal_idx = np.argmin(bics)
        return k_range[optimal_idx]

    def _cluster_agglomerative(self, context: ClusteringContext, **kwargs):
        """层次聚类 - 适合中小规模数据"""
        try:
            n_samples = context.pca_data.shape[0]

            # 层次聚类不适合大数据
            if n_samples > self.MEDIUM_DATA_THRESHOLD:
                self.logger.info(f"数据量过大 ({n_samples})，跳过层次聚类")
                return None

            min_k = max(2, context.min_cluster)
            max_k = min(16, max(3, int(np.sqrt(n_samples // 2))))
            optimal_k = (min_k + max_k) // 2  # 简单取中值

            clusterer = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")

            cluster_labels = clusterer.fit_predict(context.pca_data)

            extra_info = {"optimal_k": optimal_k, "linkage": "ward"}

            return self.process_clustering_result(context, cluster_labels, "Agglomerative_PCA", clusterer, extra_info=extra_info)

        except Exception as e:
            self.logger.error(f"层次聚类失败: {e}")
            return None

    def _cluster_dbscan(self, context: ClusteringContext, **kwargs):
        """DBSCAN聚类"""
        try:
            n_samples = context.pca_data.shape[0]

            # 估计eps参数
            eps = self._estimate_dbscan_eps(context.pca_data)
            min_samples = max(3, int(np.log(n_samples)))

            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")

            cluster_labels = clusterer.fit_predict(context.pca_data)

            extra_info = {
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters_found": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "noise_points": sum(cluster_labels == -1),
            }

            return self.process_clustering_result(context, cluster_labels, "DBSCAN_PCA", clusterer, extra_info=extra_info)

        except Exception as e:
            self.logger.error(f"DBSCAN聚类失败: {e}")
            return None

    def _estimate_dbscan_eps(self, data: np.ndarray) -> float:
        """估计DBSCAN的eps参数"""
        try:
            # 使用k距离方法估计eps
            k = max(3, int(np.log(len(data))))

            # 对大数据进行采样
            if len(data) > 10000:
                sample_idx = np.random.RandomState(self.random_state).choice(len(data), 10000, replace=False)
                sample_data = data[sample_idx]
            else:
                sample_data = data

            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors.fit(sample_data)
            distances, _ = neighbors.kneighbors(sample_data)

            # 取k-th距离的中位数作为eps的估计
            k_distances = np.sort(distances[:, -1])
            eps = np.median(k_distances)

            return float(eps)
        except:
            # 备用方法：使用数据标准差的一定比例
            return float(np.std(data) * 0.5)

    def _cluster_spectral(self, context: ClusteringContext, **kwargs):
        """谱聚类 - 适合中小规模数据"""
        try:
            n_samples = context.pca_data.shape[0]

            # 谱聚类不适合大数据
            if n_samples > self.MEDIUM_DATA_THRESHOLD:
                self.logger.info(f"数据量过大 ({n_samples})，跳过谱聚类")
                return None

            min_k = max(2, context.min_cluster)
            max_k = min(16, max(3, int(np.sqrt(n_samples // 2))))
            optimal_k = (min_k + max_k) // 2  # 简单取中值

            clusterer = SpectralClustering(n_clusters=optimal_k, random_state=self.random_state, affinity="rbf", gamma=1.0)

            cluster_labels = clusterer.fit_predict(context.pca_data)

            extra_info = {"optimal_k": optimal_k, "affinity": "rbf"}

            return self.process_clustering_result(context, cluster_labels, "Spectral_PCA", clusterer, extra_info=extra_info)

        except Exception as e:
            self.logger.error(f"谱聚类失败: {e}")
            return None

    def save_cluster_model(self, cluster_model, cluster_column_name: List[str]) -> str:
        """保存聚类模型到文件"""
        columns_name = "&".join(cluster_column_name) if isinstance(cluster_column_name, list) else str(cluster_column_name)
        save_path = self.base_save_path / f"{columns_name}.pkl"

        with open(save_path, "wb") as f:
            pickle.dump(cluster_model, f)

        self.logger.info(f"聚类模型已保存到: {save_path}")

    def load_cluster_model(self, cluster_column_name) -> Dict[str, Any]:
        """从文件加载聚类结果"""
        columns_name = "&".join(cluster_column_name) if isinstance(cluster_column_name, list) else str(cluster_column_name) + ".pkl"
        save_path = self.base_save_path / columns_name
        if not save_path.exists():
            return None
        self.logger.info(f"加载聚类模型: {save_path}")
        with open(save_path, "rb") as f:
            model = pickle.load(f)
        return model

    def pred_cluster(self, src_data, processed_data: pd.DataFrame, target_column):
        """预测聚类"""
        pca_data = pca_reduce_dimensions(processed_data)
        context = ClusteringContext.create_optimized(src_data)
        model = self.load_cluster_model(target_column)
        cluster_labels = model.fit_predict(pca_data)
        result = self.process_clustering_result(context, cluster_labels, "KMeans_PCA", model)
        return result, context

    def process_clustering_result(
        self,
        context: ClusteringContext,
        cluster_labels: np.ndarray,
        method_name: str,
        cluster,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """处理聚类结果"""
        data_len = len(context.pca_data)
        min_samples = self.config.min_samples

        stats = ClusteringStats.from_labels(cluster_labels, data_len)
        if stats.n_clusters <= 0:
            raise ValueError("未能识别任何聚类")

        discovery_stats = {
            "method_name": method_name,
            "n_clusters": stats.n_clusters,
            "n_noise": stats.n_noise,
            "noise_ratio": stats.noise_ratio,
            "cluster_sizes": {},
        }

        if extra_info:
            discovery_stats.update(extra_info)

        cluster_types = []
        warning_messages = []

        try:
            cluster_centers_info = self._compute_cluster_centers(cluster, context, stats.valid_labels)
        except Exception:
            cluster_centers_info = {}

        valid_data = context.pca_data[stats.cluster_mask]
        valid_labels = cluster_labels[stats.cluster_mask]

        for label in stats.valid_labels:
            label_mask = valid_labels == label
            cluster_data = valid_data[label_mask]
            cluster_size = len(cluster_data)

            discovery_stats["cluster_sizes"][f"cluster_{label}"] = cluster_size

            if cluster_size < min_samples:
                warning_messages.append(f"簇 {label} 样本数不足: {cluster_size}, 建议至少 {min_samples}")

            sample_indices = np.where(cluster_labels == label)[0].tolist()
            cluster_info = cluster_centers_info.get(label, {})

            cluster_types.append(
                Cluster(
                    type_id=f"cluster_{label}",
                    name=f"聚类_{label}",
                    sample_indices=sample_indices,
                    coverage_rate=cluster_size / data_len,
                    cluster_size=cluster_size,
                    cluster_center=cluster_info.get("center"),
                    cluster_radius=cluster_info.get("avg_radius", None),
                )
            )

        quality_metrics = self._compute_quality_metrics(context, cluster_labels, stats, cluster_types)
        return ClusteringInfo(
            **{
                "model": cluster,
                "name": method_name,
                "description": f"聚类方法: {method_name}",
                "cluster_types": cluster_types,
                "cluster_labels": cluster_labels,
                "discovery_stats": discovery_stats,
                "quality_metrics": quality_metrics,
                "warning_messages": warning_messages,
            }
        )

    def _compute_cluster_centers(self, cluster, context: ClusteringContext, valid_labels: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """计算聚类中心和半径"""
        centers_and_radius = {}

        if hasattr(cluster, "cluster_centers_"):
            # KMeans/GMM情况
            cluster_labels = cluster.labels_
            for label in valid_labels:
                if label < len(cluster.cluster_centers_):
                    center = cluster.cluster_centers_[label]

                    mask = cluster_labels == label
                    cluster_points = context.pca_data[mask]
                    distances = np.linalg.norm(cluster_points - center, axis=1)

                    avg_radius = np.mean(distances) if len(distances) > 0 else 0
                    max_radius = np.max(distances) if len(distances) > 0 else 0

                    centers_and_radius[label] = {
                        "center": center,
                        "avg_radius": float(avg_radius),
                        "max_radius": float(max_radius),
                        "std_radius": float(np.std(distances)) if len(distances) > 1 else 0,
                    }
        else:
            # 其他算法：手动计算中心
            cluster_labels = getattr(cluster, "labels_", None)
            if cluster_labels is not None:
                for label in valid_labels:
                    mask = cluster_labels == label
                    if mask.sum() > 0:
                        cluster_points = context.pca_data[mask]
                        center = np.mean(cluster_points, axis=0)

                        distances = np.linalg.norm(cluster_points - center, axis=1)
                        avg_radius = np.mean(distances) if len(distances) > 0 else 0
                        max_radius = np.max(distances) if len(distances) > 0 else 0

                        centers_and_radius[label] = {
                            "center": center,
                            "avg_radius": float(avg_radius),
                            "max_radius": float(max_radius),
                            "std_radius": float(np.std(distances)) if len(distances) > 1 else 0,
                        }

        return centers_and_radius

    def _compute_quality_metrics(
        self,
        context: ClusteringContext,
        cluster_labels: np.ndarray,
        stats: ClusteringStats,
        content_types: List[Any],
    ) -> Dict[str, float]:
        """批量计算聚类质量指标"""
        quality_metrics = {}

        if stats.n_clusters > 1 and stats.noise_ratio < 0.8:
            try:
                valid_data = context.pca_data[stats.cluster_mask]
                valid_labels = cluster_labels[stats.cluster_mask]

                silhouette = silhouette_score(valid_data, valid_labels)
                quality_metrics["silhouette_score"] = float(silhouette)

                ch_score = calinski_harabasz_score(valid_data, valid_labels)
                quality_metrics["calinski_harabasz_score"] = float(ch_score)

                db_score = davies_bouldin_score(valid_data, valid_labels)
                quality_metrics["davies_bouldin_score"] = float(db_score)
            except Exception as e:
                self.logger.warning(f"计算质量指标失败: {e}")

        if content_types:
            sizes = np.array([ct.cluster_size for ct in content_types])
            coverages = np.array([ct.coverage_rate for ct in content_types])
            quality_metrics.update(
                {
                    "avg_cluster_size": float(np.mean(sizes)),
                    "std_cluster_size": float(np.std(sizes)),
                    "total_coverage": float(np.sum(coverages)),
                }
            )

        # 计算综合评分
        n_clusters = len(content_types)
        silhouette = quality_metrics.get("silhouette_score", 0)
        total_coverage = quality_metrics.get("total_coverage", 0)
        score = np.array(
            [
                self.scoring_weights["cluster_count"] * min(1.0, n_clusters / self.scoring_thresholds["max_clusters_for_full_score"]),
                (
                    self.scoring_weights["silhouette"] * min(1.0, silhouette / self.scoring_thresholds["silhouette_full_score"])
                    if silhouette > 0
                    else 0
                ),
                self.scoring_weights["coverage"] * min(1.0, total_coverage / self.scoring_thresholds["coverage_full_score"]),
                self.scoring_weights["noise_ratio"] * max(0, 1.0 - stats.noise_ratio / self.scoring_thresholds["max_noise_ratio"]),
            ]
        )
        quality_metrics["score"] = float(sum(score))

        return quality_metrics
