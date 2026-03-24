import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from ...utils.logger import get_logger


class ClusteringAbilityAssessor:
    """可聚性评估器

    专门用于评估数据的聚类可行性，包括Hopkins统计量、方差指标、特征相关性等指标
    """

    def __init__(self, random_state: int = 42):
        """初始化可聚性评估器

        Args:
            random_state: 随机种子
        """
        self.logger = get_logger(__name__)
        self.random_state = random_state
        np.random.seed(self.random_state)

        # 评估阈值配置
        self.thresholds = {
            "hopkins": {"highly_clusterable": 0.3, "moderately_clusterable": 0.5, "weakly_clusterable": 0.7},
            "variance_cv": {"high_variation": 0.5, "moderate_variation": 1.0, "low_variation": 2.0},
            "correlation": {"high_correlation": 0.7, "moderate_correlation": 0.4, "low_correlation": 0.2},
        }

        # 综合评估权重
        self.weights = {"hopkins": 0.5, "variance_cv": 0.3, "correlation": 0.2}

    def assess(self, cluster_data: np.ndarray) -> Dict[str, Any]:
        """评估数据的可聚性"""
        if not self._is_data_valid_for_assessment(cluster_data):
            return {"assessment": "insufficient_data", "confidence": 0.0, "details": {"error": "数据不足或无效，无法进行聚类可行性评估"}}

        try:
            # 计算各项指标
            cluster_ability = {
                "hopkins_statistic": self._calculate_hopkins_statistic(cluster_data),
                "variance_cv": self._calculate_variance_cv(cluster_data),
                "feature_correlation": self._calculate_feature_correlation(cluster_data),
                "density": self._estimate_data_density(cluster_data),
            }

            # 综合评估
            assessment_result = self._assess_overall_cluster_ability(cluster_ability)
            cluster_ability.update(assessment_result)
            cluster_ability["recommended_methods"] = self._recommend_clustering_methods(cluster_data, cluster_ability)

            return cluster_ability

        except Exception as e:
            self.logger.warning(f"可聚性评估失败: {e}")
            return {"assessment": "evaluation_failed", "confidence": 0.0, "details": {"error": f"评估过程出错: {str(e)}"}}

    def _is_data_valid_for_assessment(self, data: np.ndarray) -> bool:
        """判断数据是否适合进行可聚性评估"""
        if data.size == 0 or data.shape[0] < 5 or data.shape[1] < 1:
            return False

        # 检查是否有足够的非空数据
        non_null_mask = ~np.isnan(data).any(axis=1)
        return np.sum(non_null_mask) >= 5

    def _calculate_hopkins_statistic(self, data: np.ndarray, n_samples: int = 100) -> float:
        """计算Hopkins统计量"""
        try:
            # 移除缺失值
            mask = ~np.isnan(data).any(axis=1)
            clean_data = data[mask]

            if len(clean_data) < 10:
                return 0.5

            n_samples = min(n_samples, len(clean_data) // 2, 200)
            if n_samples <= 0:
                return 0.5

            # 标准化数据
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)

            # 计算Hopkins统计量
            nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(scaled_data)

            # 随机采样实际数据点
            sample_indices = np.random.choice(len(scaled_data), n_samples, replace=False)
            sample_data = scaled_data[sample_indices]

            distances, _ = nbrs.kneighbors(sample_data)
            real_distances = distances[:, 1]  # 排除自己

            # 生成随机点
            data_min, data_max = np.min(scaled_data, axis=0), np.max(scaled_data, axis=0)
            data_range = data_max - data_min
            zero_range_mask = data_range == 0
            if zero_range_mask.any():
                data_range[zero_range_mask] = 1.0
                data_max[zero_range_mask] = data_min[zero_range_mask] + 1.0

            random_points = np.random.uniform(data_min, data_max, (n_samples, len(data_min)))
            random_distances, _ = nbrs.kneighbors(random_points)
            random_nn_distances = random_distances[:, 0]

            # 计算Hopkins统计量
            u, w = np.sum(random_nn_distances), np.sum(real_distances)
            if (u + w) == 0:
                return 0.5

            return float(np.clip(u / (u + w), 0.0, 1.0))

        except Exception as e:
            self.logger.warning(f"Hopkins统计量计算失败: {e}")
            return 0.5

    def _calculate_variance_cv(self, data: np.ndarray) -> float:
        """计算方差变异系数"""
        try:
            # 计算每列的方差
            variances = np.nanvar(data, axis=0)
            valid_variances = variances[variances > 1e-10]

            if len(valid_variances) == 0:
                return 0.0

            mean_var, std_var = np.mean(valid_variances), np.std(valid_variances)
            return float(std_var / mean_var) if mean_var != 0 else 0.0

        except Exception as e:
            self.logger.warning(f"方差变异系数计算失败: {e}")
            return 0.0

    def _calculate_feature_correlation(self, data: np.ndarray) -> float:
        """计算特征间平均相关性"""
        if data.shape[1] < 2:
            return 0.0

        try:
            # 移除缺失值较多的行
            mask = ~np.isnan(data).any(axis=1)
            clean_data = data[mask]

            if len(clean_data) < 3:
                return 0.0

            # 移除常数列
            valid_cols = []
            for col_idx in range(clean_data.shape[1]):
                col_data = clean_data[:, col_idx]
                if len(np.unique(col_data)) > 1 and np.nanstd(col_data) > 1e-10:
                    valid_cols.append(col_idx)

            if len(valid_cols) < 2:
                return 0.0

            valid_data = clean_data[:, valid_cols]
            corr_matrix = np.corrcoef(valid_data.T)

            # 处理可能的NaN值
            corr_matrix = np.abs(corr_matrix)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # 提取上三角矩阵
            n = corr_matrix.shape[0]
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            correlations = corr_matrix[mask]

            return float(np.mean(correlations)) if len(correlations) > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"特征相关性计算失败: {e}")
            return 0.0

    def _estimate_data_density(self, data: np.ndarray) -> float:
        """估算数据密度"""
        try:
            # 移除缺失值
            mask = ~np.isnan(data).any(axis=1)
            clean_data = data[mask]

            if len(clean_data) < 2:
                return 0.0

            # 采样以提高效率
            sample_size = min(1000, len(clean_data))
            if sample_size < len(clean_data):
                sample_indices = np.random.choice(len(clean_data), sample_size, replace=False)
                sample_data = clean_data[sample_indices]
            else:
                sample_data = clean_data

            # 计算最近邻距离
            nbrs = NearestNeighbors(n_neighbors=min(5, len(sample_data)), metric="euclidean")
            nbrs.fit(sample_data)
            distances, _ = nbrs.kneighbors(sample_data)

            if distances.shape[1] <= 1:
                return 0.0

            avg_distance = np.mean(distances[:, 1:])
            n_dims = sample_data.shape[1]
            dimension_factor = 1.0 / (1.0 + 0.1 * n_dims)
            density = dimension_factor / (1.0 + avg_distance)

            return float(np.clip(density, 0.0, 1.0))

        except Exception as e:
            self.logger.warning(f"数据密度估算失败: {e}")
            return 0.0

    def _get_metric_score(self, metric: str, value: Union[float, None]) -> float:
        """根据指标值计算评分"""
        if value is None:
            return 0.0

        if metric == "hopkins":
            thresholds = self.thresholds["hopkins"]
            if value < thresholds["highly_clusterable"]:
                return 1.0
            elif value < thresholds["moderately_clusterable"]:
                return 0.7
            elif value < thresholds["weakly_clusterable"]:
                return 0.4
            else:
                return 0.1

        elif metric == "variance_cv":
            thresholds = self.thresholds["variance_cv"]
            if value > thresholds["low_variation"]:
                return 0.3
            elif value > thresholds["moderate_variation"]:
                return 0.6
            elif value > thresholds["high_variation"]:
                return 1.0
            else:
                return 0.8

        elif metric == "correlation":
            thresholds = self.thresholds["correlation"]
            if value > thresholds["high_correlation"]:
                return 0.3
            elif value > thresholds["moderate_correlation"]:
                return 1.0
            elif value > thresholds["low_correlation"]:
                return 0.8
            else:
                return 0.5

        return 0.0

    def _assess_overall_cluster_ability(self, cluster_ability: Dict[str, Any]) -> Dict[str, Any]:
        """综合评估可聚性"""
        # 计算各指标得分
        scores = {
            "hopkins": self._get_metric_score("hopkins", cluster_ability.get("hopkins_statistic")),
            "variance_cv": self._get_metric_score("variance_cv", cluster_ability.get("variance_cv")),
            "correlation": self._get_metric_score("correlation", cluster_ability.get("feature_correlation")),
        }

        # 计算加权总分
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        if not valid_scores:
            final_score = 0.0
            valid_weights = 0.0
        else:
            total_score = sum(self.weights[metric] * score for metric, score in valid_scores.items())
            valid_weights = sum(self.weights[metric] for metric in valid_scores.keys())
            final_score = total_score / valid_weights

        # 确定评估结果
        if final_score >= 0.8:
            assessment = "highly_clusterable"
        elif final_score >= 0.6:
            assessment = "moderately_clusterable"
        elif final_score >= 0.4:
            assessment = "weakly_clusterable"
        else:
            assessment = "not_clusterable"

        return {
            "assessment": assessment,
            "confidence": float(min(valid_weights, final_score)),
            "details": {
                "final_score": float(final_score),
                "individual_scores": scores,
                "metrics_used": list(valid_scores.keys()),
            },
        }

    def _recommend_clustering_methods(self, data: np.ndarray, cluster_ability: Dict[str, Any]) -> List[str]:
        """推荐聚类方法"""
        assessment = cluster_ability.get("assessment", "unknown")
        confidence = cluster_ability.get("confidence", 0.0)

        n_samples, n_features = data.shape
        is_high_dim = n_features > 20

        # 根据评估结果推荐方法
        method_priority = {
            "highly_clusterable": ["hdbscan_basic", "hdbscan_pca", "hdbscan_umap", "dbscan_fallback"],
            "moderately_clusterable": ["hdbscan_pca", "hdbscan_basic", "hdbscan_umap", "dbscan_fallback"],
            "weakly_clusterable": ["hdbscan_pca", "hdbscan_umap", "hdbscan_basic", "dbscan_fallback"],
        }

        base_methods = method_priority.get(assessment, ["hdbscan_pca", "hdbscan_umap", "hdbscan_basic", "dbscan_fallback"])

        # 根据数据特征调整推荐顺序
        if is_high_dim:
            # 高维数据优先降维方法
            recommendations = [m for m in base_methods if "pca" in m or "umap" in m]
            recommendations.extend([m for m in base_methods if "pca" not in m and "umap" not in m])
        else:
            recommendations = base_methods

        # 去重并限制数量
        seen = set()
        unique_recommendations = []
        for method in recommendations:
            if method not in seen:
                seen.add(method)
                unique_recommendations.append(method)
                if len(unique_recommendations) >= 4:
                    break

        return unique_recommendations
