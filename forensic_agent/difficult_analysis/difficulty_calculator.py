"""
核心难度计算模块
实现基于多模型聚合的样本难度度量
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any
import warnings

from ..config.settings import DifficultyConfig
from ..utils.difficult_utils import (
    calculate_nll,
    calculate_entropy,
    calculate_disagreement_rate,
    robust_standardize,
    validate_prediction_matrix,
    calculate_percentiles,
    clip_probabilities,
)


class DifficultyCalculator:
    """难度计算器"""

    def __init__(self, config: Optional[DifficultyConfig] = None):
        """
        初始化难度计算器

        Args:
            config: 配置对象，默认使用平衡配置
        """
        self.config = config or DifficultyConfig()

    def calculate_base_difficulty(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        计算基础难度 D_base

        D_base(i) = 平均_m NLL_im
        其中 NLL_im = -[y_i * log(p_im) + (1-y_i) * log(1-p_im)]

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)
            labels: 真实标签 (n_samples,)

        Returns:
            基础难度数组 (n_samples,)
        """
        # 验证输入
        is_valid, error_msg = validate_prediction_matrix(predictions, labels)
        if not is_valid:
            raise ValueError(f"输入验证失败: {error_msg}")

        n_samples, n_models = predictions.shape
        nll_matrix = np.zeros_like(predictions)

        # 为每个模型计算NLL
        for m in range(n_models):
            nll_matrix[:, m] = calculate_nll(labels, predictions[:, m], eps=self.config.eps)

        # 计算每个样本的平均NLL作为基础难度
        base_difficulty = np.mean(nll_matrix, axis=1)

        return base_difficulty

    def calculate_disagreement_metrics(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算分歧度指标

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)

        Returns:
            包含分歧度指标的字典
        """
        # 分歧度：模型间二值预测的不一致率
        disagree_rate = calculate_disagreement_rate(predictions, threshold=0.5)

        # 概率方差：模型概率的方差
        prob_variance = np.var(predictions, axis=1)

        # 概率标准差
        prob_std = np.std(predictions, axis=1)

        # 平均概率
        mean_prob = np.mean(predictions, axis=1)

        return {"disagree_rate": disagree_rate, "prob_variance": prob_variance, "prob_std": prob_std, "mean_prob": mean_prob}

    def calculate_uncertainty_decomposition(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算基于熵的不确定性分解

        根据指导文档：
        - p̄_i = 平均_m p_im
        - 总不确定性 H_tot(i) = 熵(p̄_i)
        - 平均熵 H_mean(i) = 平均_m 熵(p_im)
        - 认知不确定性 Epistemic(i) = H_tot(i) - H_mean(i)
        - 数据内在不确定性 Aleatoric(i) = H_mean(i)

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)

        Returns:
            包含不确定性分解的字典
        """
        n_samples, n_models = predictions.shape

        # 计算平均概率
        mean_probs = np.mean(predictions, axis=1)

        # 计算总不确定性：H(平均概率)
        total_uncertainty = calculate_entropy(mean_probs, eps=self.config.eps)

        # 计算每个模型的熵
        entropy_matrix = np.zeros_like(predictions)
        for m in range(n_models):
            entropy_matrix[:, m] = calculate_entropy(predictions[:, m], eps=self.config.eps)

        # 计算平均熵
        mean_entropy = np.mean(entropy_matrix, axis=1)

        # 计算认知不确定性（模型间分歧导致的不确定性）
        epistemic_uncertainty = total_uncertainty - mean_entropy

        # 数据内在不确定性（模型内平均不确定性）
        aleatoric_uncertainty = mean_entropy

        return {
            "total_uncertainty": total_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "mean_entropy": mean_entropy,
            "mean_probs": mean_probs,
        }

    def calculate_final_difficulty(self, base_difficulty: np.ndarray, epistemic_uncertainty: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算最终难度分数（稳健融合）

        根据指导文档：
        D_final_raw(i) = z(D_base) + λ1 · z(Epistemic)
        最终输出采用分位裁剪到[1%, 99%]后min-max归一化到[0,1]

        Args:
            base_difficulty: 基础难度
            epistemic_uncertainty: 认知不确定性

        Returns:
            包含最终难度的字典
        """
        # 对基础难度和认知不确定性进行稳健标准化
        d_base_normalized = robust_standardize(
            base_difficulty,
            method=self.config.standardization_method,
            lower_q=self.config.quantile_lower,
            upper_q=self.config.quantile_upper,
        )

        epistemic_normalized = robust_standardize(
            epistemic_uncertainty,
            method=self.config.standardization_method,
            lower_q=self.config.quantile_lower,
            upper_q=self.config.quantile_upper,
        )

        # 融合：D_final_raw = z(D_base) + λ1 · z(Epistemic)
        final_difficulty_raw = d_base_normalized + self.config.epistemic_weight * epistemic_normalized

        # 最终归一化到[0, 1]
        final_difficulty_normalized = robust_standardize(
            final_difficulty_raw, method="quantile", lower_q=self.config.quantile_lower, upper_q=self.config.quantile_upper
        )

        return {
            "d_base_normalized": d_base_normalized,
            "epistemic_normalized": epistemic_normalized,
            "final_difficulty_raw": final_difficulty_raw,
            "final_difficulty_normalized": final_difficulty_normalized,
        }

    def run(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        计算综合难度度量

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)
            labels: 真实标签 (n_samples,)

        Returns:
            包含所有难度度量的字典
        """
        # 验证输入
        is_valid, error_msg = validate_prediction_matrix(predictions, labels)
        if not is_valid:
            raise ValueError(f"输入验证失败: {error_msg}")

        n_samples, n_models = predictions.shape

        # 1. 计算基础难度
        base_difficulty = self.calculate_base_difficulty(predictions, labels)

        # 2. 计算分歧度指标
        disagreement_metrics = self.calculate_disagreement_metrics(predictions)

        # 3. 计算不确定性分解
        uncertainty_metrics = self.calculate_uncertainty_decomposition(predictions)

        # 4. 计算最终难度
        final_difficulty_metrics = self.calculate_final_difficulty(base_difficulty, uncertainty_metrics["epistemic_uncertainty"])

        # 5. 计算统计信息
        statistics = self._calculate_statistics(base_difficulty, uncertainty_metrics, disagreement_metrics)

        # 合并所有结果
        results = {
            "n_samples": n_samples,
            "n_models": n_models,
            # 核心难度度量
            "base_difficulty": base_difficulty,
            "final_difficulty_raw": final_difficulty_metrics["final_difficulty_raw"],
            "final_difficulty_normalized": final_difficulty_metrics["final_difficulty_normalized"],
            # 不确定性分解
            "epistemic_uncertainty": uncertainty_metrics["epistemic_uncertainty"],
            "aleatoric_uncertainty": uncertainty_metrics["aleatoric_uncertainty"],
            "total_uncertainty": uncertainty_metrics["total_uncertainty"],
            "mean_entropy": uncertainty_metrics["mean_entropy"],
            "mean_probs": uncertainty_metrics["mean_probs"],
            # 分歧度指标
            "disagree_rate": disagreement_metrics["disagree_rate"],
            "prob_variance": disagreement_metrics["prob_variance"],
            "prob_std": disagreement_metrics["prob_std"],
            # 标准化中间结果
            "d_base_normalized": final_difficulty_metrics["d_base_normalized"],
            "epistemic_normalized": final_difficulty_metrics["epistemic_normalized"],
            # 统计信息
            "statistics": statistics,
        }

        return results

    def _calculate_statistics(
        self, base_difficulty: np.ndarray, uncertainty_metrics: Dict[str, np.ndarray], disagreement_metrics: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """计算各指标的统计信息"""

        metrics_to_analyze = {
            "base_difficulty": base_difficulty,
            "epistemic_uncertainty": uncertainty_metrics["epistemic_uncertainty"],
            "aleatoric_uncertainty": uncertainty_metrics["aleatoric_uncertainty"],
            "total_uncertainty": uncertainty_metrics["total_uncertainty"],
            "disagree_rate": disagreement_metrics["disagree_rate"],
            "prob_variance": disagreement_metrics["prob_variance"],
            "mean_probs": uncertainty_metrics["mean_probs"],
        }

        statistics = {}
        for name, values in metrics_to_analyze.items():
            statistics[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                **calculate_percentiles(values, self.config.percentiles),
            }

        return statistics

    def create_difficulty_dataframe(self, results: Dict[str, Any], image_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        将难度计算结果转换为DataFrame格式

        Args:
            results: compute_comprehensive_difficulty的输出结果
            image_ids: 图像ID数组，如果提供则添加到DataFrame中

        Returns:
            包含所有难度度量的DataFrame
        """
        n_samples = results["n_samples"]

        # 创建基础DataFrame
        df_data = {
            "base_difficulty": results["base_difficulty"],
            "final_difficulty_raw": results["final_difficulty_raw"],
            "final_difficulty_normalized": results["final_difficulty_normalized"],
            "epistemic_uncertainty": results["epistemic_uncertainty"],
            "aleatoric_uncertainty": results["aleatoric_uncertainty"],
            "total_uncertainty": results["total_uncertainty"],
            "disagree_rate": results["disagree_rate"],
            "prob_variance": results["prob_variance"],
            "prob_std": results["prob_std"],
            "mean_probs": results["mean_probs"],
            "mean_entropy": results["mean_entropy"],
        }

        # 添加图像ID
        if image_ids is not None:
            if len(image_ids) != n_samples:
                raise ValueError(f"image_ids长度({len(image_ids)})与样本数量({n_samples})不匹配")
            df_data["image_id"] = image_ids
        else:
            df_data["image_id"] = np.arange(n_samples)

        df = pd.DataFrame(df_data)

        # 重新排列列顺序
        columns_order = [
            "image_id",
            "final_difficulty_normalized",
            "base_difficulty",
            "epistemic_uncertainty",
            "aleatoric_uncertainty",
            "total_uncertainty",
            "disagree_rate",
            "prob_variance",
            "prob_std",
            "mean_probs",
            "mean_entropy",
            "final_difficulty_raw",
        ]

        df = df[columns_order]

        return df

    def rank_samples_by_difficulty(
        self, results: Dict[str, Any], difficulty_key: str = "final_difficulty_normalized", ascending: bool = False
    ) -> np.ndarray:
        """
        根据难度对样本进行排序

        Args:
            results: 难度计算结果
            difficulty_key: 用于排序的难度指标键名
            ascending: 是否升序排列

        Returns:
            排序后的样本索引
        """
        if difficulty_key not in results:
            available_keys = [k for k in results.keys() if isinstance(results[k], np.ndarray)]
            raise ValueError(f"难度键 '{difficulty_key}' 不存在。可用键: {available_keys}")

        difficulty_values = results[difficulty_key]
        sorted_indices = np.argsort(difficulty_values)

        if not ascending:
            sorted_indices = sorted_indices[::-1]

        return sorted_indices

    def get_top_difficult_samples(
        self, results: Dict[str, Any], top_k: int = 100, difficulty_key: str = "final_difficulty_normalized"
    ) -> Dict[str, Any]:
        """
        获取最难的k个样本

        Args:
            results: 难度计算结果
            top_k: 返回样本数量
            difficulty_key: 难度指标键名

        Returns:
            包含最难样本信息的字典
        """
        sorted_indices = self.rank_samples_by_difficulty(results, difficulty_key, ascending=False)

        top_indices = sorted_indices[:top_k]

        # 提取最难样本的信息
        top_samples = {
            "indices": top_indices,
            "difficulties": results[difficulty_key][top_indices],
            "base_difficulties": results["base_difficulty"][top_indices],
            "epistemic_uncertainties": results["epistemic_uncertainty"][top_indices],
            "aleatoric_uncertainties": results["aleatoric_uncertainty"][top_indices],
            "disagree_rates": results["disagree_rate"][top_indices],
            "mean_probs": results["mean_probs"][top_indices],
        }

        return top_samples

    def print_summary(self, results: Dict[str, Any]):
        """
        打印难度计算结果摘要

        Args:
            results: 难度计算结果
        """
        print("=== 难度度量计算摘要 ===")
        print(f"样本数量: {results['n_samples']}")
        print(f"模型数量: {results['n_models']}")

        print(f"\n主要难度指标统计:")
        key_metrics = ["final_difficulty_normalized", "base_difficulty", "epistemic_uncertainty", "aleatoric_uncertainty"]

        for metric in key_metrics:
            if metric in results["statistics"]:
                stats = results["statistics"][metric]
                print(f"  {metric}:")
                print(f"    均值±标准差: {stats['mean']:.4f}±{stats['std']:.4f}")
                print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    中位数: {stats['median']:.4f}")
                print(f"    P90/P95: {stats['p90']:.4f}/{stats['p95']:.4f}")

        # 分歧度和方差统计
        print(f"\n分歧度指标统计:")
        disagree_stats = results["statistics"]["disagree_rate"]
        variance_stats = results["statistics"]["prob_variance"]
        print(f"  分歧度: {disagree_stats['mean']:.4f}±{disagree_stats['std']:.4f}")
        print(f"  概率方差: {variance_stats['mean']:.4f}±{variance_stats['std']:.4f}")
