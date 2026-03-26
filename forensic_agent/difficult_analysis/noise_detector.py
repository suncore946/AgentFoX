"""
噪声检测模块
实现噪声与域外样本的防护与标记
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import warnings

from ..config.settings import DifficultyConfig
from ..utils.difficult_utils import calculate_percentiles


class SampleRiskType(Enum):
    """样本风险类型枚举"""

    NORMAL = "normal"  # 正常样本
    LABEL_NOISE = "label_noise"  # 可能标错标签
    OOD_SHIFT = "ood_shift"  # 域外/分布偏移
    AMBIGUOUS = "ambiguous"  # 含糊/本质不确定
    HARD_VALUABLE = "hard_valuable"  # 稳健难样本
    MULTIPLE_RISK = "multiple_risk"  # 多重风险


class NoiseDetector:
    """噪声检测器"""

    def __init__(self, config: Optional[DifficultyConfig] = None):
        """
        初始化噪声检测器

        Args:
            config: 配置对象
        """
        self.config = config or DifficultyConfig()

    def run(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainty_metrics: Dict[str, np.ndarray],
        disagreement_metrics: Dict[str, np.ndarray],
        difficulty_metrics: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        检测样本风险类型

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)
            labels: 真实标签 (n_samples,)
            uncertainty_metrics: 不确定性指标
            disagreement_metrics: 分歧度指标
            difficulty_metrics: 难度指标

        Returns:
            包含风险检测结果的字典
        """
        n_samples = len(labels)

        # 提取关键指标
        mean_probs = uncertainty_metrics["mean_probs"]
        epistemic = uncertainty_metrics["epistemic_uncertainty"]
        aleatoric = uncertainty_metrics["aleatoric_uncertainty"]
        total_uncertainty = uncertainty_metrics["total_uncertainty"]
        mean_entropy = uncertainty_metrics["mean_entropy"]

        disagree_rate = disagreement_metrics["disagree_rate"]
        prob_variance = disagreement_metrics["prob_variance"]

        base_difficulty = difficulty_metrics["base_difficulty"]

        # 计算分位数阈值
        thresholds = self._calculate_thresholds(
            mean_probs, epistemic, aleatoric, total_uncertainty, disagree_rate, prob_variance, base_difficulty
        )

        # 检测各类风险
        label_noise_mask = self._detect_label_noise(mean_probs, labels, disagree_rate, thresholds)

        ood_shift_mask = self._detect_ood_shift(epistemic, prob_variance, mean_entropy, thresholds)

        ambiguous_mask = self._detect_ambiguous(aleatoric, total_uncertainty, mean_probs, thresholds)

        hard_valuable_mask = self._detect_hard_valuable(base_difficulty, epistemic, label_noise_mask, ood_shift_mask, thresholds)

        # 组合风险类型
        risk_types, risk_flags = self._combine_risk_types(label_noise_mask, ood_shift_mask, ambiguous_mask, hard_valuable_mask)

        # 计算统计信息
        statistics = self._calculate_risk_statistics(risk_types, risk_flags, thresholds)

        return {
            "risk_types": risk_types,
            "risk_flags": risk_flags,
            "label_noise_mask": label_noise_mask,
            "ood_shift_mask": ood_shift_mask,
            "ambiguous_mask": ambiguous_mask,
            "hard_valuable_mask": hard_valuable_mask,
            "thresholds": thresholds,
            "statistics": statistics,
        }

    def _calculate_thresholds(
        self,
        mean_probs: np.ndarray,
        epistemic: np.ndarray,
        aleatoric: np.ndarray,
        total_uncertainty: np.ndarray,
        disagree_rate: np.ndarray,
        prob_variance: np.ndarray,
        base_difficulty: np.ndarray,
    ) -> Dict[str, float]:
        """计算动态阈值（基于分位数）"""

        # 计算各指标的分位数阈值
        epistemic_threshold = np.percentile(epistemic, self.config.ood_epistemic_threshold * 100)
        var_threshold = np.percentile(prob_variance, self.config.ood_var_threshold * 100)
        aleatoric_threshold = np.percentile(aleatoric, self.config.ambiguous_aleatoric_threshold * 100)
        total_uncertainty_threshold = np.percentile(total_uncertainty, self.config.ambiguous_aleatoric_threshold * 100)
        base_difficulty_threshold = np.percentile(base_difficulty, 90)  # 高难度样本的90%分位数

        # 置信度阈值（固定）
        confidence_threshold = self.config.label_noise_confidence_threshold
        disagree_threshold = self.config.label_noise_disagree_threshold
        prob_threshold = self.config.ambiguous_prob_threshold

        return {
            "confidence_threshold": confidence_threshold,
            "disagree_threshold": disagree_threshold,
            "epistemic_threshold": epistemic_threshold,
            "var_threshold": var_threshold,
            "aleatoric_threshold": aleatoric_threshold,
            "total_uncertainty_threshold": total_uncertainty_threshold,
            "base_difficulty_threshold": base_difficulty_threshold,
            "prob_threshold": prob_threshold,
        }

    def _detect_label_noise(
        self, mean_probs: np.ndarray, labels: np.ndarray, disagree_rate: np.ndarray, thresholds: Dict[str, float]
    ) -> np.ndarray:
        """
        检测可能标错标签的样本

        条件：p̄_i 与 y_i 明显相反且置信高（max(p̄_i, 1-p̄_i) 高分位，如 >0.9），
              且 Disagree_rate 低（如 <0.2），说明多数模型一致且自信地与标注相反。
        """
        # 计算模型置信度：max(p̄_i, 1-p̄_i)
        model_confidence = np.maximum(mean_probs, 1 - mean_probs)

        # 检查预测与标签是否相反且置信度高
        prediction_opposite = (mean_probs > 0.5) != (labels == 1)
        high_confidence = model_confidence > thresholds["confidence_threshold"]
        low_disagreement = disagree_rate < thresholds["disagree_threshold"]

        # 标签噪声风险 = 预测相反 & 高置信度 & 低分歧
        label_noise_mask = prediction_opposite & high_confidence & low_disagreement

        return label_noise_mask

    def _detect_ood_shift(
        self, epistemic: np.ndarray, prob_variance: np.ndarray, mean_entropy: np.ndarray, thresholds: Dict[str, float]
    ) -> np.ndarray:
        """
        检测域外/分布偏移样本

        条件：Epistemic 高分位（如 >0.9）且 Var_p 高分位（如 >0.9），
              同时 H_mean 不高（模型各自自信但彼此冲突）
        """
        high_epistemic = epistemic > thresholds["epistemic_threshold"]
        high_variance = prob_variance > thresholds["var_threshold"]

        # H_mean不高表示模型各自比较自信
        # 使用中位数作为"不高"的判断标准
        mean_entropy_median = np.median(mean_entropy)
        low_mean_entropy = mean_entropy <= mean_entropy_median

        # OOD风险 = 高认知不确定性 & 高方差 & 低平均熵
        ood_shift_mask = high_epistemic & high_variance & low_mean_entropy

        return ood_shift_mask

    def _detect_ambiguous(
        self, aleatoric: np.ndarray, total_uncertainty: np.ndarray, mean_probs: np.ndarray, thresholds: Dict[str, float]
    ) -> np.ndarray:
        """
        检测含糊/本质不确定样本

        条件：Aleatoric 和 H_tot 都较高，p̄_i 接近 0.5，多模型也不自信，属于任务内在难点。
        """
        high_aleatoric = aleatoric > thresholds["aleatoric_threshold"]
        high_total_uncertainty = total_uncertainty > thresholds["total_uncertainty_threshold"]

        # p̄_i 接近 0.5
        prob_near_half = np.abs(mean_probs - 0.5) < thresholds["prob_threshold"]

        # 含糊样本 = 高数据不确定性 & 高总不确定性 & 概率接近0.5
        ambiguous_mask = high_aleatoric & high_total_uncertainty & prob_near_half

        return ambiguous_mask

    def _detect_hard_valuable(
        self,
        base_difficulty: np.ndarray,
        epistemic: np.ndarray,
        label_noise_mask: np.ndarray,
        ood_shift_mask: np.ndarray,
        thresholds: Dict[str, float],
    ) -> np.ndarray:
        """
        检测稳健难样本

        条件：D_base 高分位与中高 Epistemic，但无明显噪声或 OOD 信号。
        """
        high_base_difficulty = base_difficulty > thresholds["base_difficulty_threshold"]

        # 中高认知不确定性：使用75%分位数作为阈值
        epistemic_75th = np.percentile(epistemic, 75)
        medium_high_epistemic = epistemic > epistemic_75th

        # 无明显噪声或OOD信号
        no_noise_ood = ~(label_noise_mask | ood_shift_mask)

        # 稳健难样本 = 高基础难度 & 中高认知不确定性 & 无噪声/OOD
        hard_valuable_mask = high_base_difficulty & medium_high_epistemic & no_noise_ood

        return hard_valuable_mask

    def _combine_risk_types(
        self, label_noise_mask: np.ndarray, ood_shift_mask: np.ndarray, ambiguous_mask: np.ndarray, hard_valuable_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """组合风险类型"""
        n_samples = len(label_noise_mask)
        risk_types = np.full(n_samples, SampleRiskType.NORMAL.value, dtype=object)

        # 记录各类风险标识
        risk_flags = {
            "label_noise": label_noise_mask,
            "ood_shift": ood_shift_mask,
            "ambiguous": ambiguous_mask,
            "hard_valuable": hard_valuable_mask,
        }

        # 计算每个样本的风险数量
        total_risks = (
            label_noise_mask.astype(int) + ood_shift_mask.astype(int) + ambiguous_mask.astype(int) + hard_valuable_mask.astype(int)
        )

        # 分配风险类型（按优先级）
        # 1. 多重风险
        multiple_risk_mask = total_risks > 1
        risk_types[multiple_risk_mask] = SampleRiskType.MULTIPLE_RISK.value

        # 2. 单一风险类型（按检测严重性排序）
        single_risk_mask = total_risks == 1

        risk_types[single_risk_mask & label_noise_mask] = SampleRiskType.LABEL_NOISE.value
        risk_types[single_risk_mask & ood_shift_mask] = SampleRiskType.OOD_SHIFT.value
        risk_types[single_risk_mask & ambiguous_mask] = SampleRiskType.AMBIGUOUS.value
        risk_types[single_risk_mask & hard_valuable_mask] = SampleRiskType.HARD_VALUABLE.value

        return risk_types, risk_flags

    def _calculate_risk_statistics(
        self, risk_types: np.ndarray, risk_flags: Dict[str, np.ndarray], thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """计算风险统计信息"""

        # 各类风险的数量统计
        risk_counts = {}
        for risk_type in SampleRiskType:
            risk_counts[risk_type.value] = np.sum(risk_types == risk_type.value)

        # 总样本数
        total_samples = len(risk_types)

        # 风险比例
        risk_proportions = {k: v / total_samples for k, v in risk_counts.items()}

        # 各类风险标识统计
        flag_counts = {k: int(np.sum(v)) for k, v in risk_flags.items()}
        flag_proportions = {k: v / total_samples for k, v in flag_counts.items()}

        return {
            "total_samples": total_samples,
            "risk_counts": risk_counts,
            "risk_proportions": risk_proportions,
            "flag_counts": flag_counts,
            "flag_proportions": flag_proportions,
            "thresholds_used": thresholds,
            "high_risk_samples": risk_counts[SampleRiskType.LABEL_NOISE.value] + risk_counts[SampleRiskType.OOD_SHIFT.value],
            "valuable_samples": risk_counts[SampleRiskType.HARD_VALUABLE.value],
            "ambiguous_samples": risk_counts[SampleRiskType.AMBIGUOUS.value],
        }

    def filter_samples_by_risk(
        self, risk_detection_results: Dict[str, Any], include_types: Optional[List[str]] = None, exclude_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        根据风险类型过滤样本

        Args:
            risk_detection_results: 风险检测结果
            include_types: 包含的风险类型列表
            exclude_types: 排除的风险类型列表

        Returns:
            符合条件的样本索引
        """
        risk_types = risk_detection_results["risk_types"]
        n_samples = len(risk_types)

        if include_types is not None:
            # 只包含指定类型
            include_mask = np.zeros(n_samples, dtype=bool)
            for risk_type in include_types:
                include_mask |= risk_types == risk_type
            filter_mask = include_mask
        else:
            # 默认包含所有样本
            filter_mask = np.ones(n_samples, dtype=bool)

        if exclude_types is not None:
            # 排除指定类型
            exclude_mask = np.zeros(n_samples, dtype=bool)
            for risk_type in exclude_types:
                exclude_mask |= risk_types == risk_type
            filter_mask &= ~exclude_mask

        return np.where(filter_mask)[0]

    def get_high_value_samples(self, risk_detection_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        获取高价值样本（推荐用于训练改进）

        Args:
            risk_detection_results: 风险检测结果

        Returns:
            包含各类高价值样本索引的字典
        """
        # 稳健难样本：最有价值
        hard_valuable_indices = self.filter_samples_by_risk(risk_detection_results, include_types=[SampleRiskType.HARD_VALUABLE.value])

        # 含糊样本：可用但需谨慎
        ambiguous_indices = self.filter_samples_by_risk(risk_detection_results, include_types=[SampleRiskType.AMBIGUOUS.value])

        # 正常样本：基础训练样本
        normal_indices = self.filter_samples_by_risk(risk_detection_results, include_types=[SampleRiskType.NORMAL.value])

        return {
            "hard_valuable": hard_valuable_indices,
            "ambiguous": ambiguous_indices,
            "normal": normal_indices,
            "recommended_for_training": np.concatenate([hard_valuable_indices, normal_indices]),
        }

    def get_suspicious_samples(self, risk_detection_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        获取可疑样本（建议复核或剔除）

        Args:
            risk_detection_results: 风险检测结果

        Returns:
            包含各类可疑样本索引的字典
        """
        # 标签噪声样本：优先复核
        label_noise_indices = self.filter_samples_by_risk(risk_detection_results, include_types=[SampleRiskType.LABEL_NOISE.value])

        # OOD样本：可能需要剔除
        ood_indices = self.filter_samples_by_risk(risk_detection_results, include_types=[SampleRiskType.OOD_SHIFT.value])

        # 多重风险样本：高度可疑
        multiple_risk_indices = self.filter_samples_by_risk(risk_detection_results, include_types=[SampleRiskType.MULTIPLE_RISK.value])

        return {
            "label_noise": label_noise_indices,
            "ood_shift": ood_indices,
            "multiple_risk": multiple_risk_indices,
            "all_suspicious": np.concatenate([label_noise_indices, ood_indices, multiple_risk_indices]),
        }

    def create_risk_dataframe(self, risk_detection_results: Dict[str, Any], image_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        将风险检测结果转换为DataFrame格式

        Args:
            risk_detection_results: 风险检测结果
            image_ids: 图像ID数组

        Returns:
            包含风险信息的DataFrame
        """
        n_samples = len(risk_detection_results["risk_types"])

        # 创建基础DataFrame
        df_data = {
            "risk_type": risk_detection_results["risk_types"],
        }

        # 添加风险标识
        for flag_name, flag_values in risk_detection_results["risk_flags"].items():
            df_data[f"is_{flag_name}"] = flag_values

        # 添加图像ID
        if image_ids is not None:
            if len(image_ids) != n_samples:
                raise ValueError(f"image_ids长度({len(image_ids)})与样本数量({n_samples})不匹配")
            df_data["image_id"] = image_ids
        else:
            df_data["image_id"] = np.arange(n_samples)

        df = pd.DataFrame(df_data)

        # 重新排列列顺序
        columns_order = ["image_id", "risk_type"] + [col for col in df.columns if col.startswith("is_")]
        df = df[columns_order]

        return df

    def print_risk_summary(self, risk_detection_results: Dict[str, Any]):
        """
        打印风险检测结果摘要

        Args:
            risk_detection_results: 风险检测结果
        """
        stats = risk_detection_results["statistics"]
        print("=== 样本风险检测摘要 ===")
        print(f"总样本数量: {stats['total_samples']}")
        print(f"\n风险类型分布:")
        for risk_type, count in stats["risk_counts"].items():
            proportion = stats["risk_proportions"][risk_type]
            print(f"  {risk_type}: {count} ({proportion:.2%})")

        print(f"\n风险标识统计:")
        for flag_name, count in stats["flag_counts"].items():
            proportion = stats["flag_proportions"][flag_name]
            print(f"  {flag_name}: {count} ({proportion:.2%})")

        print(f"\n关键指标:")
        print(f"  高风险样本 (标签噪声+OOD): {stats['high_risk_samples']} " f"({stats['high_risk_samples']/stats['total_samples']:.2%})")
        print(f"  有价值难样本: {stats['valuable_samples']} " f"({stats['valuable_samples']/stats['total_samples']:.2%})")
        print(f"  含糊样本: {stats['ambiguous_samples']} " f"({stats['ambiguous_samples']/stats['total_samples']:.2%})")

        print(f"\n使用建议:")
        print(f"  - 优先复核标签噪声样本: {stats['flag_counts']['label_noise']} 个")
        print(f"  - 考虑剔除OOD样本: {stats['flag_counts']['ood_shift']} 个")
        print(f"  - 推荐用于困难样本训练: {stats['valuable_samples']} 个")

        print(f"\n检测阈值:")
        thresholds = stats["thresholds_used"]
        for key, value in thresholds.items():
            print(f"  {key}: {value:.4f}")
