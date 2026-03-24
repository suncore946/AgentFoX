"""
校准质量评估器 - 优化版
提供简化的校准效果评估，减少冗余逻辑，保持统计严谨性
"""

from enum import Enum
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
from loguru import logger
from scipy.stats import friedmanchisquare


from .calibration_metrics import CalibrationMetrics
from ..configs.config_dataclass import CalibrationConfig
from ..processor.calibration_profile_processor import CalibrationProfilesProcessor


class CalibrationQuality(Enum):
    """校准质量等级"""

    EXCELLENT = "excellent"  # 优秀：ECE很低且显著改进
    GOOD = "good"  # 良好：ECE可接受且有改进
    STABLE_GOOD = "stable_good"  # 稳定良好：原本就好，保持稳定
    MARGINAL = "marginal"  # 边际：有改进但效果微弱
    NO_IMPROVEMENT = "no_improvement"  # 无改进：统计上无显著差异
    DEGRADED = "degraded"  # 降级：原本好的变差了
    INSUFFICIENT = "insufficient"  # 不足：改进后仍不满足要求
    FAILED = "failed"  # 失败：校准方法执行失败


class CalibrationEvaluator:
    """
    校准质量评估器

    核心功能：
    1. 统计显著性检验 + 效应大小评估
    2. 多维质量评估（ECE、准确率、稳定性）
    3. 统一的决策逻辑
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.calibration_llm = CalibrationProfilesProcessor(config.llm, prompt_path=self.config.calibration_analysis_prompt_path)

    def evaluate_calibration(
        self,
        model_name,
        original_result: Dict[str, Dict],
        calibrated_result: Dict[str, Dict],
        model_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        综合评估校准性能与质量
        """
        logger.info(f"开始校准性能与质量综合评估: {model_name}")

        # 1. 性能差异计算
        metric_ori = original_result
        metric_cal = calibrated_result

        # 获取校准前后差异
        ece_diff = metric_ori["ece"] - metric_cal["ece"]
        mce_diff = metric_ori["mce"] - metric_cal["mce"]
        ace_diff = metric_ori["ace"] - metric_cal["ace"]
        brier_diff = metric_ori["brier_score"] - metric_cal["brier_score"]
        f1_diff = metric_ori["f1_score"] - metric_cal["f1_score"]
        log_loss_diff = metric_ori["log_loss"] - metric_cal["log_loss"]

        # 2. 计算Bootstrap置信区间
        # 置信区间的下界，表示在多次重采样下，ECE改进的分布的较低端值（如 2.5% 分位点，置信度 95% 时）。
        # 置信区间的上界，表示在多次重采样下，ECE改进的分布的较高端值（如 97.5% 分位点，置信度 95% 时）。
        ci_lower, ci_upper = CalibrationMetrics.calculate_ece_bootstrap_ci(
            y_true=model_df["gt_label"].values,
            y_prob_original=model_df["pred_prob"].values,
            y_prob_calibrated=model_df["calibration_prob"].values,
        )

        data = {
            "original_result": metric_ori,
            "calibration_result": metric_cal,
            "performance_diff": {
                "ece_diff": round(float(ece_diff), 4),
                # "mce_diff": round(float(mce_diff), 4),
                # "ace_diff": round(float(ace_diff), 4),
                "brier_diff": round(float(brier_diff), 4),
                "f1_diff": round(float(f1_diff), 4),
                "log_loss_diff": round(float(log_loss_diff), 4),
            },
            "ece_bootstrap_ci": {
                "description": "95% confidence interval of ECE improvement calculated using the Bootstrap method",
                "ci_lower": round(float(ci_lower), 4),
                "ci_upper": round(float(ci_upper), 4),
            },
        }
        # 输入大模型进行分析校准是否有效
        analysis_info = self.calibration_llm.run(calibration_data=data)
        # analysis_info = None
        return {
            **data,
            "original_result": {
                "description": "Performance metrics before calibration on the validation set",
                "metrics": metric_ori,
            },
            "calibration_result": {
                "description": "Performance metrics after calibration on the validation set",
                "metrics": metric_cal,
            },
            "ece_bootstrap_ci": {
                "description": "95% confidence interval of ECE improvement calculated using the Bootstrap method",
                "ci_lower": round(float(ci_lower), 4),
                "ci_upper": round(float(ci_upper), 4),
            },
            "calibration_analysis": {
                "description": "Evaluation of calibration quality based on statistical results and LLM analysis",
                "analysis": analysis_info,
            },
        }

    def evaluate_groups_performance(
        self,
        result_df: pd.DataFrame,
        target_column: str = "pred_label",
        group_columns: str | List[str] | None = None,
    ) -> Dict[str, Dict]:
        """评估各个分组的性能指标，如果group_columns为空则计算全局性能"""
        individual_performance = {}
        assert target_column in result_df.columns, f"目标列 '{target_column}' 不存在于数据框中"
        assert not result_df.empty, "输入数据为空，无法计算全局性能指标"

        # 处理 group_columns 为 None 或空的情况：计算全局性能
        if group_columns is None or (isinstance(group_columns, (list, tuple)) and len(group_columns) == 0):
            logger.info(f"计算全局性能指标, 指标目标列'{target_column}'")
            return {"global": self.evaluate_metrics(result_df, target_column)}

        # 支持 group_columns 为 str 或 List[str]
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        else:
            group_columns = list(group_columns)

        # 验证分组列存在
        for col in group_columns:
            assert col in result_df.columns, f"分组列 '{col}' 不存在于数据框中"

        for group_name, group_df in result_df.groupby(group_columns):
            # 检查分组数据是否有效
            if group_df.empty:
                logger.warning(f"分组 '{group_name}' 数据为空")
                individual_performance[str(group_name)] = {}
                continue

            # 处理group_name可能是元组的情况
            if isinstance(group_name, tuple):
                # 如果是多级分组，将元组转换为字符串键
                if len(group_name) == 1:
                    key = str(group_name[0])
                else:
                    key = "*".join(str(x) for x in group_name)
            else:
                key = str(group_name)

            # 直接将分组数据输入给性能分析器
            individual_performance[key] = self.evaluate_metrics(group_df, target_column)

        return individual_performance

    def evaluate_ensemble_improvement(
        self,
        ensemble_performances: Dict[str, float],
        original_performances: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        直接计算集成相对于分组的性能提升

        Args:
            ensemble_performance: 集成的性能指标字典
            group_performances: 各分组的性能指标字典

        Returns:
            Dict[str, float]: 包含性能提升对比的字典
        """
        # 提取各分组的性能指标
        group_ece_scores = original_performances["ece"]
        group_accuracy_scores = original_performances["acc"]

        # 计算分组统计指标
        avg_group_ece = float(np.mean(group_ece_scores))
        best_group_ece = float(min(group_ece_scores))

        avg_group_accuracy = float(np.mean(group_accuracy_scores))

        best_group_accuracy = float(max(group_accuracy_scores))

        # 获取集成性能
        ensemble_ece = float(ensemble_performances["ece"])
        ensemble_accuracy = float(ensemble_performances["accuracy"])

        # 构建结果字典
        result = {
            # 集成性能
            "ensemble_ece": ensemble_ece,
            "ensemble_accuracy": ensemble_accuracy,
            "ensemble_brier_score": float(ensemble_performances.get("brier_score", 0.0)),
            # "ensemble_mce": float(ensemble_performances.get("mce", 0.0)),
            # 分组统计
            "avg_group_ece": avg_group_ece,
            "avg_group_accuracy": avg_group_accuracy,
            "best_group_ece": best_group_ece,
            "best_group_accuracy": best_group_accuracy,
            "worst_group_ece": float(max(group_ece_scores)),
            "worst_group_accuracy": float(min(group_accuracy_scores)),
            # 直接的性能对比（正值表示集成更好）
            "ece_vs_avg": avg_group_ece - ensemble_ece,
            "ece_vs_best": best_group_ece - ensemble_ece,
            "accuracy_vs_avg": ensemble_accuracy - avg_group_accuracy,
            "accuracy_vs_best": ensemble_accuracy - best_group_accuracy,
            # 元信息
            "n_groups": len(original_performances),
            "group_names": list(original_performances.keys()),
        }

        # 记录性能对比
        logger.info(
            f"集成性能对比 - ECE: {ensemble_ece:.4f} vs 平均{avg_group_ece:.4f} vs 最佳{best_group_ece:.4f}; "
            f"准确率: {ensemble_accuracy:.4f} vs 平均{avg_group_accuracy:.4f} vs 最佳{best_group_accuracy:.4f}"
        )

        return result

    def evaluate_metrics(self, df: pd.DataFrame, target_column: str = "pred_label") -> dict:
        """
        计算基础指标

        Args:
            df: 包含预测结果的数据框
            target_column: 目标列名，包含预测概率或标签
            ece_bins: ECE计算的分箱数量
            threshold: 二分类阈值

        Returns:
            包含各项指标的数据框
        """
        if df.empty:
            return {}

        # 检查必需的列
        required_columns = ["gt_label", target_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")

        y_true = df["gt_label"].values
        y_prob = df[target_column].values

        # result = {
        #     "overall": CalibrationMetrics.calculate_class_specific_metrics(y_true, y_prob),
        #     "real": CalibrationMetrics.calculate_class_specific_metrics(y_true, y_prob, 0),
        #     "fake": CalibrationMetrics.calculate_class_specific_metrics(y_true, y_prob, 1),
        # }
        result = CalibrationMetrics.calculate_class_specific_metrics(y_true, y_prob)
        return result

    def evaluate_significance_rank(self, performance_data: pd.DataFrame):
        """
        接受一个 pandas.DataFrame（每行为一次重复/fold/bootstrap），必须包含列:
        - model_name
        - f1_score
        - brier_score
        """
        n_models = performance_data["model_name"].nunique()
        if n_models < 3:
            raise ValueError("模型数量不足（需 >= 3）以进行 Friedman 检验")

        f1_lists = performance_data["f1_score"].astype(float).values.tolist()
        brier_lists = performance_data["brier_score"].astype(float).values.tolist()
        try:
            friedman_stat_f1, p_value_f1 = friedmanchisquare(*f1_lists)
            friedman_stat_brier, p_value_brier = friedmanchisquare(*brier_lists)
            alpha = getattr(self.config, "significance_level", 0.05) if hasattr(self, "config") else 0.05
            significance_rankings = {
                "f1_score": {
                    "statistic": round(float(friedman_stat_f1), 4),
                    "p_value": round(float(p_value_f1), 4),
                    "significant": bool(p_value_f1 < alpha),
                    "description": "如果p值小于显著性水平（阈值设为0.05），则表示至少有两个模型在 F1 分数上存在显著差异。",
                },
                "brier_score": {
                    "statistic": round(float(friedman_stat_brier), 4),
                    "p_value": round(float(p_value_brier), 4),
                    "significant": bool(p_value_brier < alpha),
                    "description": "如果p值小于显著性水平（阈值设为0.05），则表示至少有两个模型在 Brier 分数上存在显著差异。",
                },
            }
        except Exception as e:
            logger.warning(f"Statistical testing failed: {str(e)}")
            significance_rankings = {"error": str(e)}

        return significance_rankings
