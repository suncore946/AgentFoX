"""
统一报告管理器模块
负责生成、打印和管理所有类型的模型报告
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from operator import itemgetter
from collections import Counter


class ReportManager:
    """统一报告管理器 - 集成所有报告功能和文件保存功能"""

    # 配置常量
    PERFORMANCE_THRESHOLDS = [0.8, 0.7, 0.6]  # 优秀, 良好, 中等
    CALIBRATION_THRESHOLDS = [0.05, 0.1, 0.15]  # 极佳, 良好, 一般

    PERFORMANCE_LABELS = ["优秀", "良好", "中等", "有待提升"]
    CALIBRATION_LABELS = ["极佳", "良好", "一般", "较差"]

    RANK_EMOJIS = {1: "🥇 最佳模型", 2: "🥈 第二名", 3: "🥉 第三名"}

    def __init__(
        self,
        profiles_dir: Optional[Path] = None,
    ):
        self.profiles_dir = profiles_dir

    @classmethod
    def _get_tier_label(cls, value: float, thresholds: list, labels: list, reverse: bool = False) -> str:
        """获取等级标签"""
        if reverse:  # 对于ECE等指标，值越小越好
            for i, threshold in enumerate(thresholds):
                if value <= threshold:
                    return labels[i]
            return labels[-1]
        else:  # 对于F1等指标，值越大越好
            for i, threshold in enumerate(thresholds):
                if value >= threshold:
                    return labels[i]
            return labels[-1]

    @classmethod
    def _calculate_distribution(cls, values: list, thresholds: list, reverse: bool = False) -> Counter:
        """计算分布统计"""
        distribution = Counter()
        for value in values:
            if reverse:
                tier = next((f"tier_{i+1}" for i, t in enumerate(thresholds) if value <= t), "tier_4")
            else:
                tier = next((f"tier_{i+1}" for i, t in enumerate(thresholds) if value >= t), "tier_4")
            distribution[tier] += 1
        return distribution

    @classmethod
    def _format_rank_description(cls, rank: int) -> str:
        """格式化排名描述"""
        if rank in cls.RANK_EMOJIS:
            return cls.RANK_EMOJIS[rank]
        elif rank <= 5:
            return f"⭐ 前五名 (第{rank}名)"
        else:
            return f"第{rank}名"

    @staticmethod
    def generate_ranking_report(rankings, all_profiles, weights) -> Dict[str, Any]:
        """生成完整排名报告"""
        # 排序模型
        sorted_models = sorted(rankings.values(), key=itemgetter("weighted_score"), reverse=True)

        # 生成排名表
        ranking_table = []
        f1_scores, ece_values = [], []

        for i, model_info in enumerate(sorted_models):
            rank = i + 1
            model_name = model_info["model_name"]
            profile = all_profiles[model_name]
            scores = model_info["individual_scores"]
            raw_metrics = scores["raw_metrics"]
            f1_score = raw_metrics["f1_score"]
            calibrated_ece = raw_metrics["calibrated_ece"]

            f1_scores.append(f1_score)
            ece_values.append(calibrated_ece)

            # 生成描述
            performance_tier = ReportManager._get_tier_label(
                f1_score, ReportManager.PERFORMANCE_THRESHOLDS, ReportManager.PERFORMANCE_LABELS
            )
            calibration_level = ReportManager._get_tier_label(
                calibrated_ece, ReportManager.CALIBRATION_THRESHOLDS, ReportManager.CALIBRATION_LABELS, reverse=True
            )

            rank_desc = ReportManager._format_rank_description(rank)

            description_parts = [
                f"{rank_desc} - {model_name}",
                f"性能水平: {performance_tier} (F1: {f1_score:.3f})",
                f"校准质量: {calibration_level} (ECE: {calibrated_ece:.4f})",
            ]
            if rank == 1:
                description_parts.append("🎯 推荐优先使用")

            ranking_table.append(
                {
                    "rank": rank,
                    "model_name": model_name,
                    "weighted_score": round(model_info["weighted_score"], 3),
                    "description": " | ".join(description_parts),
                    "scores": {k: round(v, 2) for k, v in scores.items() if k != "raw_metrics"},
                    "score_breakdown": {k: round(v, 3) for k, v in model_info["score_breakdown"].items()},
                    "raw_metrics": raw_metrics,
                    "calibration_method": profile.get("calibration_parameters", {}).get("method", "unknown"),
                    "overall_rating": profile.get("ratings", {}).get("overall_rating", 0.0),
                }
            )

        # 计算统计信息
        if ranking_table:
            df = pd.DataFrame(ranking_table)
            weighted_scores = df["weighted_score"]
            score_stats = weighted_scores.describe()

            # 计算分布
            performance_dist = ReportManager._calculate_distribution(f1_scores, ReportManager.PERFORMANCE_THRESHOLDS)
            calibration_dist = ReportManager._calculate_distribution(ece_values, ReportManager.CALIBRATION_THRESHOLDS, reverse=True)

            stats = {
                "score_statistics": {
                    "weighted_score": {
                        "mean": float(score_stats["mean"]),
                        "std": float(score_stats["std"]),
                        "min": float(score_stats["min"]),
                        "max": float(score_stats["max"]),
                        "median": float(score_stats["50%"]),
                    },
                },
                "performance_distribution": dict(performance_dist),
                "calibration_distribution": dict(calibration_dist),
                "top_performers": {
                    "top_3_models": df.head(3)["model_name"].tolist(),
                    "best_calibration": min(ranking_table, key=lambda x: x["raw_metrics"]["calibrated_ece"])["model_name"],
                    "best_performance": max(ranking_table, key=lambda x: x["raw_metrics"]["f1_score"])["model_name"],
                },
            }

            # 生成推荐
            recommendations = ReportManager._generate_ranking_recommendations(ranking_table, stats, performance_dist, calibration_dist)
        else:
            stats = {}
            recommendations = []

        return {
            "ranking_summary": {
                "total_models": len(ranking_table),
                "best_model": ranking_table[0]["model_name"] if ranking_table else None,
                "evaluation_date": datetime.now().isoformat(),
                "ranking_method": "weighted_comprehensive_scoring",
            },
            "ranking_table": ranking_table,
            "statistics": stats,
            "recommendations": recommendations,
            "ranking_weights": weights,
        }

    @staticmethod
    def _generate_ranking_recommendations(ranking_table, stats, performance_dist, calibration_dist) -> list:
        """生成排名推荐建议"""
        recommendations = []
        if not ranking_table:
            return recommendations

        best_model = ranking_table[0]
        recommendations.append(
            f"🎯 **主要推荐**: {best_model['model_name']} " f"(综合得分: {best_model['weighted_score']:.2f}) - 综合表现最佳"
        )

        top_performers = stats["top_performers"]
        best_calibration = top_performers["best_calibration"]
        best_performance = top_performers["best_performance"]

        if best_calibration != best_model["model_name"]:
            recommendations.append(f"📐 **校准质量最佳**: {best_calibration} - 适用于对预测置信度要求较高的场景")

        if best_performance != best_model["model_name"]:
            recommendations.append(f"🚀 **性能最佳**: {best_performance} - 适用于对预测准确率要求最高的场景")

        total_models = len(ranking_table)
        excellent_ratio = performance_dist.get("tier_1", 0) / total_models
        poor_calib_ratio = calibration_dist.get("tier_4", 0) / total_models

        if excellent_ratio < 0.3:
            recommendations.append("⚠️ **模型性能提醒**: 高性能模型数量较少，建议考虑模型优化或数据增强")
        if poor_calib_ratio > 0.4:
            recommendations.append("📊 **校准质量提醒**: 多数模型校准效果不佳，建议重新评估校准策略")

        if total_models >= 3:
            top_3 = top_performers["top_3_models"]
            recommendations.append(f"🏆 **Top 3推荐**: {', '.join(top_3)} - 建议重点关注这些模型")

        return recommendations

    @staticmethod
    def generate_summary_report(all_profiles) -> Dict[str, Any]:
        """生成完整汇总报告"""
        if not all_profiles:
            raise ValueError("缺少模型配置文件数据")

        # 提取性能数据
        performance_data = []
        for profile in all_profiles.values():
            basic_metrics = profile.get("performance_metrics", {}).get("basic_metrics", {})
            calibration_validation = profile.get("calibration_validation", {})

            f1 = basic_metrics.get("overall_f1")
            if f1 is not None and not np.isnan(f1):
                performance_data.append(
                    {
                        "f1_score": f1,
                        "calibration_success": calibration_validation.get("calibration_success", False),
                        "low_reliability": calibration_validation.get("low_calibration_reliability", False),
                    }
                )

        # 分析性能分布
        f1_scores = [item["f1_score"] for item in performance_data]
        performance_distribution = {}

        if f1_scores:
            f1_series = pd.Series(f1_scores)
            stats = f1_series.describe()

            # 使用统一的分级方法
            tiers = Counter()
            tier_mapping = {0: "excellent", 1: "good", 2: "fair", 3: "poor"}
            for f1 in f1_scores:
                tier_index = next((i for i, t in enumerate(ReportManager.PERFORMANCE_THRESHOLDS) if f1 >= t), 3)
                tiers[tier_mapping[tier_index]] += 1

            performance_distribution = {
                "f1_statistics": {
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "min": float(stats["min"]),
                    "max": float(stats["max"]),
                    "median": float(stats["50%"]),
                },
                "performance_tiers": dict(tiers),
            }

        # 分析校准质量
        total_models = len(performance_data)
        calibration_success = sum(item["calibration_success"] for item in performance_data)
        low_reliability = sum(item["low_reliability"] for item in performance_data)

        calibration_analysis = {
            "successful_calibrations": calibration_success,
            "calibration_success_rate": calibration_success / total_models if total_models > 0 else 0,
            "low_reliability_count": low_reliability,
            "low_reliability_rate": low_reliability / total_models if total_models > 0 else 0,
        }

        # 生成推荐
        recommendations = ReportManager._generate_summary_recommendations(
            performance_data, f1_scores, total_models, calibration_success, low_reliability
        )

        return {
            "overview": {"total_models": len(all_profiles), "generated_at": datetime.now().isoformat()},
            "performance_distribution": performance_distribution,
            "calibration_analysis": calibration_analysis,
            "recommendations": recommendations,
        }

    @staticmethod
    def _generate_summary_recommendations(performance_data, f1_scores, total_models, calibration_success, low_reliability) -> list:
        """生成汇总推荐建议"""
        recommendations = []

        if not performance_data:
            return ["暂无性能数据可供分析"]

        if calibration_success < total_models * 0.5:
            recommendations.append("建议检查校准数据质量，超过一半的模型校准失败")
        if low_reliability > total_models * 0.3:
            recommendations.append("建议重新评估校准策略，较多模型存在校准可靠性问题")
        if f1_scores and np.std(f1_scores) > 0.2:
            recommendations.append("模型性能差异较大，建议针对性优化低性能模型")

        return recommendations
