import numpy as np
import logging

import pandas as pd
from sklearn.metrics import f1_score


class ProfilerFactory:
    """模型画像工厂类 - 负责生成单个和多个模型的画像"""

    def __init__(self, min_samples_for_analysis: int = 50, significance_threshold: float = 0.2):
        """初始化模型画像工厂

        Args:
            min_samples_for_analysis: 分析所需最小样本数
        """
        self.min_samples_for_analysis = min_samples_for_analysis
        self.significance_threshold = significance_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_performance(self, cluster_name: str, predictions_data: pd.DataFrame):
        performance = {}
        for name, data in predictions_data.groupby(cluster_name):
            name = "cluster_" + str(int(name)) if pd.notna(name) else "cluster_nan"
            performance[name] = self.create_model_profile(cluster_name, data)
        return performance

    def generate_rank(self, model_performance):
        """模型对比分析

        Args:
            model_performance: 模型画像字典, 结构如下:
            {
                "overall": {"model_name": {"模型的性能结果": value}},
                "cluster": {
                    "聚类1": {"model_name": {"模型的性能结果": value}},
                    "聚类2": {"model_name": {"模型的性能结果": value}},
                    ...
                    },
            }

        Returns:
            Dict: 模型对比结果字典，包含整体排名和各簇的排名对比结果

        要求：
        给我每个簇Top1模型和Btm1模型, 以F1-score为例, 计算Top1和Btm1的差异
        差异 = (Top1 - Btm1) / Btm1
        如果差异 > significance_threshold, 则认为差异显著, 可以利用该簇进行模型画像说明
        如果差异 <= significance_threshold, 则认为差异不显著, 不进行模型画像说明
        """
        rank_results = {"overall_ranking": {}, "cluster_rankings": {}, "significant_clusters": [], "analysis_summary": {}}

        # 1. 整体模型排名分析
        if "overall" in model_performance and model_performance["overall"]:
            overall_models = model_performance["overall"]
            # 筛选出有效的模型（样本数量足够）
            valid_overall_models = {
                model_name: model_data for model_name, model_data in overall_models.items() if model_data.get("result") == "success"
            }

            if valid_overall_models:
                # 按F1-score排序
                sorted_overall_models = sorted(valid_overall_models.items(), key=lambda x: x[1]["f1_score"], reverse=True)

                rank_results["overall_ranking"] = {
                    "models_by_f1": [
                        {
                            "model_name": model_name,
                            "f1_score": model_data["f1_score"],
                            "accuracy": model_data["accuracy"],
                            "total_samples": model_data["total_samples"],
                        }
                        for model_name, model_data in sorted_overall_models
                    ],
                    "best_model": sorted_overall_models[0][0] if sorted_overall_models else None,
                    "worst_model": sorted_overall_models[-1][0] if sorted_overall_models else None,
                }

        # 2. 各簇模型排名分析
        if "cluster" in model_performance and model_performance["cluster"]:
            for cluster_id, cluster_models in model_performance["cluster"].items():
                # 筛选出有效的模型（样本数量足够）
                valid_cluster_models = {
                    model_name: model_data for model_name, model_data in cluster_models.items() if model_data.get("result") == "success"
                }

                if len(valid_cluster_models) < 2:
                    self.logger.warning(f"簇[{cluster_id}]有效模型数量不足2个，跳过对比分析")
                    continue

                # 按F1-score排序
                sorted_cluster_models = sorted(valid_cluster_models.items(), key=lambda x: x[1]["f1_score"], reverse=True)

                # 获取Top1和Bottom1模型
                top1_model_name, top1_model_data = sorted_cluster_models[0]
                btm1_model_name, btm1_model_data = sorted_cluster_models[-1]

                top1_f1 = top1_model_data["f1_score"]
                btm1_f1 = btm1_model_data["f1_score"]

                # 计算差异（避免除零错误）
                if btm1_f1 > 0:
                    difference_ratio = (top1_f1 - btm1_f1) / btm1_f1
                else:
                    # 如果bottom1的F1为0，直接使用top1的F1作为差异
                    difference_ratio = top1_f1 if top1_f1 > 0 else 0

                # 判断是否显著
                is_significant = difference_ratio > self.significance_threshold

                cluster_analysis = {
                    "cluster_id": cluster_id,
                    "total_valid_models": len(valid_cluster_models),
                    "top1_model": {
                        "name": top1_model_name,
                        "f1_score": top1_f1,
                        "accuracy": top1_model_data["accuracy"],
                        "total_samples": top1_model_data["total_samples"],
                    },
                    "btm1_model": {
                        "name": btm1_model_name,
                        "f1_score": btm1_f1,
                        "accuracy": btm1_model_data["accuracy"],
                        "total_samples": btm1_model_data["total_samples"],
                    },
                    "f1_difference": top1_f1 - btm1_f1,
                    "f1_difference_ratio": difference_ratio,
                    "is_significant": is_significant,
                    "significance_threshold": self.significance_threshold,
                    "all_models_ranking": [
                        {
                            "model_name": model_name,
                            "f1_score": model_data["f1_score"],
                            "accuracy": model_data["accuracy"],
                            "total_samples": model_data["total_samples"],
                        }
                        for model_name, model_data in sorted_cluster_models
                    ],
                }

                rank_results["cluster_rankings"][cluster_id] = cluster_analysis

                # 如果差异显著，加入显著簇列表
                if is_significant:
                    rank_results["significant_clusters"].append(
                        {
                            "cluster_id": cluster_id,
                            "difference_ratio": difference_ratio,
                            "top1_model": top1_model_name,
                            "btm1_model": btm1_model_name,
                            "recommendation": f"簇[{cluster_id}]中模型性能差异显著({difference_ratio:.3f} > {self.significance_threshold})，{top1_model_name}模型更适合处理此类内容",
                        }
                    )

                self.logger.info(
                    f"簇[{cluster_id}]分析完成: Top1({top1_model_name}: {top1_f1:.3f}) vs "
                    f"Btm1({btm1_model_name}: {btm1_f1:.3f}), 差异比例: {difference_ratio:.3f}, "
                    f"显著性: {'是' if is_significant else '否'}"
                )

        # 3. 生成分析摘要
        total_clusters = len(rank_results["cluster_rankings"])
        significant_clusters_count = len(rank_results["significant_clusters"])

        rank_results["analysis_summary"] = {
            "total_clusters_analyzed": total_clusters,
            "significant_clusters_count": significant_clusters_count,
            "significance_rate": significant_clusters_count / total_clusters if total_clusters > 0 else 0,
            "significance_threshold_used": self.significance_threshold,
            "analysis_metric": "f1_score",
            "recommendation_summary": f"在{total_clusters}个分析的簇中，有{significant_clusters_count}个簇显示出显著的模型性能差异，可用于模型画像分析",
        }

        self.logger.info(f"模型排名分析完成: {rank_results['analysis_summary']['recommendation_summary']}")

        return rank_results

    def create_model_profile(self, cluster_name, cluster_data: pd.DataFrame, threshold=0.5):
        # 根据是否传入cluster_name来确定分析类型
        if cluster_name is None:
            clustering_name = "全部数据"
        else:
            clustering_name = cluster_name

        # 计算每个聚类的性能
        if cluster_data is None or cluster_data.empty:
            raise ValueError(f"{clustering_name}数据为空，无法生成画像")

        # 检查必要的列是否存在
        required_columns = ["model_name", "gt_label", "pred_prob", "calibration_prob"]
        missing_columns = [col for col in required_columns if col not in cluster_data.columns]
        if missing_columns:
            raise ValueError(f"{clustering_name}缺少必要的列: {missing_columns}")

        model_results = {}

        for model_name, model_data in cluster_data.groupby("model_name"):

            self.logger.info(f"计算{clustering_name}模型[{model_name}]的模型画像")

            # 提取数据并进行验证
            y_true = model_data["gt_label"].values
            y_pred_proba = model_data["pred_prob"].values
            # y_pred_proba = model_data["calibration_prob"].values

            # 数据验证
            if len(y_true) != len(y_pred_proba):
                raise ValueError(f"{clustering_name}模型[{model_name}]的真实标签和预测概率长度不匹配")

            # 检查标签是否为二分类
            unique_labels = np.unique(y_true)
            if not set(unique_labels).issubset({0, 1}):
                self.logger.warning(f"模型[{model_name}]标签不是二分类格式: {unique_labels}")

            # 检查概率值范围
            if np.any((y_pred_proba < 0) | (y_pred_proba > 1)):
                self.logger.warning(f"模型[{model_name}]预测概率超出[0,1]范围")

            # 计算预测标签
            y_pred = (y_pred_proba > threshold).astype(int)

            # 计算基础统计
            total_samples = len(y_true)
            correct_samples = np.sum(y_true == y_pred)
            positive_samples = np.sum(y_true == 1)
            incorrect_samples = total_samples - correct_samples
            negative_samples = total_samples - positive_samples

            # 计算性能指标
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            # 计算额外的有用指标
            precision = correct_samples / total_samples if total_samples > 0 else 0

            # 检查样本数量是否足够
            if len(model_data) < self.min_samples_for_analysis:
                self.logger.warning(f"{clustering_name}模型[{model_name}]样本数量不足({len(model_data)} < {self.min_samples_for_analysis})")
                results_info = f"样本数量不足({len(model_data)} < {self.min_samples_for_analysis})"
            else:
                results_info = "success"

            model_results[model_name] = {
                "f1_score": round(float(f1), 4),
                "precision": round(float(precision), 4),
                # "accuracy": float(accuracy_score(y_true, y_pred)), # 由于accuracy在不平衡数据上不可靠
                # "total_samples": float(total_samples),
                # "clustering_name": clustering_name,
            }
        return model_results
