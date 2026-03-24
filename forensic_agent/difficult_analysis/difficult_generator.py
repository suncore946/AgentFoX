"""
难度分析主入口模块
提供统一的难度度量分析接口
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path


from .difficulty_calculator import DifficultyCalculator
from .noise_detector import NoiseDetector
from .irt_difficulty import IRTDifficultyEstimator
from ..data_operation.dataset_loader import load_project_data
from ..utils.difficult_utils import validate_prediction_matrix
from ..configs.config_dataclass import DifficultyConfig


class DifficultyGenerator:
    """难度分析器 - 统一接口"""

    def __init__(self, config: Dict = None):
        """
        初始化难度分析器

        Args:
            config: 配置对象或配置名称
        """
        self.config = DifficultyConfig(**config) if isinstance(config, dict) else DifficultyConfig()

        # 初始化各个组件
        self.difficulty_calculator = DifficultyCalculator(self.config)
        self.noise_detector = NoiseDetector(self.config)
        self.irt_estimator = IRTDifficultyEstimator(self.config)

        # 存储分析结果
        self.results = {}
        self.data_info = {}

    def analyze_from_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        image_ids: Optional[np.ndarray] = None,
        include_irt: bool = True,
    ) -> Dict[str, Any]:
        """
        从预测矩阵进行难度分析

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)
            labels: 真实标签 (n_samples,)
            image_ids: 图像ID数组，可选
            include_irt: 是否包含IRT分析

        Returns:
            完整的难度分析结果
        """
        # 输入验证
        is_valid, error_msg = validate_prediction_matrix(predictions, labels)
        if not is_valid:
            raise ValueError(f"输入数据验证失败: {error_msg}")

        n_samples, n_models = predictions.shape

        if image_ids is None:
            image_ids = np.arange(n_samples)
        elif len(image_ids) != n_samples:
            raise ValueError(f"image_ids长度({len(image_ids)})与样本数量({n_samples})不匹配")

        print(f"开始难度分析: {n_samples} 个样本, {n_models} 个模型")

        # 1. 核心难度计算
        print("正在计算基础难度指标...")
        difficulty_results = self.difficulty_calculator.run(predictions, labels)

        # 2. 噪声检测
        print("正在进行噪声检测...")
        noise_results = self.noise_detector.run(
            predictions=predictions,
            labels=labels,
            uncertainty_metrics={
                "mean_probs": difficulty_results["mean_probs"],
                "epistemic_uncertainty": difficulty_results["epistemic_uncertainty"],
                "aleatoric_uncertainty": difficulty_results["aleatoric_uncertainty"],
                "total_uncertainty": difficulty_results["total_uncertainty"],
                "mean_entropy": difficulty_results["mean_entropy"],
            },
            disagreement_metrics={
                "disagree_rate": difficulty_results["disagree_rate"],
                "prob_variance": difficulty_results["prob_variance"],
                "prob_std": difficulty_results["prob_std"],
            },
            difficulty_metrics={"base_difficulty": difficulty_results["base_difficulty"]},
        )

        # 3. IRT分析（可选）
        irt_results = None
        if include_irt:
            print("正在进行IRT难度分析...")
            try:
                irt_results = self.irt_estimator.fit(predictions, labels)

                # IRT与其他难度的一致性检验
                consistency_check = self.irt_estimator.compare_with_other_difficulty(
                    irt_results,
                    difficulty_results["final_difficulty_normalized"],
                    "final_difficulty_normalized",
                )
                irt_results["consistency_check"] = consistency_check

            except Exception as e:
                warnings.warn(f"IRT分析失败: {e}")
                irt_results = None

        # 4. 综合结果
        print("整合分析结果...")
        comprehensive_results = self._integrate_results(difficulty_results, noise_results, irt_results, image_ids)

        # 5. 存储结果
        self.results = comprehensive_results
        self.data_info = {"n_samples": n_samples, "n_models": n_models, "has_irt": irt_results is not None}

        print("难度分析完成!")
        return comprehensive_results

    def analyze_from_database(
        self,
        dataset_name: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        db_path: str = "data/predictions.db",
        include_irt: bool = True,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        从数据库加载数据并进行难度分析

        Args:
            dataset_name: 数据集名称过滤
            model_names: 模型名称列表过滤
            data_source: 数据源过滤
            db_path: 数据库路径
            include_irt: 是否包含IRT分析
            max_samples: 最大样本数量限制

        Returns:
            完整的难度分析结果
        """
        # 加载数据
        print("正在从数据库加载数据...")
        data, loader = load_project_data(db_dir=db_path, model_names=model_names, dataset_names=dataset_name)
        if data.empty:
            raise ValueError("加载的数据为空")
        print(f"加载了 {len(data)} 条预测记录")

        # 转换为多模型预测矩阵
        predictions, labels, image_ids = self._prepare_prediction_matrix(data, max_samples)

        # 进行分析
        return self.analyze_from_predictions(predictions, labels, image_ids, include_irt)

    def _prepare_prediction_matrix(
        self,
        data: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从数据库数据准备预测矩阵

        Args:
            data: 数据库数据
            max_samples: 最大样本数量

        Returns:
            (预测矩阵, 标签, 图像ID)
        """
        # 选择概率列
        prob_column = (
            self.config.calibrated_prob_column
            if self.config.use_calibrated_probs and self.config.calibrated_prob_column in data.columns
            else self.config.raw_prob_column
        )

        # 创建透视表：image_id x model_name
        pivot_data = data.pivot_table(index="image_id", columns="model_name", values=prob_column, aggfunc="first")

        # 只保留有所有模型预测的样本
        complete_samples = pivot_data.dropna()

        if len(complete_samples) == 0:
            raise ValueError("没有找到所有模型都有预测的样本")

        # 应用样本数量限制
        if max_samples and len(complete_samples) > max_samples:
            complete_samples = complete_samples.sample(n=max_samples, random_state=42)

        # 提取预测矩阵
        predictions = complete_samples.values

        # 获取对应的标签
        labels_data = data.drop_duplicates("image_id").set_index("image_id")
        labels = labels_data.loc[complete_samples.index, "gt_label"].values

        # 图像ID
        image_ids = complete_samples.index.values

        print(f"准备了 {len(predictions)} 个样本, {predictions.shape[1]} 个模型的预测矩阵")

        return predictions, labels, image_ids

    def _integrate_results(
        self,
        difficulty_results: Dict[str, Any],
        noise_results: Dict[str, Any],
        irt_results: Optional[Dict[str, Any]],
        image_ids: np.ndarray,
    ) -> Dict[str, Any]:
        """
        整合各组件的分析结果

        Args:
            difficulty_results: 难度计算结果
            noise_results: 噪声检测结果
            irt_results: IRT分析结果（可选）
            image_ids: 图像ID数组

        Returns:
            综合分析结果
        """
        # 创建综合DataFrame
        df = self.difficulty_calculator.create_difficulty_dataframe(difficulty_results, image_ids)

        # 添加风险信息
        risk_df = self.noise_detector.create_risk_dataframe(noise_results, image_ids)
        df = df.merge(risk_df, on="image_id", how="left")

        # 添加IRT信息（如果有）
        if irt_results:
            irt_df = self.irt_estimator.create_irt_dataframe(irt_results, image_ids)
            df = df.merge(irt_df, on="image_id", how="left")

        # 综合结果字典
        integrated_results = {
            "dataframe": df,
            "summary": self._create_analysis_summary(difficulty_results, noise_results, irt_results),
            "difficulty_results": difficulty_results,
            "noise_results": noise_results,
            "irt_results": irt_results,
            "sample_recommendations": self._generate_sample_recommendations(noise_results, df),
        }

        return integrated_results

    def _create_analysis_summary(
        self, difficulty_results: Dict[str, Any], noise_results: Dict[str, Any], irt_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """创建分析摘要"""

        summary = {
            "total_samples": difficulty_results["n_samples"],
            "n_models": difficulty_results["n_models"],
            "difficulty_statistics": difficulty_results["statistics"],
            "risk_statistics": noise_results["statistics"],
        }

        if irt_results:
            summary["irt_statistics"] = {
                "converged": irt_results["fit_info"]["converged"],
                "filtered_samples": irt_results["filtered_samples"],
                "quality_metrics": irt_results.get("quality_metrics", {}),
                "consistency_check": irt_results.get("consistency_check", {}),
            }

        return summary

    def _generate_sample_recommendations(self, noise_results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """生成样本使用建议"""

        # 获取各类样本
        high_value_samples = self.noise_detector.get_high_value_samples(noise_results)
        suspicious_samples = self.noise_detector.get_suspicious_samples(noise_results)

        # 按难度排序的推荐样本
        recommended_indices = high_value_samples["recommended_for_training"]
        if len(recommended_indices) > 0:
            recommended_df = df.iloc[recommended_indices].copy()
            # 按最终难度降序排列
            recommended_df = recommended_df.sort_values("final_difficulty_normalized", ascending=False)
        else:
            recommended_df = pd.DataFrame()

        return {
            "high_value_samples": high_value_samples,
            "suspicious_samples": suspicious_samples,
            "recommended_for_training": recommended_df,
            "training_recommendations": {
                "n_recommended": len(recommended_indices),
                "n_high_value": len(high_value_samples["hard_valuable"]),
                "n_suspicious": len(suspicious_samples["all_suspicious"]),
                "training_sample_ratio": len(recommended_indices) / len(df) if len(df) > 0 else 0,
            },
        }

    def get_top_difficult_samples(
        self,
        n: int = 100,
        difficulty_key: str = "final_difficulty_normalized",
        exclude_suspicious: bool = True,
    ) -> pd.DataFrame:
        """
        获取最难的n个样本

        Args:
            n: 返回样本数量
            difficulty_key: 难度指标键名
            exclude_suspicious: 是否排除可疑样本

        Returns:
            最难样本的DataFrame
        """
        if not self.results:
            raise ValueError("请先运行分析")

        df = self.results["dataframe"].copy()

        # 排除可疑样本（可选）
        if exclude_suspicious:
            suspicious_types = ["label_noise", "ood_shift", "multiple_risk"]
            mask = ~df["risk_type"].isin(suspicious_types)
            df = df[mask]

        # 按难度排序
        df_sorted = df.sort_values(difficulty_key, ascending=False)

        return df_sorted.head(n)

    def export_results(self, output_dir: Union[str, Path], export_formats: List[str] = ["csv", "json"]) -> Dict[str, str]:
        """
        导出分析结果

        Args:
            output_dir: 输出目录
            export_formats: 导出格式列表

        Returns:
            导出文件路径字典
        """
        if not self.results:
            raise ValueError("请先运行分析")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 导出主要结果DataFrame
        df = self.results["dataframe"]
        version_tag = self.config.get_version_tag()

        for fmt in export_formats:
            if fmt == "csv":
                filepath = output_dir / f"difficulty_analysis_{version_tag}.csv"
                df.to_csv(filepath, index=False)
                exported_files["main_csv"] = str(filepath)

            elif fmt == "json":
                filepath = output_dir / f"difficulty_analysis_{version_tag}.json"
                df.to_json(filepath, orient="records", indent=2)
                exported_files["main_json"] = str(filepath)

            elif fmt == "parquet":
                filepath = output_dir / f"difficulty_analysis_{version_tag}.parquet"
                df.to_parquet(filepath, index=False)
                exported_files["main_parquet"] = str(filepath)

        # 导出摘要
        summary_path = output_dir / f"difficulty_summary_{version_tag}.json"
        import json

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results["summary"], f, indent=2, ensure_ascii=False, default=str)
        exported_files["summary_json"] = str(summary_path)

        # 导出推荐样本
        recommendations = self.results["sample_recommendations"]
        if len(recommendations["recommended_for_training"]) > 0:
            rec_path = output_dir / f"recommended_samples_{version_tag}.csv"
            recommendations["recommended_for_training"].to_csv(rec_path, index=False)
            exported_files["recommendations_csv"] = str(rec_path)

        return exported_files

    def print_analysis_summary(self):
        """打印分析结果摘要"""
        if not self.results:
            raise ValueError("请先运行分析")

        print("=" * 60)
        print("           难度分析结果摘要")
        print("=" * 60)

        # 基本信息
        info = self.data_info
        print(f"样本数量: {info['n_samples']}")
        print(f"模型数量: {info['n_models']}")
        print(f"包含IRT分析: {'是' if info['has_irt'] else '否'}")
        print(f"配置版本: {info['version_tag']}")

        # 难度统计
        print(f"\n{'='*20} 难度统计 {'='*20}")
        self.difficulty_calculator.print_summary(self.results["difficulty_results"])

        # 风险检测统计
        print(f"\n{'='*20} 风险检测 {'='*20}")
        self.noise_detector.print_risk_summary(self.results["noise_results"])

        # IRT统计（如果有）
        if self.results["irt_results"]:
            print(f"\n{'='*20} IRT分析 {'='*20}")
            self.irt_estimator.print_irt_summary(self.results["irt_results"])

        # 样本推荐
        print(f"\n{'='*20} 样本推荐 {'='*20}")
        rec = self.results["sample_recommendations"]["training_recommendations"]
        print(f"推荐训练样本: {rec['n_recommended']} ({rec['training_sample_ratio']:.1%})")
        print(f"高价值难样本: {rec['n_high_value']}")
        print(f"可疑样本: {rec['n_suspicious']}")

        print("=" * 60)


# 便捷函数
def analyze_difficulty_from_database(
    dataset_name: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    db_path: str = "data/predictions.db",
    config: str = "default",
    include_irt: bool = True,
    max_samples: Optional[int] = None,
    print_summary: bool = True,
) -> DifficultyGenerator:
    """
    便捷函数：从数据库进行难度分析

    Args:
        dataset_name: 数据集名称过滤
        model_names: 模型名称列表
        data_source: 数据源过滤
        db_path: 数据库路径
        config: 配置名称
        include_irt: 是否包含IRT分析
        max_samples: 最大样本数量
        print_summary: 是否打印摘要

    Returns:
        配置好的DifficultyAnalyzer实例
    """
    analyzer = DifficultyGenerator(config)

    analyzer.analyze_from_database(
        dataset_name=dataset_name,
        model_names=model_names,
        db_path=db_path,
        include_irt=include_irt,
        max_samples=max_samples,
    )

    if print_summary:
        analyzer.print_analysis_summary()

    return analyzer
