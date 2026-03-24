from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from ..generate_profile.clustering.clustering_dataclass import ClusteringInfo

from ..utils.logger import get_logger, setup_logging

import pandas as pd

from .profiler.profiler_factory import ProfilerFactory
from ..visualization.model_profile_visualizer import ModelPerformanceVisualizer

from .profiler.profiler_dataclass import ModelProfiles
from ..processor.clustering_profile_processor import ClusteringProfilesProcessor
from ..configs.config_dataclass import ProfileConfig


class ProfileGenerator:
    """模型画像生成器"""

    def __init__(self, config: Dict[str, Any] = None):
        """初始化模型画像生成器

        Args:
            performance_threshold: 性能差异阈值
            significance_threshold: 显著差异阈值
            min_samples_for_analysis: 分析所需最小样本数
        """
        self.logger = get_logger(__name__)
        self.config: ProfileConfig = ProfileConfig(**config) if config else ProfileConfig()
        self.performance_threshold = self.config.performance_threshold
        self.significance_threshold = self.config.significance_threshold
        self.min_samples_for_analysis = self.config.min_samples_for_analysis

        # 初始化模型画像工厂
        self.profile_generator = ProfilerFactory(
            min_samples_for_analysis=self.min_samples_for_analysis,
            significance_threshold=self.significance_threshold,
        )
        self.visualizer = ModelPerformanceVisualizer()

        # 读取json文件，加载模型信息
        with open(self.config.model_profile, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.model_profiles = ModelProfiles(**content)

        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clustering_profile = ClusteringProfilesProcessor(self.config.llm)
        self.max_workers = self.config.max_workers

    def _process_all_clustering(self, clustering_infos: List[ClusteringInfo]):
        self.logger.info("开始生成整体聚类分析报告")
        desc = []
        for item in clustering_infos:
            desc.append(
                {
                    "clustering_name": item.clustering_columns,
                    "clustering_description": item.clustering_desc,
                    "clustering_quality_metrics": item.quality_metrics,
                    "clustering_ability": item.clustering_ability,
                    "clustering_discovery_stats": item.discovery_stats,
                }
            )
        clustering_infos = self.clustering_profile.clustering_profile(desc).model_dump()
        return clustering_infos

    def _process_single_clustering(self, item: ClusteringInfo, cluster_data: pd.DataFrame, overall_performance: Dict) -> tuple:
        """处理单个聚类结果（用于多线程）

        Args:
            item: 单个聚类结果
            cluster_data: 聚类数据
            overall_performance: 整体性能数据

        Returns:
            tuple: (clustering_columns, analysis_result, performance_result)
        """
        try:
            # 生成性能数据
            self.logger.info(f"开始处理聚类: {item.clustering_columns}")
            cluster_performance = self.profile_generator.generate_performance(item.clustering_columns, cluster_data)
            if len(item.clustering_desc) == 1:
                clustering_description = item.clustering_desc[item.clustering_columns]
            else:
                clustering_description = item.clustering_desc

            performance_result = {
                "clustering_name": item.clustering_columns,
                "clustering_description": item.clustering_desc,
                "clustering_quality_metrics": item.quality_metrics,
                "clustering_ability": item.clustering_ability,
                "clustering_discovery_stats": item.discovery_stats,
                "forensic_expert_cluster_performance": cluster_performance,
            }

            cluster_analysis = self.clustering_profile.run(
                clustering_name=item.clustering_columns,
                clustering_description=clustering_description,
                performance={"cluster_performance": cluster_performance, "overall_performance": overall_performance},
                model_profiles=self.model_profiles.profiles,
            )
            return (item.clustering_columns, cluster_analysis, performance_result)

        except Exception as e:
            self.logger.exception(e)
            return (item.clustering_columns, None, None)

    def run(self, clustering_result: List[ClusteringInfo], clustering_data: pd.DataFrame):
        """生成模型画像报告

        Args:
            discovery_result: 内容发现结果（来自差异驱动聚类器）
            model_predictions: 模型预测结果

        Returns:
            ModelPortraitReport: 完整的模型画像报告
        """
        dataset_names = clustering_data["dataset_name"].drop_duplicates().to_list()
        dataset_names = "-".join(dataset_names)
        self.logger.info(f"开始生成聚类画像报告: {dataset_names}")
        # 生成整体性能数据
        overall_performance = self.profile_generator.create_model_profile(cluster_name=None, cluster_data=clustering_data)
        all_performance = {}
        all_performance["overall_performance"] = {
            "description": "In a non-clustered, the overall performance of models",
            "performance": overall_performance,
        }

        all_analysis = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(self._process_single_clustering, item, clustering_data, overall_performance): item
                for item in clustering_result
            }

            # 收集结果
            for future in as_completed(future_to_item):
                clustering_columns, analysis_result, performance_result = future.result()
                if analysis_result is not None and performance_result is not None:
                    all_analysis[clustering_columns] = analysis_result
                    all_performance[clustering_columns] = performance_result
                    self.logger.info(f"聚类 {clustering_columns} 处理完成")

            clustering_analysis = self._process_all_clustering(clustering_result)
            all_analysis["overall_analysis"] = clustering_analysis

        output_file = self.output_dir / f"{dataset_names}_performance_profiles.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_performance, f, ensure_ascii=False, indent=4)
        self.logger.info(f"画像性能报告已保存至 {output_file}")

        output_file = self.output_dir / f"{dataset_names}_analysis_profiles.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_analysis, f, ensure_ascii=False, indent=4)
        self.logger.info(f"画像分析报告已保存至 {output_file}")
