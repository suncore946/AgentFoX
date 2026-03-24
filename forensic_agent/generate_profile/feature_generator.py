import os
import pandas as pd
from typing import Dict, Optional, Tuple
import concurrent.futures

from .image_features import ManualFeatureProcessor, DepthFeatureProcessor
from ..data_operation.feature_cache_data import FeatureCacheData
from ..data_operation.database_manager import DatabaseManager
from ..utils.logger import get_logger


class FeatureGenerator:
    """特征管理器，负责协调手工特征和深度特征的提取"""

    def __init__(self, config: Dict, db_dir: Optional[str] = None):
        """初始化特征管理器"""
        self.logger = get_logger(__name__)

        # 配置参数
        self.n_workers = self._calculate_workers(config["num_workers"])
        self.batch_size = max(1, config["batch_size"]) if config["batch_size"] > 0 else 256

        # 初始化数据缓存
        self.db_manager = DatabaseManager(db_dir)
        self.data_cache = FeatureCacheData(self.db_manager)

        # 获取特征提取器并初始化处理器
        self.cache_dir = config["cache_dir"]
        self.depth_processor = DepthFeatureProcessor(
            batch_size=32,
            cache_dir=self.cache_dir,
            n_workers=self.n_workers,
            extractors_list=["CLIP", "CFA", "SRM"],
        )
        self.manual_processor = ManualFeatureProcessor(
            batch_size=256,
            data_cache=self.data_cache,
            n_workers=self.n_workers,
            cache_dir=self.cache_dir,
        )

        # 初始化线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

        self.extractors_desc: Dict[str, str] = {
            name: value["class_description"]
            for processor in [self.manual_processor, self.depth_processor]
            for name, value in processor.extractor_configs.items()
        }
        self.config = config

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保线程池正确关闭"""
        self.close()

    def close(self):
        """关闭线程池"""
        self.logger.info("关闭特征管理器，释放资源")
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

    def _calculate_workers(self, num_workers: Optional[int]) -> int:
        """计算工作进程数"""
        cpu_count = os.cpu_count() or 1
        if num_workers is None or num_workers <= 0:
            return max(1, cpu_count - 1)
        return min(num_workers, cpu_count)

    def run_concurrent(self, images_path: pd.Series, dataset_name, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """并发版本的特征提取"""
        self.logger.info(f"开始并发特征提取: {len(images_path)} 张图像")

        # 使用预创建的线程池并发执行
        manual_future = self.executor.submit(self.manual_processor.process_features, images_path, dataset_name, use_cache)
        depth_future = self.executor.submit(self.depth_processor.process_features, images_path, dataset_name, use_cache)

        # 等待结果
        manual_result_df = manual_future.result()
        depth_result_df = depth_future.result()

        self.logger.info("并发特征提取完成")
        return manual_result_df, depth_result_df

    def run_data(self, data: pd.DataFrame, group_name="dataset_name", use_cache=True):
        manual_futures, depth_futures = [], []
        dataset_info = []  # 记录数据集信息用于错误处理

        for dataset_name, group in data.groupby(group_name):
            unique_images = group["image_path"].drop_duplicates().reset_index(drop=True)

            self.logger.info(f"处理数据集: {dataset_name}, 包含 {len(unique_images)} 张唯一图像")

            # 提交任务时添加数据集名称用于错误追踪
            # manual_future = self.executor.submit(self.manual_processor.process_features, unique_images, dataset_name, use_cache)
            depth_future = self.executor.submit(self.depth_processor.process_features, unique_images, dataset_name, use_cache)
            # manual_futures.append(manual_future)
            depth_futures.append(depth_future)
            dataset_info.append(dataset_name)

        manual_results, depth_results = [], []
        for i, (manual_future, depth_future) in enumerate(zip(manual_futures, depth_futures)):
            try:
                # manual_result_df = manual_future.result()
                depth_result_df = depth_future.result()

                # manual_results.append(manual_result_df)
                depth_results.append(depth_result_df)

                self.logger.info(f"数据集 {dataset_info[i]} 处理完成")

            except Exception as e:
                self.logger.error(f"数据集 {dataset_info[i]} 处理失败: {str(e)}")

        return manual_results, depth_results

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_requests = self.data_cache.stats["cache_hits"] + self.data_cache.stats["cache_misses"]
        cache_rate = self.data_cache.stats["cache_hits"] / max(1, total_requests) if total_requests > 0 else 0

        db_stats = self.data_cache.get_db_stats()

        return {
            "cache_hit_rate": cache_rate,
            "registered_manual_features": len(self.manual_processor.extractor_configs),
            "registered_depth_features": len(self.depth_processor.extractor_configs),
            "db_stats": db_stats,
        }
