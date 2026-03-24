import traceback
import pandas as pd
import multiprocessing as mp

from typing import Dict, Tuple, List
from loguru import logger

from concurrent.futures import ProcessPoolExecutor, as_completed
from ...data_operation.feature_cache_data import FeatureCacheData
from .base_feature_processor import BaseFeatureProcessor, init_worker_extractors, execute_worker
from .base_feature_extractor import BaseFeatureExtractor


class ManualFeatureProcessor(BaseFeatureProcessor):
    """手工特征处理类（支持多进程）"""

    def __init__(
        self,
        batch_size: int,
        data_cache: FeatureCacheData,
        extractors_list: Dict[str, BaseFeatureExtractor] = None,
        cache_dir: str = None,  # 改为使用cache_dir，与基类保持一致
        n_workers: int = None,
    ):
        super().__init__(batch_size=batch_size, extractors_list=extractors_list, cache_dir=cache_dir, n_workers=n_workers)
        self.data_cache = data_cache
        logger.info(f"🎮 初始化手工特征处理器 - Workers: {self.n_workers}")

    def _get_feature_type(self) -> str:
        return "manual"

    def save_to_cache(
        self,
        extracted_features: Dict[str, Dict],
        dataset_name,
        *args,
        **kwargs,
    ):
        """保存到缓存"""
        if not extracted_features:
            return
        try:
            self.data_cache.save_cache(extracted_features, dataset_name)
            logger.info(f"共保存 {len(extracted_features)} 张图像的手工特征到数据库缓存")
        except Exception as e:
            logger.error(f"保存到数据库缓存失败: {e}")

    def load_cached_features(
        self,
        images_path: pd.Series,
        use_cache: bool,
        dataset_name,
        *args,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """加载缓存的特征"""
        cached_features = pd.DataFrame()
        extractor_names = list(self.extractor_configs.keys())
        uncached_images = {path: extractor_names for path in images_path}

        if not use_cache:
            return cached_features, uncached_images

        try:
            cached_features, uncached_images_result = self.data_cache.load_cache(images_path, extractor_names, dataset_name)
            # 转换未缓存图片为字典格式
            if isinstance(uncached_images_result, pd.Series):
                uncached_images = {path: extractor_names for path in uncached_images_result}
            elif isinstance(uncached_images_result, dict):
                uncached_images = uncached_images_result
            elif hasattr(uncached_images_result, "__iter__"):
                uncached_images = {path: extractor_names for path in uncached_images_result}
            else:
                uncached_images = {path: extractor_names for path in images_path}

            logger.info(f"从数据库缓存加载{self.feature_type}特征: {len(cached_features)} 张图像")

        except Exception as e:
            logger.error(f"从数据库缓存加载失败: {e}")

        return cached_features, uncached_images

    def extract_features_impl(self, uncached_images: Dict[str, List[str]], dataset_name, *args, **kwargs) -> Dict[str, Dict]:
        """
        多进程提取手工特征
        Args:
            uncached_images: {image_path: [extractor_name, ...], ...}
        Returns:
            Dict[str, Dict]: 提取的特征结果
        """
        extracted_features = {}
        image_count = len(uncached_images)

        total_features = sum(len(extractors) for extractors in uncached_images.values())
        logger.info(f"开始多进程提取手工特征，共 {image_count} 张图像，{total_features}个特征任务")

        # 创建批次
        batches = self._create_batches(uncached_images)
        self.n_workers = min(self.n_workers, len(batches))
        logger.info(f"📦 手工特征分为 {len(batches)} 个批次，使用 {self.n_workers} 个进程")

        try:
            # 设置多进程启动方法（如果需要）
            if hasattr(mp, "set_start_method"):
                try:
                    mp.set_start_method("spawn", force=True)
                except RuntimeError:
                    pass

            with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=init_worker_extractors,  # 关键：设置初始化函数
                initargs=(self.extractor_configs,),  # 传递初始化参数
            ) as executor:

                # 提交所有任务
                future_to_batch = {}
                for batch in batches:
                    future = executor.submit(execute_worker, batch)
                    future_to_batch[future] = batch

                # 收集结果
                for future in as_completed(future_to_batch):
                    try:
                        result = future.result(timeout=300)
                        extracted_features.update(result["results"])

                        # 定期保存中间结果, 每25个批次保存一次
                        if result["batch_id"] % 25 == 0:
                            self.save_to_cache(extracted_features, dataset_name)
                            extracted_features = {}  # 清空已保存的特征，释放内存
                            logger.info(f"中间保存：已完成 {result['batch_id']}/{len(batches)} 个批次")

                    except Exception as e:
                        batch = future_to_batch[future]
                        batch_size = len(batch["batch_images"])
                        logger.error(f"批次 {batch['batch_id']} 处理失败 (包含 {batch_size} 张图像): {e}")

        except Exception as e:
            logger.error(f"多进程处理失败: {e}")
            logger.error(traceback.format_exc())

        # 最终保存
        if extracted_features:
            logger.info(f"💾 手工特征最终保存: {len(extracted_features)} 张图像的特征")
            self.save_to_cache(extracted_features, dataset_name)

        # 统计结果
        success_rate = (len(extracted_features) / image_count * 100) if image_count > 0 else 0
        logger.info(f"✅ 深度特征提取完成 - 成功: {len(extracted_features)}/{image_count} ({success_rate:.1f}%)")

        return extracted_features
