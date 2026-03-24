import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import pickle
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

from .base_feature_processor import BaseFeatureProcessor, init_worker_extractors, execute_worker


class DepthFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, batch_size: int, extractors_list=None, cache_dir: str = None, n_workers: int = None):
        # 先调用父类初始化
        super().__init__(batch_size, extractors_list, cache_dir, n_workers)
        logger.info(f"🎮 初始化深度特征处理器 - Workers: {self.n_workers}, GPUs: {len(self.available_gpus) if self.available_gpus else 0}")

    def _get_feature_type(self):
        return "depth"

    def extract_features_impl(self, uncached_images: Dict[str, List[str]], dataset_name, *args, **kwargs) -> Dict[str, Dict]:
        """实际的特征提取实现"""
        image_count = len(uncached_images)
        total_features = sum(len(extractors) for extractors in uncached_images.values())

        gpu_info = f"使用 {len(self.available_gpus)} 个GPU: {self.available_gpus}" if self.available_gpus else "使用CPU"
        logger.info(f"🚀 开始深度特征提取 - {image_count}张图像, {total_features}个特征任务, {gpu_info}")

        # 创建批次
        batches = self._create_batches(uncached_images)
        self.n_workers = min(self.n_workers, len(batches))
        logger.info(
            f"📦 深度特征分为 {len(batches)} 个批次，每批{self.batch_size}张图片, 使用 {self.n_workers} 个进程, 可用GPU数为: {len(self.available_gpus)}"
        )

        extracted_features = {}
        try:
            # 设置多进程启动方法（如果需要）
            if hasattr(mp, "set_start_method"):
                try:
                    mp.set_start_method("spawn", force=True)
                except RuntimeError:
                    pass  # 已经设置过了

            # 为每个worker分配GPU
            worker_gpu_map = {}
            if self.available_gpus:
                for i in range(self.n_workers):
                    worker_gpu_map[i] = self.available_gpus[i % len(self.available_gpus)]
                logger.info(f"Worker-GPU映射: {worker_gpu_map}")

            with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=init_worker_extractors,  # 关键：设置初始化函数
                initargs=(self.extractor_configs, worker_gpu_map),  # 传递初始化参数
            ) as executor:
                # 提交所有任务
                future_to_batch = {}
                for batch in batches:
                    future = executor.submit(execute_worker, batch)
                    future_to_batch[future] = batch

                # 收集结果
                for future in as_completed(future_to_batch):
                    try:
                        result = future.result()
                        extracted_features.update(result["results"])
                        # 定期保存中间结果, 每25个批次保存一次
                        if result["batch_id"] % 25 == 0:
                            self.save_to_cache(extracted_features, dataset_name)
                            logger.info(f"触发保存中间节点, 执行保存: {result['batch_id']}")
                            extracted_features = {}  # 清空已保存的特征，节省内存

                    except Exception as e:
                        batch = future_to_batch[future]
                        logger.error(f"批次 {batch['batch_id']} 处理失败: {e}")

        except Exception as e:
            logger.error(f"多进程处理失败: {e}")
            logger.error(traceback.format_exc())

        # 最终保存
        if extracted_features:
            logger.info(f"💾 深度特征最终保存: {len(extracted_features)} 张图像的特征")
            self.save_to_cache(extracted_features, dataset_name)

        # 统计结果
        success_rate = (len(extracted_features) / image_count * 100) if image_count > 0 else 0
        logger.info(f"✅ 深度特征提取完成 - 成功: {len(extracted_features)}/{image_count} ({success_rate:.1f}%)")

        return extracted_features

    def _load_cache(self, dataset_name: str) -> pd.DataFrame:
        """加载特征缓存"""
        target_column = ["image_path"] + list(self.extractor_configs.keys())

        cache_file = self.cache_dir / f"{dataset_name}_depth_features.pkl"
        if not cache_file.exists():
            logger.info("深度特征缓存文件不存在, 需要全部重新提取")
            return pd.DataFrame(), list(self.extractor_configs.keys())

        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            if isinstance(cache, pd.DataFrame):
                # 检查索引是不是 image_path, 如果是则重置索引
                if cache.index.name == "image_path":
                    cache_pd = cache.reset_index().rename(columns={"index": "image_path"})
                else:
                    cache_pd = cache
            else:
                cache_pd = pd.DataFrame.from_dict(cache, orient="index").reset_index().rename(columns={"index": "image_path"})
            missing_columns = [col for col in target_column if col not in cache_pd.columns]
            existing_columns = [col for col in target_column if col in cache_pd.columns]
            return cache_pd[existing_columns], missing_columns
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return pd.DataFrame(), target_column

    def save_to_cache(self, extracted_features: Dict[str, Dict], dataset_name, has_existed=True, *args, **kwargs):
        """保存特征缓存"""
        if not self.cache_dir or not extracted_features:
            return

        cache_file = self.cache_dir / f"{dataset_name}_depth_features.pkl"

        # 将新特征转换为 DataFrame
        new_features_df = pd.DataFrame.from_dict(extracted_features, orient="index").reset_index().rename(columns={"index": "image_path"})

        # 加载已有缓存
        if has_existed and cache_file.exists():
            existing_cache_df, _ = self._load_cache(dataset_name)
            existing_cache_df = self.merge_features(existing_cache_df, new_features_df)
        else:
            existing_cache_df = new_features_df

        # 保存缓存
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(existing_cache_df, f)
            logger.info(f"💾 保存缓存: {len(existing_cache_df)} 张图像的特征")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def load_cached_features(
        self,
        images_path: pd.Series,
        use_cache: bool,
        dataset_name,
        *args,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """加载缓存的特征"""
        cached_pd = pd.DataFrame()

        if not use_cache:
            valid_feature_names = list(self.extractor_configs.keys())
            uncached_dict = {path: valid_feature_names for path in images_path}
            return cached_pd, uncached_dict

        try:
            cached_pd, missing_columns = self._load_cache(dataset_name)
            valid_feature_names = set(self.extractor_configs.keys())

            # 预先初始化所有路径的字典 - 一次性操作
            missing_columns_set = set(missing_columns) if missing_columns else set()
            logger.debug(f"缺失的列: {missing_columns_set}")
            uncached_dict = {path: missing_columns_set.copy() for path in images_path}
            if cached_pd.empty:
                logger.info("缓存数据为空, 需要全部重新提取")
                return cached_pd, uncached_dict

            # 只保留请求的图片路径的缓存数据
            cached_pd = cached_pd[cached_pd["image_path"].isin(images_path)]

            # 1. 处理完全没有缓存的图片路径
            cached_paths = set(cached_pd["image_path"].unique())
            missing_image_paths = set(images_path) - cached_paths
            if missing_image_paths:
                logger.info(f"未命中缓存的 image_path 数量: {len(missing_image_paths)}")
                # 批量更新缺失路径的特征
                for img_path in missing_image_paths:
                    uncached_dict[img_path] = valid_feature_names

            # 2. 处理有缓存但某些特征为空值的情况
            missing_features_dict = self.get_missing_features_dict(cached_pd)
            for img_path, missing_features in missing_features_dict.items():
                uncached_dict[img_path].update(missing_features)

            # 计算统计信息
            total_requests = len(images_path) * len(valid_feature_names)
            uncached_requests = sum(len(features) for features in uncached_dict.values())
            hit_rate = (total_requests - uncached_requests) / total_requests if total_requests > 0 else 0

            logger.info(
                f"深度特征缓存命中率: {hit_rate:.2%}, "
                f"完全命中: {len([k for k, v in uncached_dict.items() if not v])}, "
                f"需要补充: {len([k for k, v in uncached_dict.items() if v])}"
            )

            # 过滤掉空集合
            uncached_dict = {k: list(v) for k, v in uncached_dict.items() if v}

        except Exception as e:
            logger.error(f"从文件缓存加载失败: {e}")
            import traceback

            logger.error(traceback.format_exc())
            uncached_dict = {path: set(self.extractor_configs.keys()) for path in images_path}

        return cached_pd, uncached_dict

    def get_missing_features_dict(self, cached_pd):
        """并行处理版本（适用于超大数据集）"""
        if cached_pd.empty:
            logger.info("缓存数据为空，跳过缺失特征检查")
            return {}

        logger.info(f"检查已缓存的 {len(cached_pd)} 张图像是否存在缺失特征")
        feature_columns = [col for col in cached_pd.columns if col != "image_path"]
        if not feature_columns:
            logger.warning("没有找到特征列")
            return {}

        def check_row_missing(row_data):
            """检查单行的缺失特征"""
            img_path, *features = row_data
            missing_features = set()

            for i, col in enumerate(feature_columns):
                val = features[i]
                if isinstance(val, (np.ndarray, list)):
                    if len(val) == 0:
                        missing_features.add(col)
                elif pd.isna(val):
                    missing_features.add(col)

            return (img_path, missing_features) if missing_features else None

        # 准备数据
        columns_to_check = ["image_path"] + feature_columns
        data_to_process = cached_pd[columns_to_check].values

        # 可以根据需要启用多进程
        # from concurrent.futures import ProcessPoolExecutor
        # with ProcessPoolExecutor() as executor:
        #     results = list(filter(None, executor.map(check_row_missing, data_to_process)))

        # 单线程版本（通常已经足够快）
        results = list(filter(None, map(check_row_missing, data_to_process)))

        uncached_dict = dict(results)

        for img_path, missing_features in uncached_dict.items():
            logger.debug(f"图片 '{img_path}' 缺失特征: {missing_features}")
        logger.info(f"发现 {len(uncached_dict)} 张图像存在缺失特征")
        return uncached_dict
