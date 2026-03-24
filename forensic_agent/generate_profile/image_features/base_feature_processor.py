from collections import defaultdict
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Type, Tuple
from abc import ABC, abstractmethod
import os
import re
import importlib
import inspect

from loguru import logger
import multiprocessing as mp
import importlib

from .base_feature_extractor import BaseFeatureExtractor
from .image_preloader import ImagePreloader
from ...utils import get_available_gpus

_EXTRACTOR_GLOBAL_STORE = {}


def _get_extractor_name(extractor_class: Type, class_name: str) -> str:
    """获取提取器名称，优先从类属性获取，否则根据类名生成"""
    try:
        # 尝试从类属性获取默认名称
        if hasattr(extractor_class, "default_name"):
            return extractor_class.default_name
        elif hasattr(extractor_class, "name"):
            return extractor_class.name
        else:
            # 根据类名生成名称（去掉Extractor后缀并转换为小写下划线格式）
            name = class_name.replace("Extractor", "")
            # 将驼峰命名转换为下划线命名
            name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
            return name.upper()
    except Exception as e:
        logger.warning(f"Failed to get name for {class_name}: {e}")
        return class_name.upper().replace("extractor", "")


def _get_init_params(extractor_class: Type) -> Dict[str, Any]:
    """分析提取器构造函数参数并返回默认参数字典"""
    try:
        sig = inspect.signature(extractor_class.__init__)
        params = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # 跳过 name 参数，这个会在实例化时单独处理
            if param_name == "name":
                continue

            if param_name == "device_id":
                # device_id 将在子进程中设置，这里不设置默认值
                pass

            # 如果有默认值，使用默认值
            if param.default != inspect.Parameter.empty:
                params[param_name] = param.default
            # 对于一些常见参数，提供合理的默认值
            elif param_name == "config":
                params[param_name] = {}
            # 其他必需参数暂时设为None，可能需要在具体使用时配置
            elif param.annotation != inspect.Parameter.empty:
                # 根据类型注解设置默认值
                if param.annotation == str:
                    params[param_name] = ""
                elif param.annotation == int:
                    params[param_name] = 0
                elif param.annotation == bool:
                    params[param_name] = False
                elif param.annotation == dict:
                    params[param_name] = {}
                elif param.annotation == list:
                    params[param_name] = []

        return params

    except Exception as e:
        logger.warning(f"Failed to analyze init params for {extractor_class}: {e}")
        return {}


def discover_extractors(
    package_name: str,
    base_class: Type = BaseFeatureExtractor,
    special_extractor: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """自动发现提取器类并返回实例化配置
    Args:
        package_name: 包名
        base_class: 基类类型
        special_extractor: 指定要加载的提取器名称列表，如果为None则加载所有
    Returns:
        提取器配置字典，key为提取器名称，value为包含实例化信息的配置字典
        配置字典格式：
        {
            "class_name": str,      # 类名
            "class_module": str,    # 模块路径
            "params": dict,         # 实例化参数
            "extractor_name": str   # 提取器名称
        }
    """

    extractor_configs = {}

    # 如果指定了special_extractor，转换为set以提高查找效率
    target_extractors = set(special_extractor) if special_extractor else None

    try:
        # 构建包的导入路径
        current_package = __name__.rsplit(".", 1)[0]  # 获取 'model_profile.features'
        full_package_name = f"{current_package}.{package_name}"

        # 获取子包的物理路径
        current_dir = os.path.dirname(__file__)
        package_dir = os.path.join(current_dir, package_name)

        if not os.path.exists(package_dir):
            logger.warning(f"Package directory not found: {package_dir}")
            return extractor_configs

        logger.debug(f"Scanning package: {full_package_name} at {package_dir}")

        # 如果指定了特定提取器，记录目标信息
        if target_extractors:
            logger.debug(f"Looking for specific extractors: {target_extractors}")

        # 遍历目录中的Python文件
        for filename in os.listdir(package_dir):
            if filename.endswith("extractor.py") and not filename.startswith("__"):
                module_name = filename[:-3]  # 去掉.py后缀
                module_path = f"{full_package_name}.{module_name}"

                try:
                    # 使用importlib动态导入模块，使用相对导入
                    module = importlib.import_module(module_path, package=__name__)

                    # 查找以Extractor结尾且继承自BaseFeatureExtractor的类
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if name.endswith("Extractor") and issubclass(obj, base_class) and obj != base_class and not inspect.isabstract(obj):

                            try:
                                # 获取提取器的默认名称
                                extractor_name = _get_extractor_name(obj, name)

                                # 检查是否需要加载此提取器
                                if target_extractors is None or extractor_name in target_extractors:
                                    # 获取构造函数参数
                                    init_params = _get_init_params(obj)

                                    # 构建配置字典
                                    config = {
                                        "class_name": name,
                                        "class_description": getattr(obj, "description", ""),
                                        "class_module": module_path,
                                        "params": init_params,
                                    }

                                    extractor_configs[extractor_name] = config
                                    logger.success(f"Successfully discovered {name} ('{extractor_name}') from {module_path}")

                                    # 如果指定了特定提取器，从目标集合中移除已找到的
                                    if target_extractors is not None:
                                        target_extractors.discard(extractor_name)
                                else:
                                    logger.debug(f"Skipped {name} ('{extractor_name}') - not in target list")

                            except Exception as e:
                                logger.warning(f"Failed to instantiate {name} from {module_path}: {e}")

                except ImportError as e:
                    logger.warning(f"Failed to import module {module_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing module {module_path}: {e}")

        # 检查是否有指定的提取器未找到
        if target_extractors and len(target_extractors) > 0:
            logger.error(f"The following specified extractors were not found: {target_extractors}")

    except Exception as e:
        logger.error(f"Error in discover_extractors: {e}")

    logger.info(f"Total extractors discovered: {len(extractor_configs )}")
    return extractor_configs


def init_worker_extractors(extractors_config: Dict[str, Any], gpu_map: int = None):
    """进程池初始化函数，在每个子进程中初始化特征提取器"""

    global _EXTRACTOR_GLOBAL_STORE
    pid = os.getpid()

    # 实例化多线程数据加载器
    if "image_preloader" not in _EXTRACTOR_GLOBAL_STORE:
        _EXTRACTOR_GLOBAL_STORE["image_preloader"] = ImagePreloader()

    for extractor_name, config in extractors_config.items():
        if extractor_name in _EXTRACTOR_GLOBAL_STORE:
            continue  # 已经初始化

        try:
            module = importlib.import_module(config["class_module"])
            extractor_class = getattr(module, config["class_name"])
            extractor_params = config.get("params", {}).copy()

            # 改进的GPU分配逻辑
            if gpu_map is not None and gpu_map:
                # 使用worker_id作为键，如果没有则使用PID作为fallback
                worker_id = getattr(mp.current_process(), "_identity", [0])[0] if hasattr(mp.current_process(), "_identity") else 0

                # 更健壮的GPU分配策略
                if worker_id in gpu_map:
                    device_id = gpu_map[worker_id]
                else:
                    # Fallback: 使用PID取模
                    available_gpus = list(gpu_map.values())
                    device_id = available_gpus[pid % len(available_gpus)]
                extractor_params["device_id"] = device_id
                logger.debug(f"进程 {pid} (Worker {worker_id}) 为提取器 {extractor_name} 分配GPU {device_id}")
            else:
                logger.debug(f"进程 {pid} 使用CPU模式初始化提取器 {extractor_name}")

            # 实例化提取器
            _EXTRACTOR_GLOBAL_STORE[extractor_name] = extractor_class(**extractor_params)

            logger.debug(f"进程 {pid} 成功初始化深度特征提取器: {extractor_name}")

        except Exception as e:
            logger.error(f"进程 {pid} 初始化深度特征提取器 {extractor_name} 失败: {e}")
            raise


def execute_worker(batch_info: Dict[str, Any]) -> Dict[str, Any]:
    global _EXTRACTOR_GLOBAL_STORE

    batch_id = batch_info["batch_id"]

    pid = os.getpid()
    logger.debug(f"Worker进程 {pid} 开始处理批次 {batch_id}")

    processed_count = 0
    failed_count = 0

    # 多线程载入图像
    if "image_preloader" not in _EXTRACTOR_GLOBAL_STORE:
        _EXTRACTOR_GLOBAL_STORE["image_preloader"] = ImagePreloader()
    images_dict = _EXTRACTOR_GLOBAL_STORE["image_preloader"].load_images_batch(batch_info["image_paths"])
    features = defaultdict(dict)
    try:
        extractor_names = list(batch_info["extractors_needed"].keys())
        logger.debug(f"批次 {batch_id} 载入图像完成，开始提取{extractor_names}特征")
        for extractor_name, image_paths in batch_info["extractors_needed"].items():
            extractor = _EXTRACTOR_GLOBAL_STORE.get(extractor_name)
            if not extractor:
                logger.warning(f"提取器 {extractor_name} 不存在")
                continue

            # 提取特征
            image_info = [images_dict.get(path) for path in image_paths]
            if not image_info:
                logger.warning(f"批次 {batch_id} 中没有有效图像供提取器 {extractor_name} 使用")
                continue

            feature_values = extractor.extract(image_info)
            for img_path, feature in zip(image_paths, feature_values):
                features[img_path][extractor_name] = feature
            processed_count += len(image_paths)

    except Exception as e:
        failed_count += len(image_paths)
        logger.error(f"批次 {batch_id} 处理失败: {e}")

    logger.info(f"Worker进程 {pid} 完成批次 {batch_id} - 成功: {processed_count}, 失败: {failed_count}")

    return {
        "results": features,
        "batch_id": batch_id,
        "processed_count": processed_count,
        "failed_count": failed_count,
    }


class BaseFeatureProcessor(ABC):
    """特征处理基类"""

    def __init__(self, batch_size: int, extractors_list: None, cache_dir: str = None, n_workers: int = 8):
        # 准备提取器配置
        self.available_gpus = get_available_gpus()
        self.n_workers = n_workers or max(1, min(mp.cpu_count() // 2, len(self.available_gpus)))
        self.batch_size = batch_size
        self.feature_type = self._get_feature_type()

        # 设置缓存目录
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 如果没有提供extractors_dict，则自动发现
        self.extractors_list = extractors_list
        self.extractor_configs = self._auto_discover_extractors()

    @abstractmethod
    def _get_feature_type(self) -> str:
        """获取特征类型标识"""
        pass

    @abstractmethod
    def extract_features_impl(self, uncached_images: pd.Series) -> Dict[str, Dict]:
        """具体的特征提取实现"""
        pass

    @abstractmethod
    def load_cached_features(
        self,
        images_path: pd.Series,
        use_cache: bool,
        dataset_name: str,
        *args,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载缓存的特征，返回已缓存特征和未缓存图像的DataFrame"""
        pass

    @abstractmethod
    def save_to_cache(self, extracted_features: Dict[str, Dict]):
        """保存到缓存 - 基类实现使用数据库缓存"""
        pass

    def _auto_discover_extractors(self) -> Dict[str, BaseFeatureExtractor]:
        """自动发现提取器"""
        extractor_dir = self._get_feature_type()
        extractor_configs = discover_extractors(extractor_dir, special_extractor=self.extractors_list)

        if extractor_configs:
            logger.info(f"自动发现{self.feature_type}提取器: {list(extractor_configs.keys())}")
        else:
            logger.warning(f"未找到{self.feature_type}提取器在目录: {extractor_dir}")

        return extractor_configs

    def _create_batches(self, uncached_images: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """创建批次"""
        image_paths = list(uncached_images.keys())
        batches = []

        for batch_id, start in enumerate(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[start : start + self.batch_size]

            # 调整 extractors_needed 的数据结构
            extractors_needed = {}
            for path in batch_paths:
                for extractor_name in uncached_images[path]:
                    if extractor_name not in extractors_needed:
                        extractors_needed[extractor_name] = []
                    extractors_needed[extractor_name].append(path)
            batches.append(
                {
                    "image_paths": batch_paths,
                    "extractors_needed": extractors_needed,
                    "batch_id": batch_id + 1,
                }
            )
        return batches

    def _prepare_extractors_config(self):
        """准备提取器配置，用于子进程初始化"""
        self.extractors_config = {}

        if not hasattr(self, "extractors_dict") or not self.extractor_configs:
            logger.warning("未找到extractors_dict，跳过配置准备")
            return

        for name, extractor in self.extractor_configs.items():
            try:
                # 获取提取器的类和模块信息
                extractor_class = extractor.__class__
                class_module = extractor_class.__module__
                class_name = extractor_class.__name__

                # 尝试多种方式获取初始化参数
                init_params = {}

                # 优先从_init_params获取
                if hasattr(extractor, "_init_params") and isinstance(extractor._init_params, dict):
                    init_params = extractor._init_params.copy()
                # 然后从config获取
                elif hasattr(extractor, "config") and isinstance(extractor.config, dict):
                    init_params = extractor.config.copy()
                # 最后从params获取
                elif hasattr(extractor, "params") and isinstance(extractor.params, dict):
                    init_params = extractor.params.copy()
                else:
                    # 如果没有找到参数，尝试从__dict__中过滤，但要小心
                    if hasattr(extractor, "__dict__"):
                        # 过滤掉一些不应该作为初始化参数的属性
                        exclude_attrs = {"logger", "_logger", "model", "preprocess", "device"}
                        init_params = {
                            k: v
                            for k, v in extractor.__dict__.items()
                            if not k.startswith("_") and k not in exclude_attrs and not callable(v)
                        }

                # 确保移除可能导致冲突的参数
                params_to_remove = ["name", "_name", "device", "_device"]
                for param in params_to_remove:
                    if param in init_params:
                        del init_params[param]

                # 保留重要的模型参数
                if hasattr(extractor, "model_name"):
                    init_params["model_name"] = extractor.model_name
                if hasattr(extractor, "model_path"):
                    init_params["model_path"] = extractor.model_path

                self.extractors_config[name] = {
                    "class_module": class_module,
                    "class_name": class_name,
                    "params": init_params,
                }
                logger.debug(f"准备深度特征提取器配置: {name}, 参数: {list(init_params.keys())}")

            except Exception as e:
                logger.error(f"准备提取器 {name} 配置失败: {e}")
                # 如果配置失败，使用空参数
                self.extractors_config[name] = {
                    "class_module": extractor.__class__.__module__,
                    "class_name": extractor.__class__.__name__,
                    "params": {},
                }

    @staticmethod
    def merge_features(cached_features: pd.DataFrame, extracted_features: pd.DataFrame) -> pd.DataFrame:
        if cached_features.empty and extracted_features.empty:
            return pd.DataFrame()
        if cached_features.empty:
            return extracted_features
        if extracted_features.empty:
            return cached_features

        # 确保 image_path 是列而不是索引
        if cached_features.index.name == "image_path":
            cached_features = cached_features.reset_index()
        if extracted_features.index.name == "image_path":
            extracted_features = extracted_features.reset_index()

        # 断言两者的列名完全一致
        logger.debug(f"合并特征: 缓存特征列 {list(cached_features.columns)}, 提取特征列 {list(extracted_features.columns)}")
        # 基于image_path合并两个DataFrame, 相同的列以extracted_features为主, 不同的列保留,
        if "image_path" not in cached_features.columns or "image_path" not in extracted_features.columns:
            result = pd.concat([cached_features, extracted_features], ignore_index=True)
        else:
            # 根据cache_features中的image_path, 找到extracted_features对应的image_path, 并用extracted_features中的相应的列的值更新到cached_features中
            # 使用左连接合并，保留所有cached_features的行，并添加extracted_features中的新行
            result = pd.merge(cached_features, extracted_features, on="image_path", how="outer", suffixes=("_cached", "_extracted"))

            # 智能合并重叠列：优先使用extracted_features的非空值
            overlap_cols = set(cached_features.columns) & set(extracted_features.columns)
            overlap_cols.discard("image_path")  # 排除连接键

            for col in overlap_cols:
                cached_col = f"{col}_cached"
                extracted_col = f"{col}_extracted"

                # 使用extracted值更新，如果extracted为空则保留cached值
                result[col] = result[extracted_col].fillna(result[cached_col])

                # 清理临时列
                result.drop(columns=[cached_col, extracted_col], inplace=True)
        # assert set(cached_features.columns) == set(extracted_features.columns), "缓存特征和提取特征的列名不匹配"
        # result = pd.concat([cached_features, extracted_features], ignore_index=True)
        return result

    def process_features(self, images_path: pd.Series, dataset_name, use_cache: bool = True) -> pd.DataFrame:
        """处理特征提取的主流程"""
        if not self.extractor_configs:
            raise ValueError(f"没有可用的{self.feature_type}特征提取器")

        logger.info(
            f"开始处理{self.feature_type}特征: {len(images_path)} 张图像, {len(self.extractor_configs)} 个特征, {self.extractor_configs.keys()}"
        )

        # 加载缓存特征和获取未缓存图像
        cached_features, uncached_images = self.load_cached_features(
            images_path=images_path,
            use_cache=use_cache,
            dataset_name=dataset_name,
        )

        # 处理未缓存的图像
        extracted_features = self._process_uncached_images(uncached_images, dataset_name)

        return self.merge_features(cached_features, extracted_features)

    def _process_uncached_images(self, uncached_images, dataset_name) -> pd.DataFrame:
        """处理未缓存的图像"""
        is_empty = (
            (isinstance(uncached_images, pd.Series) and uncached_images.empty)
            or (isinstance(uncached_images, pd.DataFrame) and uncached_images.empty)
            or (isinstance(uncached_images, dict) and not uncached_images)
        )
        if is_empty:
            logger.info(f"所有{self.feature_type}特征已从缓存加载，无需提取")
            return pd.DataFrame()

        logger.info(f"需要提取{self.feature_type}特征的图像: {len(uncached_images)}")

        # 从uncached_images中随机抽取一张图片进行测试
        extracted_features = self.extract_features_impl(uncached_images, dataset_name)
        return pd.DataFrame.from_dict(extracted_features, orient="index").reset_index().rename(columns={"index": "image_path"})
