import hashlib
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.logger import get_logger, LogContext, log_execution_time
from .database_manager import DatabaseManager, PredictionRecord
from .dataset_split import DatasetSplit


@dataclass
class FileInfo:
    """文件信息数据结构"""

    file_path: Path
    model_name: str
    dataset_name: str
    data_source: str


@dataclass
class DatasetScanResult:
    """数据集扫描结果"""

    total_files: int
    valid_files: int
    invalid_files: int
    file_infos: List[FileInfo]  # 包含文件路径和元信息的列表
    model_names: List[str]
    dataset_names: List[str]
    data_sources: List[str]


@dataclass
class LoadResult:
    """数据加载结果"""

    data: pd.DataFrame
    processor: "DatasetLoader"
    summary: Dict[str, Any]
    load_source: str  # 'json', 'sqlite', 'mixed'


class DatasetLoader:
    """统一数据加载器

    支持从resources/datasets读取JSON文件并存储到SQLite数据库，
    同时支持从SQLite数据库读取数据进行分析
    """

    def __init__(self, datasets_root: str = None, db_dir: str = None):
        """初始化数据加载器

        Args:
            datasets_root: 数据集根目录
            db_path: SQLite数据库路径
        """
        # 断言datasets_root和db_dir不同时为None
        if datasets_root is None and db_dir is None:
            raise ValueError("datasets_root和db_dir不能同时为None")

        self.datasets_root = Path(datasets_root) if datasets_root else None
        if db_dir:
            self.db_manager = DatabaseManager(db_dir)
        else:
            self.db_manager = None
        self.logger = get_logger(__name__)

        # 初始化数据分割器
        self.dataset_splitter = DatasetSplit(self.db_manager)

        # 加载的数据
        self.loaded_data: Optional[pd.DataFrame] = None
        self.load_source: Optional[str] = None

    @staticmethod
    def generate_image_id(image_path: str, dataset_name: str = "", data_source: str = "", model_name: str = "") -> Optional[str]:
        """生成图片ID的hash值

        Args:
            image_path: 图片路径
            model_name: 模型名称
            dataset_name: 数据集名称
            data_source: 数据源

        Returns:
            str: 生成的图片ID hash值，如果image_path为空则返回None
        """
        if not image_path or not isinstance(image_path, str):
            return None
        image_name = Path(image_path).name
        hash_string = f"{image_name}_{dataset_name}_{data_source}_{model_name}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    @log_execution_time
    def scan_datasets(self, model_name: Optional[Union[str, List[str]]] = None, dataset_name: Optional[str] = None) -> DatasetScanResult:
        """扫描数据集文件

        Args:
            model_name: 指定模型名（不指定则扫描全部）
            dataset_name: 指定数据集名（不指定则扫描全部）

        Returns:
            DatasetScanResult: 扫描结果
        """
        with LogContext("扫描数据集文件", level="INFO"):
            file_infos_candidates = []
            model_names = set()
            dataset_names = set()
            data_sources = set()

            # 递归搜索JSON文件
            if model_name:
                # 根据model_name类型判断处理方式
                if isinstance(model_name, str):
                    # 单个模型名，只搜索特定模型目录
                    model_dirs = [self.datasets_root / model_name] if (self.datasets_root / model_name).exists() else []
                else:
                    # 模型名列表，搜索所有指定的模型目录
                    model_dirs = []
                    for m_name in model_name:
                        if (self.datasets_root / m_name).exists():
                            model_dirs.append(self.datasets_root / m_name)
                        else:
                            self.logger.warning(f"模型目录不存在: {self.datasets_root / m_name}")
            else:
                # 搜索所有模型目录
                model_dirs = [d for d in self.datasets_root.iterdir() if d.is_dir()]

            for model_dir in model_dirs:
                current_model_name = model_dir.name
                self.logger.debug(f"扫描模型目录: {current_model_name}")

                if dataset_name:
                    if isinstance(dataset_name, str):
                        # 指定数据集名，只搜索特定数据集
                        dataset_dirs = [model_dir / dataset_name] if (model_dir / dataset_name).exists() else []
                    elif isinstance(dataset_name, list):
                        # 数据集名列表，搜索所有指定的数据集
                        dataset_dirs = []
                        for d_name in dataset_name:
                            if (model_dir / d_name).exists():
                                dataset_dirs.append(model_dir / d_name)
                            else:
                                self.logger.warning(f"数据集目录不存在: {model_dir / d_name}")
                else:
                    # 搜索所有数据集目录
                    dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

                for dataset_dir in dataset_dirs:
                    current_dataset_name = dataset_dir.name
                    self.logger.debug(f"扫描数据集目录: {current_dataset_name}")

                    # 递归搜索JSON文件
                    json_files = list(dataset_dir.rglob("*.json"))

                    for json_file in json_files:
                        # 解析文件路径结构获取数据来源
                        data_source = self._extract_data_source(json_file, dataset_dir)

                        # 创建FileInfo对象
                        file_info = FileInfo(
                            file_path=json_file,
                            model_name=current_model_name,
                            dataset_name=current_dataset_name,
                            data_source=data_source,
                        )
                        file_infos_candidates.append(file_info)

                        model_names.add(current_model_name)
                        dataset_names.add(current_dataset_name)
                        data_sources.add(data_source)

            # 验证文件有效性
            valid_file_infos = []
            invalid_files = 0

            for file_info in tqdm(file_infos_candidates, desc="验证JSON文件"):
                if self._validate_json_file(file_info.file_path):
                    valid_file_infos.append(file_info)
                else:
                    invalid_files += 1

            result = DatasetScanResult(
                total_files=len(file_infos_candidates),
                valid_files=len(valid_file_infos),
                invalid_files=invalid_files,
                file_infos=valid_file_infos,
                model_names=sorted(list(model_names)),
                dataset_names=sorted(list(dataset_names)),
                data_sources=sorted(list(data_sources)),
            )

            self.logger.info(f"扫描完成: 总文件 {result.total_files}, " f"有效文件 {result.valid_files}, 无效文件 {result.invalid_files}")
            self.logger.info(f"发现模型: {len(result.model_names)} 个")
            self.logger.info(f"发现数据集: {len(result.dataset_names)} 个")
            self.logger.info(f"发现数据源: {len(result.data_sources)} 个")

            return result

    def _clean_data_source_name(self, data_source: str) -> str:
        """清理数据源名称，去掉常见的模型后缀

        Args:
            data_source: 原始数据源名称

        Returns:
            str: 清理后的数据源名称
        """
        data_source = re.sub(r"_genimage.*$", "", data_source)
        data_source = re.sub(r"_model.*$", "", data_source)
        data_source = re.sub(r"_no-checkpoint$", "", data_source)
        return data_source

    def _extract_data_source(self, json_file: Path, dataset_dir: Path) -> str:
        """从文件路径提取数据来源名

        Args:
            json_file: JSON文件路径
            dataset_dir: 数据集目录路径

        Returns:
            str: 数据来源名
        """
        # 获取相对于数据集目录的路径
        relative_path = json_file.relative_to(dataset_dir)

        # 情况1: 直接在数据集目录下的JSON文件
        # 如: AIGCDetect-testset/ADM_genimage_sd14_epoch_best.json
        if len(relative_path.parts) == 1:
            # 从文件名提取数据来源，去掉后缀
            data_source = relative_path.stem
            # 清理数据源名称
            data_source = self._clean_data_source_name(data_source)
            # 额外去掉通用后缀（适用于第一种情况）
            data_source = re.sub(r"_.*$", "", data_source)
            return data_source

        # 情况2: 在子目录中的JSON文件
        # 如: AIGCDetect-testset/cyclegan/apple_genimage_sd14_epoch_best.json
        elif len(relative_path.parts) == 2:
            # 数据来源是子目录名 + 文件名前缀
            subdir = relative_path.parts[0]
            filename = relative_path.stem

            # 从文件名提取前缀并清理
            prefix = self._clean_data_source_name(filename)

            return f"{subdir}_{prefix}"

        # 其他情况，使用完整相对路径
        else:
            path_str = str(relative_path.with_suffix(""))
            return path_str.replace("/", "_").replace("\\", "_")

    def _validate_json_file(self, json_file: Path) -> bool:
        """验证JSON文件格式

        Args:
            json_file: JSON文件路径

        Returns:
            bool: 是否有效
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or not data:
            return False

        # 检查数据格式：{image_path: {pred_label: float, gt_label: int}}
        sample_key = next(iter(data))
        # 校验每个sample_key路径是否存在
        if not Path(sample_key).exists():
            self.logger.warning(f"样本路径不存在: {sample_key}")
            return False
        sample_value = data[sample_key]

        if not isinstance(sample_value, dict):
            return False

        required_fields = ["pred_label", "gt_label"]
        if not all(field in sample_value for field in required_fields):
            return False

        # 检查数据类型
        try:
            float(sample_value["pred_label"])
            int(sample_value["gt_label"])
        except (ValueError, TypeError):
            return False
        return True

    def _process_single_file(
        self,
        json_file: Path,
        file_model_name: str,
        file_dataset_name: str,
        file_data_source: str,
    ) -> Tuple[List[PredictionRecord], bool, str]:
        """处理单个JSON文件

        Args:
            json_file: JSON文件路径
            file_model_name: 模型名称（从扫描结果获取，避免重复解析）
            file_dataset_name: 数据集名称（从扫描结果获取，避免重复解析）
            file_data_source: 数据源名称（从扫描结果获取，避免重复解析）

        Returns:
            Tuple[List[PredictionRecord], bool, str]: (记录列表, 是否成功, 错误信息)
        """
        # 读取JSON数据
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 转换为PredictionRecord列表
        file_records = []
        for image_path, pred_info in data.items():
            try:
                file_records.append(
                    {
                        "image_path": image_path,
                        "model_name": file_model_name,
                        "dataset_name": file_dataset_name,
                        "data_source": file_data_source,
                        "pred_prob": float(pred_info.get("pred_prob", pred_info["pred_label"])),
                        "pred_label": int(pred_info["pred_label"]),
                        "gt_label": int(pred_info["gt_label"]),
                    }
                )
            except Exception as e:
                self.logger.warning(f"解析记录失败 {image_path}: {e}")

        self.logger.debug(f"处理文件 {json_file.name}: {len(file_records)} 条记录")
        return file_records, True, ""

    @log_execution_time
    def load_and_store(
        self,
        model_names: Optional[Union[str, List[str]]] = None,
        dataset_names: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1000,
        debug: bool = False,
        workers: int = 4,
    ) -> pd.DataFrame:
        """加载数据并存储到SQLite数据库

        Args:
            model_name: 指定模型名（不指定则加载全部）
            dataset_name: 指定数据集名（不指定则加载全部）
            batch_size: 批处理大小
            debug: 是否启用debug模式（单线程处理）
            workers: 并发工作线程数（debug模式下忽略）
        """
        with LogContext("加载数据到SQLite数据库", level="INFO"):
            # 扫描数据集
            scan_result: DatasetScanResult = self.scan_datasets(model_names, dataset_names)

            if not scan_result.file_infos:
                self.logger.warning("没有找到有效的数据文件")
                return pd.DataFrame()

            total_records = []
            files_processed = 0
            files_failed = 0

            if debug:
                # Debug模式：单线程处理
                self.logger.info("Debug模式: 使用单线程处理文件")
                for file_info in tqdm(scan_result.file_infos, desc="处理JSON文件 (单线程)"):
                    # 直接使用FileInfo对象中的元信息
                    file_records, success, error_msg = self._process_single_file(
                        file_info.file_path,
                        file_info.model_name,
                        file_info.dataset_name,
                        file_info.data_source,
                    )
                    if success:
                        total_records.extend(file_records)
                        files_processed += 1
                    else:
                        self.logger.error(error_msg)
                        files_failed += 1
            else:
                # 多线程模式：并发处理文件
                self.logger.debug(f"多线程模式: 使用 {workers} 个工作线程处理文件")

                # 使用线程池处理文件
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    # 提交所有任务，直接使用FileInfo对象
                    future_to_file_info = {}
                    for file_info in scan_result.file_infos:
                        future = executor.submit(
                            self._process_single_file,
                            file_info.file_path,
                            file_info.model_name,
                            file_info.dataset_name,
                            file_info.data_source,
                        )
                        future_to_file_info[future] = file_info

                    # 处理完成的任务
                    for future in tqdm(as_completed(future_to_file_info), total=len(future_to_file_info), desc="处理JSON文件 (多线程)"):
                        file_info = future_to_file_info[future]
                        try:
                            file_records, success, error_msg = future.result()
                            if success:
                                # 使用锁确保线程安全地添加记录
                                total_records.extend(file_records)
                                files_processed += 1
                            else:
                                files_failed += 1
                                self.logger.error(error_msg)
                        except Exception as e:
                            files_failed += 1
                            self.logger.error(f"获取任务结果失败 {file_info.file_path}: {e}")

            # 批量插入数据库
            self.logger.info(f"准备插入数据库: 总记录数 {len(total_records)}")
            total_records = pd.DataFrame(total_records)
            assert not total_records.empty, "没有有效的记录可插入数据库"

            records_inserted = self.db_manager.insert_predictions(total_records, batch_size=batch_size)
            self.logger.info(f"数据加载完成: 处理 {files_processed} 个文件, " f"插入 {records_inserted} 条记录, 失败 {files_failed} 个文件")
            return total_records

    def load_data(
        self,
        model_names: Optional[str] = None,
        dataset_names: Optional[str] = None,
        data_source: Optional[str] = None,
        force_reload: bool = False,
        max_num: Optional[int] = None,
    ) -> LoadResult:
        """加载数据

        Args:
            model_name: 指定模型名（不指定则加载全部）
            dataset_name: 指定数据集名（不指定则加载全部）
            data_source: 指定数据源（不指定则加载全部）
            force_reload: 强制重新从JSON加载
            max_datasets: 最大数据集数量限制

        Returns:
            LoadResult: 加载结果
        """
        self.logger.info("开始统一数据加载")

        # 检查数据库是否存在且有数据
        # check_res = self._check_database_data(model_names, dataset_name, data_source)
        # db_has_data = check_res["available"]
        db_has_data = True
        if db_has_data and not force_reload:
            # 从SQLite加载
            self.logger.info("从SQLite数据库加载数据")
            data_predictions: pd.DataFrame = self.db_manager.get_predictions(
                model_names=model_names,
                dataset_names=dataset_names,
                data_source=data_source,
                limit=max_num,
            )
            image_info: pd.DataFrame = self.db_manager.get_image_info(dataset_names)
            # 基于image_path, 合并这两个pd, 如果出现image_path不匹配, 则报错
            self.db_manager.close_all_connections()

            data = pd.merge(data_predictions, image_info, on="image_path", how="inner", suffixes=("", "_dup"))
            # 删除所有以'_dup'结尾的列（这些是来自image_info的重复列）
            dup_cols = [col for col in data.columns if col.endswith("_dup")]
            data = data.drop(columns=dup_cols)
            load_source = "sqlite"
        else:
            # 从JSON加载或重新加载
            if self.datasets_root.exists():
                self.logger.info("从JSON文件加载数据到SQLite数据库")
                data: pd.DataFrame = self.load_and_store(
                    model_names=model_names,
                    dataset_names=dataset_names,
                    batch_size=5000,
                    debug=False,
                    workers=16,
                )
                load_source = "json" if not db_has_data else "mixed"
            else:
                raise FileNotFoundError(f"数据集根目录不存在: {self.datasets_root}")

        if data.empty:
            raise ValueError("没有加载到任何数据")

        # 存储加载的数据
        self.loaded_data = data
        self.load_source = load_source

        # 生成摘要
        summary = self.get_summary(data)

        self.logger.info(f"数据加载完成，来源: {load_source}")
        return LoadResult(data=data, processor=self, summary=summary, load_source=load_source)

    # def _check_database_data(
    #     self,
    #     model_names: Optional[Union[str, List[str]]] = None,
    #     dataset_names: Optional[str] = None,
    #     data_sources: Optional[Union[str, List[str]]] = None,
    # ) -> bool:
    #     """检查指定的模型名、数据集名和数据源是否在数据库中存在

    #     Args:
    #         model_names: 要检查的模型名，可以是字符串或字符串列表
    #         dataset_names: 要检查的数据集名，可以是字符串或字符串列表
    #         data_sources: 要检查的数据源，可以是字符串或字符串列表

    #     Returns:
    #         Dict[str, Any]: 检查结果，包含存在/不存在的项目和总体可用性状态
    #     """

    #     # 标准化输入参数为列表
    #     def normalize_to_list(param):
    #         if param is None:
    #             return []
    #         elif isinstance(param, str):
    #             return [param]
    #         elif isinstance(param, list):
    #             return param
    #         else:
    #             return []

    #     model_list = normalize_to_list(model_names)
    #     dataset_list = normalize_to_list(dataset_names)
    #     if not model_list and not dataset_list:
    #         return {"available": True, "details": "未指定模型名或数据集名，默认数据可用"}

    #     unable_info = []
    #     for model in model_list:
    #         for dataset in dataset_list:
    #             records = self.db_manager.get_predictions(model_names=model, dataset_names=dataset, limit=1)
    #             if records.empty:
    #                 unable_info.append((model, dataset))

    #     if unable_info:
    #         self.logger.warning(f"数据库中缺少以下模型和数据集的组合: {unable_info}")
    #         return {"available": False, "details": unable_info}

    #     return {"available": True, "details": "所有指定的模型和数据集组合均有数据"}

    def get_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成数据摘要

        Args:
            data: 数据DataFrame

        Returns:
            Dict[str, Any]: 摘要信息
        """
        summary = {
            "total_images": data["image_path"].nunique(),
            "total_predictions": len(data),
            "model_count": data["model_name"].nunique() if "model_name" in data.columns else 1,
            "models": sorted(data["model_name"].unique().tolist()) if "model_name" in data.columns else [],
            "datasets": sorted(data["dataset_name"].unique().tolist()) if "dataset_name" in data.columns else [],
            "data_sources": sorted(data["data_source"].unique().tolist()) if "data_source" in data.columns else [],
        }

        # 添加生成器家族统计
        if "generator_family" in data.columns:
            family_counts = data["generator_family"].value_counts()
            summary["generator_families"] = family_counts.to_dict()

        # 添加split统计
        if "split" in data.columns:
            split_counts = data["split"].value_counts()
            summary["splits"] = split_counts.to_dict()

        # 添加标签统计
        if "gt_label" in data.columns:
            label_counts = data["gt_label"].value_counts()
            summary["label_distribution"] = {"real": int(label_counts.get(0, 0)), "fake": int(label_counts.get(1, 0))}

        return summary

    def export_processed_data(self, output_path: Union[str, Path], format: str = "parquet"):
        """导出处理后的数据

        Args:
            output_path: 输出路径
            format: 导出格式 ('parquet', 'csv', 'json')
        """
        if self.loaded_data is None:
            raise ValueError("没有可导出的数据")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            self.loaded_data.to_parquet(output_path, index=False)
        elif format == "csv":
            self.loaded_data.to_csv(output_path, index=False)
        elif format == "json":
            self.loaded_data.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        self.logger.info(f"数据已导出到: {output_path}")


def load_project_data(
    datasets_root: str = None,
    db_dir: str = None,
    model_names: Optional[Union[str, List[str]]] = None,
    dataset_names: Optional[str] = None,
    max_num: Optional[int] = None,
    force_reload: bool = False,
    *args,
    **kwargs,
) -> Tuple[pd.DataFrame, DatasetLoader]:
    """便捷函数：加载项目数据

    Args:
        datasets_root: 数据集根目录
        db_path: 数据库路径
        model_name: 指定模型名
        dataset_name: 指定数据集名
        max_datasets: 最大数据集数量

    Returns:
        Tuple[pd.DataFrame, DatasetLoader]: 数据和加载器
    """
    loader = DatasetLoader(datasets_root, db_dir)
    result = loader.load_data(model_names=model_names, dataset_names=dataset_names, max_num=max_num, force_reload=force_reload)
    return result.data, loader


def load_image_info(
    dataset_names: str,
    datasets_root: str = "resources/datasets",
    db_dir: str = "data/predictions.db",
) -> Tuple[pd.DataFrame, DatasetLoader]:
    """便捷函数：加载项目数据

    Args:
        datasets_root: 数据集根目录
        db_path: 数据库路径
        model_name: 指定模型名
        dataset_name: 指定数据集名
        max_datasets: 最大数据集数量

    Returns:
        Tuple[pd.DataFrame, DatasetLoader]: 数据和加载器
    """
    loader = DatasetLoader(datasets_root, db_dir)
    result = loader.db_manager.get_image_info(dataset_names)
    return result, loader
