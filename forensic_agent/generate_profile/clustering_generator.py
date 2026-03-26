from pathlib import Path
import pandas as pd
from typing import List, Dict, Set, Optional

from ..configs.config_dataclass import ClusteringConfig
from .clustering.clustering_algorithm import ClusteringAlgorithm
from .clustering.clustering_preprocessing import ClusteringPreprocessor
from .clustering.clustering_visualizer import ClusteringVisualizer
from ..utils.logger import get_logger, log_execution_time, LogContext
from ..data_operation.database_manager import DatabaseManager


class ClusteringGenerator:
    """聚类内容发现器 - 专注于聚类逻辑"""

    EXCLUDED_COLUMNS = ["image_path", "image_id"]
    DESCRIPTION = "基于多种聚类算法的内容类型发现"

    def __init__(self, config: Dict, extractor_desc, db_manager: DatabaseManager):
        self.logger = get_logger(__name__)
        if config is None:
            self.config = ClusteringConfig()
        else:
            self.config = ClusteringConfig(config)

        self.algorithm_manager = ClusteringAlgorithm(self.config)
        self.preprocessor = ClusteringPreprocessor(self.config)
        self.visualizer = ClusteringVisualizer()

        self.visualizer_path = Path(self.config.save_dir) / "visualizer"
        self.visualizer_path.mkdir(parents=True, exist_ok=True)

        self.extractor_desc = extractor_desc
        self.db_manager = db_manager

        self.min_difference_threshold = self.config.min_difference_threshold
        self.significant_difference_threshold = self.config.significant_difference_threshold

        self.logger.info(
            f"聚类发现器初始化完成 - "
            f"min_cluster_size_abs: {self.config.min_cluster_size_abs}, "
            f"min_samples: {self.config.min_samples}, "
            f"max_features: {self.config.max_features}"
        )

    @staticmethod
    def get_aggregation_columns(clustering_columns: List) -> str:
        # 检查列名是否为空
        if not clustering_columns:
            raise ValueError("聚类列名列表不能为空")

        if isinstance(clustering_columns, str):
            clustering_columns = [clustering_columns]

        # 对列名进行缩写转换，避免超长
        max_length = 64  # SQLite 列名最大长度限制（推荐保守设置）
        joined_name = "&".join(clustering_columns)
        if len(joined_name) > max_length:
            # 缩写逻辑：取每个列名的前3个字符，最后加上哈希值避免冲突
            shortened_columns = [col[:3] for col in clustering_columns]
            hash_suffix = str(abs(hash(joined_name)))[:8]  # 取哈希值的前8位
            joined_name = "&".join(shortened_columns) + "_" + hash_suffix

        # 返回最终列名
        return "cluster_" + joined_name

    def get_cluster_data(self, dataset_name: str, aggregation_column: List[str]) -> Optional[pd.DataFrame]:
        """从数据库中获取聚类结果数据"""
        try:
            cluster_data = self.db_manager.load_clustering_results(dataset_name, aggregation_column)
            if cluster_data.empty:
                self.logger.warning(f"数据集 {dataset_name} 中没有找到聚类结果列 {aggregation_column}")
                return None
            return cluster_data
        except Exception as e:
            self.logger.error(f"查询数据集 {dataset_name} 时出错: {e}")
            return None

    @log_execution_time
    def discover_content_types(
        self,
        data: pd.DataFrame,
        target_columns: List[Set[str]],
        is_train=False,
        save_result=False,
        **kwargs,
    ) -> Dict[str, any]:
        """发现内容类型 - 专注于聚类逻辑"""
        # 输入验证
        self._validate_inputs(data, target_columns)

        total_target_cols = sum(len(col_set) for col_set in target_columns)

        with LogContext(f"聚类内容发现: {data.shape}, 目标列集合数: {len(target_columns)}, 总目标列数: {total_target_cols}", level="INFO"):
            all_clustering_info = []

            # 对每个列集合执行聚类分析
            for i, columns in enumerate(target_columns):
                self.logger.info(f"开始处理第 {i+1} 个列集合: {columns}")

                # 准备当前列集合的数据
                clustering_columns = [col for col in columns if col in data.columns and col not in self.EXCLUDED_COLUMNS]
                if not clustering_columns:
                    raise ValueError(f"第 {i+1} 个列集合没有有效的聚类列，跳过")

                clustering_desc = {}
                for col in clustering_columns:
                    if col in self.extractor_desc:
                        clustering_desc[self.get_aggregation_columns(col)] = self.extractor_desc[col]

                # 数据预处理
                aggregation_columns = self.get_aggregation_columns(columns)
                self.logger.info(f"第 {i+1} 个列集合预处理列: {aggregation_columns}")

                # 检查是否已有聚类结果
                dataset_names = data["dataset_name"].unique().tolist()
                cluster_res: pd.DataFrame = self.db_manager.load_clustering_results(
                    dataset_names=dataset_names, cluster_names=aggregation_columns
                )
                if not cluster_res.empty:
                    # 根据image_path合并cluster_res和data, 以data为主, 不存在data中的image_path则丢弃
                    data = pd.merge(data, cluster_res, on="image_path", how="left")

                # 检查aggregation_columns列是否存在以及是否有缺失值
                if aggregation_columns not in data.columns:
                    self.logger.info(f"数据集中不存在聚类结果列 {aggregation_columns}，需要进行聚类")
                    need_cluster_data = data
                else:
                    need_cluster_data = data
                    # # 检查该列是否有缺失值
                    # missing_mask = data[aggregation_columns].isna()
                    # missing_count = missing_mask.sum()

                    # if missing_count == 0:
                    #     self.logger.info(f"数据集中已存在完整的聚类结果列 {aggregation_columns}，跳过该列集合")
                    #     all_clustering_results.append(clustering_results)
                    #     continue

                    # self.logger.info(f"数据集中聚类结果列 {aggregation_columns} 存在 {missing_count} 个缺失值，继续处理该列集合")
                    # # 如果存在部分缺失, 则只对缺失的重新执行聚类
                    # need_cluster_data = data[missing_mask]

                # 如果data为多列, 合并成一列, 便于后续处理
                processed_data = self.preprocessor.preprocess_features(need_cluster_data[clustering_columns])
                if processed_data.empty:
                    raise ValueError(f"第 {i+1} 个列集合预处理后数据为空，跳过")
                if processed_data.shape[0] < self.config.min_cluster_size_abs:
                    raise ValueError(f"第 {i+1} 个列集合预处理后样本数量过少，跳过")

                # 检查是否有model, 有则读取
                if is_train is False:
                    model = self.algorithm_manager.load_cluster_model(aggregation_columns)
                    if model:
                        self.logger.info(f"找到已有聚类模型，直接使用: {aggregation_columns}")
                    else:
                        self.logger.info(f"未找到已有聚类模型，重新训练: {aggregation_columns}")
                        raise ValueError(f"未找到已有聚类模型，重新训练: {aggregation_columns}")
                else:
                    model = None

                # 执行聚类
                clustering_info = self.algorithm_manager.run(model, processed_data, aggregation_columns, clustering_desc)

                # 聚类结果映射回原始数据（使用索引直接对齐）
                need_cluster_data.loc[processed_data.index, aggregation_columns] = clustering_info.cluster_labels

                # 聚类模型保存
                if model is None:
                    self.algorithm_manager.save_cluster_model(clustering_info.model, aggregation_columns)

                # 保存聚类结果到数据库
                if save_result:
                    self.db_manager.save_clustering_results(need_cluster_data, aggregation_columns)

                all_clustering_info.append(clustering_info)

        # 执行聚类结果可视化分析
        if is_train is False:
            for clustering_info in all_clustering_info:
                columns = clustering_info.clustering_columns
                self.visualizer.visualize(
                    discovery_result=clustering_info,
                    save_path=self.visualizer_path / f"clustering_analysis_{columns}.png",
                )
                # 1. 优雅风格缩略图
                self.visualizer.create_elegant_thumbnail(
                    discovery_result=clustering_info,
                    size=(400, 400),
                    style="elegant",
                    save_path=self.visualizer_path / f"elegant_clusterss_{columns}.png",
                )
                # 2. 极简风格
                self.visualizer.create_elegant_thumbnail(
                    discovery_result=clustering_info,
                    size=(300, 300),
                    style="minimal",
                    save_path=self.visualizer_path / f"minimal_clusterss_{columns}.png",
                )
                # 3. 艺术风格
                self.visualizer.create_elegant_thumbnail(
                    discovery_result=clustering_info,
                    size=(400, 400),
                    style="artistic",
                    save_path=self.visualizer_path / f"artistic_clusterss_{columns}.png",
                )
        # 将聚类结果进行保存
        return all_clustering_info, data

    @log_execution_time
    def pred_clustering(self, data: pd.DataFrame, target_columns: List[Set[str]], **kwargs) -> Dict[str, any]:
        """发现内容类型 - 专注于聚类逻辑"""
        # 输入验证
        self._validate_inputs(data, target_columns)

        total_target_cols = sum(len(col_set) for col_set in target_columns)
        with LogContext(f"聚类内容发现: {data.shape}, 目标列集合数: {len(target_columns)}, 总目标列数: {total_target_cols}", level="INFO"):
            all_clustering_results = []

            # 对每个列集合执行聚类分析
            for i, columns in enumerate(target_columns):
                self.logger.info(f"开始处理第 {i+1} 个列集合: {columns}")

                # 准备当前列集合的数据
                clustering_columns = [col for col in columns if col in data.columns and col not in self.EXCLUDED_COLUMNS]
                if not clustering_columns:
                    self.logger.warning(f"第 {i+1} 个列集合没有有效的聚类列，跳过")
                    continue

                columns_desc = {}
                for col in clustering_columns:
                    if col in self.extractor_desc:
                        columns_desc[col] = self.extractor_desc[col]

                # 数据预处理
                processed_data = self.preprocessor.preprocess_features(data[clustering_columns])
                if processed_data.empty:
                    self.logger.warning(f"第 {i+1} 个列集合预处理后数据为空，跳过")
                    continue

                if processed_data.shape[0] < self.config.min_cluster_size_abs:
                    self.logger.warning(
                        f"第 {i+1} 个列集合预处理后样本数量({processed_data.shape[0]})"
                        f"小于最小聚类大小({self.config.min_cluster_size_abs})，跳过"
                    )
                    continue
                clustering_results, context = self.algorithm_manager.pred_cluster(data, processed_data, clustering_columns)
                all_clustering_results.append([clustering_results, context])

        # 执行聚类结果可视化分析
        for item, context in all_clustering_results:
            columns = item.columns if isinstance(item.columns, list) else [item.columns]
            columns = "-".join(columns)
            self.visualizer.visualize(
                discovery_res=item,
                cluster_data=context.pca_data,
                save_path=self.result_manager.base_save_path / "visualizer" / f"clustering_analysis_{columns}.png",
            )
        return all_clustering_results

    def _validate_inputs(self, data: pd.DataFrame, target_columns: List[Set[str]]) -> None:
        """验证输入参数"""
        if data.empty:
            raise ValueError("输入数据为空，无法进行聚类")

        if not target_columns:
            raise ValueError("target_columns 不能为空")

        # 验证目标列格式和存在性
        all_columns = set(data.columns)
        all_target_columns = set()
        missing_columns = []

        for i, columns in enumerate(target_columns):
            for col in columns:
                if not isinstance(col, str):
                    raise ValueError(f"第 {i+1} 个集合中的列名必须是字符串，当前: {col}")
                if col not in all_columns:
                    missing_columns.append(f"集合{i+1}:{col}")
                all_target_columns.add(col)

        if missing_columns:
            raise ValueError(f"以下目标列在数据中不存在: {missing_columns}")
