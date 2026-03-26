import os
import multiprocessing


def setup_blas_threads():
    """
    动态设置BLAS库的线程数，避免OpenBLAS警告
    """
    # 获取CPU核心数
    cpu_count = multiprocessing.cpu_count()
    thread_count = cpu_count
    # 确保线程数不超过 OpenBLAS 的限制
    max_openblas_threads = 64
    thread_count = min(thread_count, max_openblas_threads)

    # 设置各种BLAS库的环境变量
    blas_env_vars = ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]

    for var in blas_env_vars:
        os.environ[var] = str(thread_count)

    print(f"设置BLAS线程数为: {thread_count} (CPU核心数: {cpu_count})")


setup_blas_threads()

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import traceback
from forensic_agent.generate_profile.feature_generator import FeatureGenerator
from forensic_agent.generate_profile.profile_generator import ProfileGenerator
from forensic_agent.generate_profile.clustering_generator import ClusteringGenerator
from forensic_agent.data_operation.dataset_loader import DatasetLoader, load_project_data
from forensic_agent.utils.logger import get_logger, setup_logging
from cfg import CONFIG


class ClusteringProfileAnalyzer:
    """模型画像分析器 - 主要分析引擎"""

    def __init__(self, output_dir: str):
        """初始化分析器

        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            cache_dir: 缓存目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = get_logger(__name__)
        dataset = CONFIG.get("dataset", {})

        self.db_dir = dataset["db_dir"]
        self.model_names = dataset["model_names"]
        self.dataset_names = dataset["dataset_names"]
        self.dataset_root = dataset.get("datasets_root")
        self.max_num = dataset.get("max_datasets", None)

        # 初始化组件
        self.dataset_processor: Optional[DatasetLoader] = None
        self.feature_manager: Optional[FeatureGenerator] = None
        self.clustering_discoverer: Optional[ClusteringGenerator] = None

        # 模型画像生成器
        self.portrait_generator = ProfileGenerator(config=CONFIG["profile"])

        # 分析结果
        self.analysis_results: Dict[str, Any] = {}

    def run_analysis(self) -> Dict[str, Any]:
        """运行完整的模型画像分析

        Args:
            dataset_root: 数据集根目录
            max_datasets: 最大数据集数量

        Returns:
            Dict[str, Any]: 分析结果
        """
        self.logger.info("开始模型画像与内容适配度分析")

        try:
            # 1. 数据加载与预处理
            self.logger.info("=" * 50)
            self.logger.info("步骤 1: 数据加载与预处理")
            train_data, test_data = None, None

            if CONFIG.get("train_dataset", None):
                # 1. 加载并初步清洗训练集
                train_data = pd.read_csv(CONFIG["train_dataset"]).drop_duplicates(subset=["image_path"]).reset_index(drop=True)

                # 2. 处理验证集逻辑
                val_path = CONFIG.get("val_dataset", None)
                test_path_for_val = CONFIG.get("test_dataset", None)  # 获取测试集路径用于判断

                if val_path:
                    # 情况 A: 提供了验证集路径 -> 直接加载
                    val_data = pd.read_csv(val_path).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
                    self.logger.info(f"使用指定验证数据集: {val_path}")

                elif test_path_for_val:
                    # 情况 B (修改): 未提供验证集，但有测试集 -> 使用测试集作为验证集
                    # 注意：这里不从训练集中剔除任何数据
                    self.logger.info(f"未配置 'val_dataset'，检测到 'test_dataset' ({test_path_for_val})，将其作为验证集使用")
                    val_data = pd.read_csv(test_path_for_val).drop_duplicates(subset=["image_path"]).reset_index(drop=True)

                else:
                    # 情况 C: 既无验证集也无测试集 -> 从训练集切分 20%
                    self.logger.info("未配置 'val_dataset' 且无 'test_dataset'，正在从训练集中随机抽取 20% 作为验证集...")
                    # random_state=42 保证每次运行切分结果一致
                    val_data = train_data.sample(frac=0.2, random_state=42)
                    # 从训练集中剔除验证集的数据
                    train_data = train_data.drop(val_data.index).reset_index(drop=True)
                    val_data = val_data.reset_index(drop=True)

                # 3. 统一记录数量与清理无用列 (Cluster Columns)
                self.logger.info(f"最终训练集记录数: {len(train_data)}")
                self.logger.info(f"最终验证集记录数: {len(val_data)}")
                # 清理训练集列
                cluster_columns_train = [col for col in train_data.columns if col.startswith("cluster_")]
                train_data = train_data.drop(columns=cluster_columns_train, errors="ignore")
                # 清理验证集列
                cluster_columns_val = [col for col in val_data.columns if col.startswith("cluster_")]
                val_data = val_data.drop(columns=cluster_columns_val, errors="ignore")

            # 加载测试集 (保持原有逻辑，用于后续单独的测试集分析流程)
            if CONFIG.get("test_dataset", None):
                test_data = pd.read_csv(CONFIG["test_dataset"]).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
                self.logger.info(f"使用测试数据集: {CONFIG['test_dataset']}, 记录数: {len(test_data)}")
                if CONFIG["clustering"].get("force_clustering", False):
                    self.logger.info("强制重新进行聚类分析，忽略已有聚类结果")
                    cluster_columns = [col for col in test_data.columns if col.startswith("cluster_")]
                    test_data = test_data.drop(columns=cluster_columns, errors="ignore")

            # 2. 特征提取（如果需要）
            self.logger.info("=" * 50)
            self.logger.info("步骤 2: 图像特征提取")
            train_features_data = self._extract_features(train_data) if train_data is not None else None
            val_features_data = self._extract_features(val_data) if val_data is not None else None
            test_features_data = self._extract_features(test_data) if test_data is not None else None

            # 3. 内容类型发现与聚类
            self.logger.info("=" * 50)
            self.logger.info("步骤 3: 内容类型发现与聚类")
            train_clustering_res, train_cluster_data, val_clustering_res, val_cluster_data, test_cluster_res, test_cluster_data = (
                self._discover_cluster(train_features_data, val_features_data, test_features_data)
            )

            # 将合并后的data保存到output_dir
            tmp = ""
            for columns in CONFIG["clustering"]["target_columns"]:
                tmp += "&".join(columns) + "_"

            # 合并所有数据集的聚类结果
            if train_cluster_data is not None and not train_cluster_data.empty:
                # 4. 聚类画像生成
                self.logger.info("=" * 50)
                self.logger.info("步骤 4: 聚类画像生成")
                # 将train_cluster_data和data根据image_path进行合并, 列名冲突时以train_cluster_data为准
                cluster_columns = [col for col in train_cluster_data.columns if col.startswith("cluster_")]
                data = pd.read_csv(CONFIG["train_dataset"])
                merge_data: pd.DataFrame = train_cluster_data[["image_path"] + cluster_columns].copy()
                merge_data = merge_data.set_index("image_path").combine_first(data.set_index("image_path")).reset_index()
                self.portrait_generator.run(clustering_result=train_clustering_res, clustering_data=merge_data)
                merge_data.to_csv(Path(CONFIG["train_dataset"]).parent / f"train_dataset_{tmp}.csv", index=False)

            if test_cluster_data is not None and not test_cluster_data.empty:
                # 将clustering_data中关于"cluster_"开头的列合并进data, 以image_path为对应索引统计
                cluster_columns = [col for col in test_cluster_data.columns if col.startswith("cluster_")]
                merge_data: pd.DataFrame = test_cluster_data[["image_path"] + cluster_columns].copy()
                merge_data = merge_data.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
                # 将all_cluster_data与data根据image_path进行合并
                data = pd.read_csv(CONFIG["test_dataset"])
                merge_data = merge_data.set_index("image_path").combine_first(data.set_index("image_path")).reset_index()
                save_path = Path(CONFIG["test_dataset"]).parent / f"test_dataset_{tmp}.csv"
                merge_data.to_csv(save_path, index=False)
                self.logger.info(f"测试数据集聚类结果已保存到: {save_path}")

            return self.analysis_results

        except Exception as e:
            self.logger.error(f"分析过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _load_and_process_data(self, dataset_root: str, max_num: Optional[int]) -> tuple:
        """加载和处理数据"""
        # 使用统一数据加载器
        data, processor = load_project_data(
            datasets_root=dataset_root,
            db_dir=self.db_dir,
            max_num=max_num,
            dataset_names=self.dataset_names,
            model_names=self.model_names,
        )

        # # 只要gt_label为0的数据
        # data = data[data["gt_label"] == 0].reset_index(drop=True)

        # 添加pred_result列和calibration_result列
        # 取 "gt_label"和"pred_prob",阈值为0.5进行比较, 相同则对应的pred_result为1, 否则为0
        data["pred_result"] = ((data["pred_prob"] >= 0.5) == data["gt_label"]).astype(int)
        if "calibration_prob" in data:
            data["calibration_result"] = ((data["calibration_prob"] >= 0.5) == data["gt_label"]).astype(int)

        # 数据摘要
        summary = processor.get_summary(data)
        self.logger.info(f"数据加载完成:")
        self.logger.info(f"  - 总图片数: {summary['total_images']}")
        self.logger.info(f"  - 总预测记录: {summary['total_predictions']}")
        self.logger.info(f"  - 模型数量: {summary['model_count']}")
        self.logger.info(f"  - 数据集数量: {len(summary.get('datasets', []))}")

        return data, processor

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取图像特征"""
        self.feature_manager = FeatureGenerator(config=CONFIG["feature"], db_dir=self.db_dir)
        extract_feat = data.copy()
        manual_results, depth_results = [], []
        for dataset_name, group in extract_feat.groupby("dataset_name"):
            unique_images = group["image_path"].drop_duplicates()
            self.logger.info(f"提取数据集的图片特征: {dataset_name}, 需要提取 {len(unique_images)} 张图片的特征")
            manual_df, depth_df = self.feature_manager.run_concurrent(unique_images, dataset_name, use_cache=True)
            manual_results.append(manual_df)
            depth_results.append(depth_df)
            self.logger.info(f"完成数据集的图片特征提取: {dataset_name}")

        self.logger.info("所有数据集的图片特征提取完成")
        self.feature_manager.close()

        manual_all = pd.concat(manual_results, ignore_index=True)
        depth_all = pd.concat(depth_results, ignore_index=True)
        features = pd.merge(manual_all, depth_all, on="image_path", how="inner")
        features = pd.merge(features, data, on="image_path", how="inner").reset_index(drop=True)
        # 释放内存
        del manual_all, depth_all, manual_results, depth_results
        self.logger.info(f"最终特征数据集大小: {features.shape}, 包含列: {list(features.columns)}")

        return features

    def _discover_cluster(
        self, train_features_data: pd.DataFrame, val_features_data: pd.DataFrame, test_features_data: pd.DataFrame
    ) -> tuple:
        # 初始化聚类发现器
        self.clustering_discoverer = ClusteringGenerator(
            CONFIG.get("clustering"),
            extractor_desc=self.feature_manager.extractors_desc,
            db_manager=self.feature_manager.db_manager,
        )

        # 进行模型聚类
        target_columns = CONFIG["clustering"]["target_columns"]
        train_clustering_res, train_cluster_data, test_clustering_res, test_cluster_data, val_clustering_res = None, None, None, None, None
        # if train_features_data is not None and not train_features_data.empty:
        #     train_clustering_res, train_cluster_data = self.clustering_discoverer.discover_content_types(
        #         train_features_data,
        #         target_columns,
        #         is_train=False,
        #         save_result=False,
        #     )

        # # 进行验证数据的聚类
        # if val_features_data is not None and not val_features_data.empty:
        #     val_clustering_res, val_cluster_data = self.clustering_discoverer.discover_content_types(
        #         val_features_data,
        #         target_columns,
        #         is_train=False,
        #         save_result=False,
        #     )

        #     # 去掉重复的列, 出现相同的列值以data为准
        #     val_cluster_data = pd.merge(
        #         val_cluster_data, val_features_data, on="image_path", how="inner", suffixes=("_features", "_data")
        #     ).reset_index(drop=True)
        #     # 如果存在重复列，优先保留data中的列值
        #     for col in val_features_data.columns:
        #         if col in val_features_data.columns and col != "image_path":
        #             val_cluster_data[col] = val_cluster_data[f"{col}_data"]
        #             val_cluster_data.drop(columns=[f"{col}_features", f"{col}_data"], inplace=True, errors="ignore")

        # 进行测试数据的聚类
        if test_features_data is not None and not test_features_data.empty:
            test_clustering_res, test_cluster_data = self.clustering_discoverer.discover_content_types(
                test_features_data,
                target_columns,
                is_train=False,
                save_result=False,
            )

            # 去掉重复的列, 出现相同的列值以data为准
            test_cluster_data = pd.merge(
                test_cluster_data, test_features_data, on="image_path", how="inner", suffixes=("_features", "_data")
            ).reset_index(drop=True)
            # 如果存在重复列，优先保留data中的列值
            for col in test_features_data.columns:
                if col in test_features_data.columns and col != "image_path":
                    test_cluster_data[col] = test_cluster_data[f"{col}_data"]
                    test_cluster_data.drop(columns=[f"{col}_features", f"{col}_data"], inplace=True, errors="ignore")

        return train_clustering_res, train_cluster_data, val_clustering_res, val_features_data, test_clustering_res, test_cluster_data


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="模型画像与内容适配度分析工具", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o", type=str, default="outputs", help="输出目录")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别 (默认: INFO)"
    )
    args = parser.parse_args()

    # 设置日志
    setup_logging(log_level=args.log_level.upper())
    logger = get_logger(__name__)

    try:
        # 初始化分析器
        analyzer = ClusteringProfileAnalyzer(output_dir=args.output)
        analyzer.run_analysis()

    except Exception as e:
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
