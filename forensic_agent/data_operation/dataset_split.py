
import hashlib
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .database_manager import DatabaseManager
import logging


class DatasetSplit:
    """数据集分割器

    专门用于数据集的训练/校准/测试分割，支持从DataFrame或SQLite数据库直接进行数据分割。

    功能特性：
    - 支持从DataFrame进行分层数据分割
    - 支持从SQLite数据库直接提取数据并进行分割
    - 自动处理图片ID去重，避免同一张图片出现在不同集合中
    - 支持分层抽样，保持各类别比例
    - 支持自定义分割比例和随机种子
    """

    def __init__(self, db_manager: DatabaseManager = None):
        """初始化数据集分割器

        Args:
            db_path: SQLite数据库路径
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def create_train_test_split_from_database(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        data_source: Optional[str] = None,
        train_ratio: float = 0.6,
        cal_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify_by: Optional[List[str]] = None,
        random_seed: int = 42,
        image_id_col: str = "image_id",
    ) -> Dict[str, pd.DataFrame]:
        """直接从SQLite数据库进行数据分割

        Args:
            model_name: 指定模型名（不指定则加载全部）
            dataset_name: 指定数据集名（不指定则加载全部）
            data_source: 指定数据源（不指定则加载全部）
            train_ratio: 训练集比例
            cal_ratio: 校准集比例
            test_ratio: 测试集比例
            stratify_by: 分层依据列名列表
            random_seed: 随机种子
            image_id_col: 图片ID列名

        Returns:
            Dict[str, pd.DataFrame]: 分割后的数据
        """
        # 从数据库加载数据
        data = self.load_data_from_database(model_name=model_name, dataset_name=dataset_name, data_source=data_source)

        # 使用现有的分割方法
        return self.create_train_test_split(
            data=data,
            train_ratio=train_ratio,
            cal_ratio=cal_ratio,
            test_ratio=test_ratio,
            stratify_by=stratify_by,
            random_seed=random_seed,
            image_id_col=image_id_col,
        )

    def create_train_test_split_with_stratification(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.6,
        cal_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        image_id_col: str = "image_id",
        min_family_ratio: float = 0.01,
    ) -> Dict[str, pd.DataFrame]:
        """创建带有分层策略的训练/校准/测试数据分割

        根据文档要求实现generator_family分层和难度分箱

        Args:
            data: 输入数据DataFrame
            train_ratio: 训练集比例（构建集）
            cal_ratio: 校准集比例
            test_ratio: 测试集比例（评估集）
            random_seed: 随机种子
            image_id_col: 图片ID列名
            min_family_ratio: 最小家族比例，低于此比例的合并为OTHER

        Returns:
            Dict[str, pd.DataFrame]: 分割后的数据，包含'train', 'calibration', 'test'
        """
        # 验证比例
        if abs(train_ratio + cal_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("比例之和必须等于1.0")

        # 1. 处理generator_family_bucket
        data_processed = self._create_generator_family_buckets(data, min_family_ratio)

        # 2. 计算难度分箱
        data_processed = self._calculate_difficulty_bins(data_processed)

        # 3. 构建分层依据
        stratify_columns = ["gt_label", "generator_family_bucket", "difficulty_bin"]

        # 4. 执行分层分割
        return self._stratified_split_with_buckets(
            data_processed, train_ratio, cal_ratio, test_ratio, stratify_columns, random_seed, image_id_col
        )

    def create_train_test_split(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.6,
        cal_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify_by: Optional[List[str]] = None,
        random_seed: int = 42,
        image_id_col: str = "image_id",
    ) -> Dict[str, pd.DataFrame]:
        """创建训练/校准/测试数据分割

        Args:
            data: 输入数据DataFrame
            train_ratio: 训练集比例
            cal_ratio: 校准集比例
            test_ratio: 测试集比例
            stratify_by: 分层依据列名列表
            random_seed: 随机种子
            image_id_col: 图片ID列名

        Returns:
            Dict[str, pd.DataFrame]: 分割后的数据
        """
        # 验证比例
        if abs(train_ratio + cal_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("比例之和必须等于1.0")

        if stratify_by is None:
            stratify_by = ["gt_label"] if "gt_label" in data.columns else []

        # 验证分层列是否存在
        missing_cols = [col for col in stratify_by if col not in data.columns]
        if missing_cols:
            raise ValueError(f"分层列不存在: {missing_cols}")

        if image_id_col not in data.columns:
            raise ValueError(f"图片ID列不存在: {image_id_col}")

        np.random.seed(random_seed)

        # 获取唯一图片（避免同一张图片出现在不同集合中）
        unique_cols = [image_id_col] + stratify_by
        if "image_path" in data.columns:
            unique_cols.append("image_path")

        unique_images = data[unique_cols].drop_duplicates()

        # 分层分割
        splits = {}
        remaining_data = unique_images.copy()

        # 计算每个分割的样本数
        n_total = len(remaining_data)
        n_train = int(n_total * train_ratio)
        n_cal = int(n_total * cal_ratio)
        n_test = n_total - n_train - n_cal

        # 执行分层抽样
        if stratify_by:
            train_indices = self._stratified_sample(remaining_data, stratify_by, n_train)
            remaining_data = remaining_data.drop(train_indices)

            cal_indices = self._stratified_sample(remaining_data, stratify_by, n_cal)
            remaining_data = remaining_data.drop(cal_indices)
        else:
            # 随机抽样
            train_indices = np.random.choice(remaining_data.index, size=n_train, replace=False)
            remaining_data = remaining_data.drop(train_indices)

            cal_indices = np.random.choice(remaining_data.index, size=n_cal, replace=False)
            remaining_data = remaining_data.drop(cal_indices)

        test_indices = remaining_data.index

        # 基于图片ID获取完整数据
        train_images = unique_images.loc[train_indices, image_id_col].tolist()
        cal_images = unique_images.loc[cal_indices, image_id_col].tolist()
        test_images = unique_images.loc[test_indices, image_id_col].tolist()

        splits["train"] = data[data[image_id_col].isin(train_images)]
        splits["calibration"] = data[data[image_id_col].isin(cal_images)]
        splits["test"] = data[data[image_id_col].isin(test_images)]

        return splits

    def _stratified_sample(self, data: pd.DataFrame, stratify_columns: List[str], n_samples: int) -> pd.Index:
        """分层抽样

        Args:
            data: 数据
            stratify_columns: 分层列
            n_samples: 抽样数量

        Returns:
            pd.Index: 抽样索引
        """
        if n_samples >= len(data):
            return data.index

        # 计算每个分层的比例
        strata_groups = data.groupby(stratify_columns)
        strata_sizes = strata_groups.size()
        strata_proportions = strata_sizes / len(data)

        # 按比例分配样本
        selected_indices = []

        for strata_key, group in strata_groups:
            expected_samples = int(n_samples * strata_proportions[strata_key])
            available_samples = len(group)

            actual_samples = min(expected_samples, available_samples)
            if actual_samples > 0:
                sampled_indices = np.random.choice(group.index, size=actual_samples, replace=False)
                selected_indices.extend(sampled_indices)

        # 如果样本不足，随机补充
        remaining_needed = n_samples - len(selected_indices)
        if remaining_needed > 0:
            remaining_indices = data.index.difference(selected_indices)
            if len(remaining_indices) > 0:
                additional_samples = min(remaining_needed, len(remaining_indices))
                additional_indices = np.random.choice(remaining_indices, size=additional_samples, replace=False)
                selected_indices.extend(additional_indices)

        return pd.Index(selected_indices)

    def _create_generator_family_buckets(self, data: pd.DataFrame, min_ratio: float = 0.01) -> pd.DataFrame:
        """创建generator_family_bucket列

        Args:
            data: 输入数据
            min_ratio: 最小比例阈值

        Returns:
            pd.DataFrame: 添加了generator_family_bucket列的数据
        """
        data_copy = data.copy()

        # 为real样本设置NONE，为fake样本设置UNKNOWN（因为我们不再有generator_family字段）
        data_copy["generator_family_bucket"] = "UNKNOWN"
        data_copy.loc[data_copy["gt_label"] == 0, "generator_family_bucket"] = "NONE"

        # 统计各家族的样本数
        family_counts = data_copy["generator_family_bucket"].value_counts()
        total_samples = len(data_copy)

        # 将占比小于min_ratio的家族合并为OTHER
        rare_families = []
        for family, count in family_counts.items():
            if family != "NONE" and (count / total_samples) < min_ratio:
                rare_families.append(family)

        if rare_families:
            self.logger.info(f"将 {len(rare_families)} 个低频家族合并为OTHER: {rare_families}")
            data_copy.loc[data_copy["generator_family_bucket"].isin(rare_families), "generator_family_bucket"] = "OTHER"

        self.logger.info(f"Generator family buckets分布:")
        bucket_counts = data_copy["generator_family_bucket"].value_counts()
        for bucket, count in bucket_counts.items():
            ratio = count / total_samples
            self.logger.info(f"  {bucket}: {count} ({ratio:.2%})")

        return data_copy

    def _calculate_difficulty_bins(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算难度分箱

        根据文档要求：对每图计算 d = median_m |pred_prob_m - 0.5|，
        把 d 按 IID 分位切成三箱（低/中/高难度）

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 添加了difficulty_bin列的数据
        """
        data_copy = data.copy()

        # 如果只有单个模型，直接使用预测概率计算难度
        if "model_name" not in data_copy.columns or data_copy["model_name"].nunique() == 1:
            # 单模型情况：使用 |pred_prob - 0.5| 作为难度
            data_copy["difficulty_score"] = np.abs(data_copy["pred_prob"] - 0.5)
        else:
            # 多模型情况：计算每张图片在所有模型上的中位数难度
            difficulty_scores = []

            for image_id in data_copy["image_id"].unique():
                image_data = data_copy[data_copy["image_id"] == image_id]
                # 计算该图片在所有模型上的难度分数
                model_difficulties = np.abs(image_data["pred_prob"].values - 0.5)
                median_difficulty = np.median(model_difficulties)
                difficulty_scores.extend([median_difficulty] * len(image_data))

            data_copy["difficulty_score"] = difficulty_scores

        # 由于我们不再有split字段，使用全部数据来确定分位数
        iid_data = data_copy

        if len(iid_data) > 0:
            # 计算三分位数（低/中/高难度）
            q33 = iid_data["difficulty_score"].quantile(0.33)
            q67 = iid_data["difficulty_score"].quantile(0.67)

            # 分配难度箱
            data_copy["difficulty_bin"] = "MEDIUM"
            data_copy.loc[data_copy["difficulty_score"] <= q33, "difficulty_bin"] = "LOW"
            data_copy.loc[data_copy["difficulty_score"] >= q67, "difficulty_bin"] = "HIGH"
        else:
            # 如果没有IID数据，均匀分配
            data_copy["difficulty_bin"] = "MEDIUM"

        # 输出统计信息
        difficulty_counts = data_copy["difficulty_bin"].value_counts()
        self.logger.info(f"Difficulty bins分布:")
        for bin_name, count in difficulty_counts.items():
            ratio = count / len(data_copy)
            self.logger.info(f"  {bin_name}: {count} ({ratio:.2%})")

        return data_copy

    def _stratified_split_with_buckets(
        self,
        data: pd.DataFrame,
        train_ratio: float,
        cal_ratio: float,
        test_ratio: float,
        stratify_columns: List[str],
        random_seed: int,
        image_id_col: str,
    ) -> Dict[str, pd.DataFrame]:
        """基于分层桶进行数据分割

        Args:
            data: 处理后的数据
            train_ratio: 训练集比例
            cal_ratio: 校准集比例
            test_ratio: 测试集比例
            stratify_columns: 分层列
            random_seed: 随机种子
            image_id_col: 图片ID列

        Returns:
            Dict[str, pd.DataFrame]: 分割后的数据
        """
        np.random.seed(random_seed)

        # 获取唯一图片（避免同一张图片出现在不同集合中）
        unique_cols = [image_id_col] + stratify_columns
        if "image_path" in data.columns:
            unique_cols.append("image_path")

        # 确保所有分层列都存在
        missing_cols = [col for col in stratify_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"分层列不存在: {missing_cols}")

        unique_images = data[unique_cols].drop_duplicates()

        # 执行分层分割
        splits = {}
        remaining_data = unique_images.copy()

        # 计算每个分割的样本数
        n_total = len(remaining_data)
        n_train = int(n_total * train_ratio)
        n_cal = int(n_total * cal_ratio)
        n_test = n_total - n_train - n_cal

        self.logger.info(f"数据分割计划: 总计 {n_total}, 训练 {n_train}, 校准 {n_cal}, 测试 {n_test}")

        # 分层抽样训练集
        train_indices = self._stratified_sample(remaining_data, stratify_columns, n_train)
        remaining_data = remaining_data.drop(train_indices)

        # 分层抽样校准集
        cal_indices = self._stratified_sample(remaining_data, stratify_columns, n_cal)
        remaining_data = remaining_data.drop(cal_indices)

        # 剩余的作为测试集
        test_indices = remaining_data.index

        # 基于图片ID获取完整数据
        train_images = unique_images.loc[train_indices, image_id_col].tolist()
        cal_images = unique_images.loc[cal_indices, image_id_col].tolist()
        test_images = unique_images.loc[test_indices, image_id_col].tolist()

        splits["train"] = data[data[image_id_col].isin(train_images)]
        splits["calibration"] = data[data[image_id_col].isin(cal_images)]
        splits["test"] = data[data[image_id_col].isin(test_images)]

        # 输出分割统计
        for split_name, split_data in splits.items():
            self.logger.info(f"{split_name}集统计:")
            self.logger.info(f"  样本数: {len(split_data)}")
            self.logger.info(f"  唯一图片数: {split_data[image_id_col].nunique()}")

            # 输出分层统计
            for col in stratify_columns:
                if col in split_data.columns:
                    dist = split_data[col].value_counts(normalize=True)
                    self.logger.info(f"  {col}分布: {dict(dist.round(3))}")

        return splits

    def create_splits(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.6,
        cal_ratio: float = 0.2,
        test_ratio: float = 0.2,
        use_stratification: bool = True,
        random_seed: int = 42,
        min_family_ratio: float = 0.01,
    ) -> Dict[str, pd.DataFrame]:
        """创建数据分割

        Args:
            data: 输入数据DataFrame
            train_ratio: 训练集比例
            cal_ratio: 校准集比例
            test_ratio: 测试集比例
            use_stratification: 是否使用分层策略
            random_seed: 随机种子
            min_family_ratio: 最小家族比例

        Returns:
            Dict[str, pd.DataFrame]: 分割后的数据
        """
        if data is None or data.empty:
            raise ValueError("输入数据为空")

        if use_stratification and hasattr(self, "create_train_test_split_with_stratification"):
            # 使用分层策略分割
            return self.create_train_test_split_with_stratification(
                data=data,
                train_ratio=train_ratio,
                cal_ratio=cal_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                min_family_ratio=min_family_ratio,
            )
        else:
            # 使用简单分割
            return self.create_train_test_split(
                data=data, train_ratio=train_ratio, cal_ratio=cal_ratio, test_ratio=test_ratio, random_seed=random_seed
            )