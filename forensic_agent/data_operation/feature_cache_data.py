import json
import pandas as pd
from typing import Dict, List, Optional
import sqlite3

from .database_manager import DatabaseManager
from .transaction_manager import TransactionManager
from ..utils.logger import get_logger, LogContext
from ..utils.custom_json_encoder import CustomJsonEncoder


class FeatureCacheData:
    """特征数据缓存类，在每个dataset数据库中管理features表"""

    def __init__(self, database_manager, batch_size: int = 1000):
        """
        初始化特征数据缓存器

        Args:
            database_manager: DatabaseManager实例
            batch_size: 批处理大小
        """
        self.logger = get_logger(__name__)
        self.db_manager: DatabaseManager = database_manager
        self.batch_size = batch_size

        # 统计信息
        self.stats = {"cache_hits": 0, "cache_misses": 0}

        # 已初始化的数据集记录
        self._initialized_datasets = set()

    def _init_feature_table(self, dataset_name: str):
        """为指定数据集初始化特征表结构"""
        pool = self.db_manager._get_connection_pool(dataset_name, "features")
        if dataset_name in self._initialized_datasets:
            return pool

        with LogContext(f"初始化数据集 {dataset_name} 的特征表", level="DEBUG"):
            try:
                with pool.get_connection(readonly=False) as conn:
                    with TransactionManager(conn):
                        cursor = conn.cursor()

                        # 创建特征表，image_path和feature_name为联合主键
                        cursor.execute(
                            """
                            CREATE TABLE IF NOT EXISTS features (
                                image_path TEXT NOT NULL,
                                feature_name TEXT NOT NULL,
                                feature_value TEXT NOT NULL,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                PRIMARY KEY (image_path, feature_name)
                            ) WITHOUT ROWID
                            """
                        )
                        # 索引定义
                        indexes = [
                            ("idx_features_image_path", "image_path"),
                            ("idx_features_name", "feature_name"),
                            ("idx_features_composite", "image_path, feature_name"),
                        ]

                        # 批量创建索引
                        for index_name, columns in indexes:
                            cursor.execute(
                                f"""
                                CREATE INDEX IF NOT EXISTS {index_name} 
                                ON features({columns})
                            """
                            )

                self._initialized_datasets.add(dataset_name)
                self.logger.debug(f"数据集 {dataset_name} 特征表初始化完成")

            except Exception as e:
                self.logger.error(f"数据集 {dataset_name} 特征表初始化失败: {e}")
                raise

        return pool

    def save_cache(self, features_dict: Dict[str, Dict[str, List]], dataset_name: str):
        """
        保存特征缓存到指定数据集

        Args:
            features_dict: {image_path: {feature_name: feature_value}}
            dataset_name: 数据集名称
        """
        if not features_dict:
            self.logger.warning("特征字典为空，跳过保存")
            return

        # 确保特征表已初始化
        pool = self._init_feature_table(dataset_name)

        try:
            with pool.get_connection(readonly=False) as conn:
                # 准备批量插入数据
                rows = []
                warning_flag = False
                duplicate_keys = set()

                for image_path, features in features_dict.items():
                    # 验证image_path不能为空
                    if not image_path or pd.isna(image_path):
                        self.logger.warning(f"跳过空的image_path: {image_path}")
                        continue

                    for feature_name, value in features.items():
                        # 验证feature_name不能为空
                        if not feature_name or pd.isna(feature_name):
                            self.logger.warning(f"跳过空的feature_name: {feature_name} (image_path: {image_path})")
                            continue

                        # 检查联合主键重复
                        key = (image_path, feature_name)
                        if key in duplicate_keys:
                            self.logger.warning(f"检测到重复的联合主键: {key}，将覆盖之前的值")
                        duplicate_keys.add(key)

                        if value is not None:  # 只存储非空值
                            try:
                                # 序列化特征值
                                serialized_value = json.dumps(value, cls=CustomJsonEncoder)
                                rows.append((image_path, feature_name, serialized_value))
                            except (TypeError, ValueError) as e:
                                self.logger.warning(f"无法序列化特征值 {feature_name} for {image_path}: {e}")
                                warning_flag = True
                                continue

                if warning_flag:
                    self.logger.warning("存在无法序列化的特征值，已跳过存储")

                if rows:
                    with TransactionManager(conn):
                        cursor = conn.cursor()

                        with LogContext(f"保存 {dataset_name} 特征缓存 ({len(rows)} 条记录)", level="INFO"):
                            for i in range(0, len(rows), self.batch_size):
                                batch = rows[i : i + self.batch_size]
                                try:
                                    cursor.executemany(
                                        """
                                        INSERT OR REPLACE INTO features (image_path, feature_name, feature_value)
                                        VALUES (?, ?, ?)
                                        """,
                                        batch,
                                    )
                                except sqlite3.IntegrityError as e:
                                    self.logger.error(f"联合主键约束违反: {e}")
                                    # 尝试逐条插入以识别具体的问题记录
                                    for row in batch:
                                        try:
                                            cursor.execute(
                                                """
                                                INSERT OR REPLACE INTO features (image_path, feature_name, feature_value)
                                                VALUES (?, ?, ?)
                                                """,
                                                row,
                                            )
                                        except sqlite3.IntegrityError as row_error:
                                            self.logger.error(f"记录插入失败 {row[:2]}: {row_error}")
                                            continue

                                # 每10个批次中间提交一次
                                if (i // self.batch_size + 1) % 10 == 0:
                                    try:
                                        conn.commit()
                                        if not conn.in_transaction:
                                            cursor.execute("BEGIN IMMEDIATE")
                                    except sqlite3.OperationalError:
                                        pass

                    self.logger.info(f"已保存 {len(rows)} 条特征记录到数据集 {dataset_name}")

        except Exception as e:
            self.logger.error(f"保存特征缓存失败: {e}")
            raise

    def load_cache(self, image_paths: pd.Series, feature_names: List[str], dataset_name: str):
        # 重置统计信息
        self.stats = {"cache_hits": 0, "cache_misses": 0}

        # 去重并过滤有效的image_paths
        unique_image_paths = image_paths.dropna().unique().tolist()

        if not unique_image_paths:
            raise ValueError("没有有效的'image_path'")

        if not feature_names:
            raise ValueError("特征名列表不能为空")

        # 过滤掉空的feature_names
        valid_feature_names = [name for name in feature_names if name and not pd.isna(name)]
        if not valid_feature_names:
            raise ValueError("没有有效的特征名")

        # 确保特征表已初始化
        pool = self._init_feature_table(dataset_name)
        with pool.get_connection(readonly=True) as conn:
            # 使用参数化查询
            placeholders = ", ".join([f"'{name}'" for name in valid_feature_names])  # 将特征名硬编码到查询中
            cache_query = f"""
                SELECT f.image_path, f.feature_name, f.feature_value
                FROM features f
                WHERE f.feature_name IN ({placeholders})
            """
            cache_df = self.db_manager._chunked_query_to_dataframe(conn, dataset_name, "features", cache_query)

        if cache_df.empty:
            missing_df = pd.MultiIndex.from_product(
                [image_paths, valid_feature_names],
                names=["image_path", "feature_name"],
            ).to_frame(index=False)
            return pd.DataFrame(), missing_df.groupby("image_path")["feature_name"].apply(list).to_dict()

        # 校验是否存在空值
        missing_image_paths = set(image_paths) - set(cache_df["image_path"].unique())
        if missing_image_paths:
            self.logger.debug(f"未命中缓存的 image_path 数量: {len(missing_image_paths)}")
        cache_miss_records: Dict[str, List] = {img_path: valid_feature_names for img_path in missing_image_paths}

        # 反序列化feature_value
        cache_df["deserialized_value"] = cache_df["feature_value"].apply(lambda x: json.loads(x) if pd.notna(x) else None)
        # 空值的记录视为未命中缓存
        empty_groups = cache_df[cache_df["deserialized_value"].isna()].groupby("image_path")["feature_name"].apply(list)
        cache_miss_records.update(empty_groups.to_dict())

        # 统计信息
        total_combinations = len(image_paths) * len(valid_feature_names)
        hits = cache_df["deserialized_value"].notna().sum()
        misses = total_combinations - hits

        self.stats["cache_hits"] += hits
        self.stats["cache_misses"] += misses

        self.logger.debug(
            f"数据集 {dataset_name} 缓存查询完成: 命中 {hits}/{total_combinations} 个组合 " f"({hits/total_combinations*100:.1f}%)"
        )

        # 将cache_df转换为 image_path, feature_name1, feature_name2, ... 的格式
        pivot_df = cache_df.pivot(index="image_path", columns="feature_name", values="deserialized_value").reset_index()
        pivot_df.columns.name = None
        self.db_manager.close_all_connections()
        return pivot_df, cache_miss_records

    def get_feature_keys(self, dataset_name: str) -> List[tuple]:
        """
        获取指定数据集中所有的联合主键 (image_path, feature_name)

        Args:
            dataset_name: 数据集名称

        Returns:
            List[tuple]: 联合主键列表
        """
        pool = self._init_feature_table(dataset_name)
        with pool.get_connection(readonly=True) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT image_path, feature_name 
                FROM features 
                ORDER BY image_path, feature_name
                """
            )
            return cursor.fetchall()

    def delete_features(self, keys_to_delete: List[tuple], dataset_name: str):
        """
        根据联合主键删除特征记录

        Args:
            keys_to_delete: 要删除的联合主键列表 [(image_path, feature_name), ...]
            dataset_name: 数据集名称
        """
        if not keys_to_delete:
            return

        pool = self._init_feature_table(dataset_name)
        try:
            with pool.get_connection(readonly=False) as conn:
                with TransactionManager(conn):
                    cursor = conn.cursor()

                    for i in range(0, len(keys_to_delete), self.batch_size):
                        batch = keys_to_delete[i : i + self.batch_size]
                        cursor.executemany(
                            """
                            DELETE FROM features 
                            WHERE image_path = ? AND feature_name = ?
                            """,
                            batch,
                        )

            self.logger.info(f"已删除 {len(keys_to_delete)} 个特征记录从数据集 {dataset_name}")

        except Exception as e:
            self.logger.error(f"删除特征记录失败: {e}")
            raise

    def clear_cache(self, dataset_name: Optional[str] = None):
        """
        清空特征缓存

        Args:
            dataset_name: 数据集名称，如果为None则清空所有数据集的特征缓存
        """
        if dataset_name:
            self._clear_dataset_cache(dataset_name)
        else:
            self._clear_all_caches()

    def _clear_dataset_cache(self, dataset_name: str):
        """清空指定数据集的特征缓存"""
        try:
            pool = self.db_manager._get_connection_pool(dataset_name)
            with pool.get_connection(readonly=False) as conn:
                with TransactionManager(conn):
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM features")

                self.logger.info(f"数据集 {dataset_name} 的特征缓存已清空")

        except Exception as e:
            self.logger.error(f"清空数据集 {dataset_name} 特征缓存失败: {e}")

    def _clear_all_caches(self):
        """清空所有数据集的特征缓存"""
        if hasattr(self.db_manager, "db_dir"):
            db_files = list(self.db_manager.db_dir.glob("*.db"))
            dataset_names = [db_file.stem for db_file in db_files]

            cleared_count = 0
            for dataset_name in dataset_names:
                try:
                    self._clear_dataset_cache(dataset_name)
                    cleared_count += 1
                except Exception as e:
                    self.logger.error(f"清空数据集 {dataset_name} 失败: {e}")

            self.logger.info(f"已清空 {cleared_count}/{len(dataset_names)} 个数据集的特征缓存")

        # 重置统计信息
        self.stats = {"cache_hits": 0, "cache_misses": 0}

    def get_cache_statistics(self, dataset_name: str) -> Dict:
        """
        获取指定数据集的缓存统计信息

        Args:
            dataset_name: 数据集名称

        Returns:
            Dict: 统计信息字典
        """
        pool = self._init_feature_table(dataset_name)
        with pool.get_connection(readonly=True) as conn:
            cursor = conn.cursor()

            # 获取基本统计
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT image_path) as unique_images,
                    COUNT(DISTINCT feature_name) as unique_features
                FROM features
                """
            )
            basic_stats = cursor.fetchone()

            # 获取每个特征的记录数
            cursor.execute(
                """
                SELECT feature_name, COUNT(*) as count 
                FROM features 
                GROUP BY feature_name 
                ORDER BY count DESC
                """
            )
            feature_counts = dict(cursor.fetchall())

            return {
                "total_records": basic_stats[0],
                "unique_images": basic_stats[1],
                "unique_features": basic_stats[2],
                "feature_counts": feature_counts,
                "session_cache_hits": self.stats["cache_hits"],
                "session_cache_misses": self.stats["cache_misses"],
            }
