import itertools
import multiprocessing
import sqlite3
from pathlib import Path
from typing import Iterator, List, Dict, Optional, Union
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock, Lock
from tqdm import tqdm
import time
from .transaction_manager import TransactionManager
from ..utils.logger import get_logger, LogContext
from .connect_pool import ConnectionPool


@dataclass
class PredictionRecord:
    """预测记录数据类"""

    image_path: str
    model_name: str
    dataset_name: str
    data_source: str
    pred_prob: float
    gt_label: int


class DatabaseManager:
    def __init__(self, db_dir: str = "data/databases", max_workers: int = 6):
        """初始化数据库管理器"""
        self.db_dir = Path(db_dir)
        self.logger = get_logger(__name__)
        self.max_connections = multiprocessing.cpu_count() * 2
        self.max_workers = max_workers

        # 存储每个数据集的连接池
        self._connection_pools: Dict[str, ConnectionPool] = {}
        self._pools_lock = RLock()

        # 线程池执行器
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="db_worker")

        # 添加数据库级别的写锁，防止同时写入同一个数据库
        self._db_write_locks: Dict[str, Lock] = {}
        self._db_locks_lock = RLock()

        # 确保数据库目录存在
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def _get_db_write_lock(self, dataset_name: str) -> Lock:
        """获取数据库写锁"""
        with self._db_locks_lock:
            if dataset_name not in self._db_write_locks:
                self._db_write_locks[dataset_name] = Lock()
            return self._db_write_locks[dataset_name]

    def _get_db_path(self, dataset_name: str, extern_name=None) -> Path:
        """获取数据集对应的数据库路径"""
        safe_name = self._sanitize_name(dataset_name)
        if extern_name:
            return self.db_dir / f"{safe_name}_{extern_name}.db"
        return self.db_dir / f"{safe_name}.db"

    def _sanitize_name(self, name: str) -> str:
        """清理名称，确保可以安全用作文件名和表名"""
        import re

        safe_name = re.sub(r"[^\w\-]", "_", name)
        if safe_name and not safe_name[0].isalpha():
            safe_name = "t_" + safe_name
        return safe_name[:60]

    def _get_connection_pool(self, dataset_name: str, extern_name=None) -> ConnectionPool:
        """获取或创建连接池"""
        with self._pools_lock:
            if dataset_name not in self._connection_pools:
                db_path = self._get_db_path(dataset_name, extern_name)
                assert db_path.parent.exists(), f"数据库目录不存在: {db_path.parent}"

                pool = ConnectionPool(
                    str(db_path),
                    max_connections=self.max_connections,
                )

                # 预热连接池并优化所有连接
                self._warmup_pool(pool)
                self._connection_pools[dataset_name] = pool

            return self._connection_pools[dataset_name]

    def _warmup_pool(self, pool: ConnectionPool):
        """预热连接池，优化所有连接"""
        with LogContext("预热数据库连接池", level="DEBUG"):
            warmup_count = max(1, self.max_connections // 2)

            try:
                # 预创建并优化连接
                for _ in range(warmup_count):
                    with pool.get_connection(readonly=False) as conn:
                        conn.commit()  # 确保PRAGMA设置生效

            except Exception as e:
                self.logger.warning(f"连接池预热失败: {e}")

    def _insert_dataset_records(self, dataset_name: str, records: pd.DataFrame, batch_size: int) -> int:
        """优化的数据集记录插入"""
        start_time = time.time()

        # 去重并准备image_info数据
        unique_images = records[["image_path", "data_source", "gt_label"]].drop_duplicates()

        # 第一步：批量插入image_info
        image_count = self._batch_insert_image_info(dataset_name, unique_images, batch_size)
        self.logger.info(f"数据集 {dataset_name} 的 image_info 插入完成: {image_count} 条记录")

        # 第二步：并入各模型的预测数据
        model_groups = records.groupby("model_name")
        total_records = 0
        for model_name, model_records in model_groups:
            self.logger.info(f"开始插入数据集 {dataset_name} 的模型 {model_name} 预测数据，共 {len(model_records)} 条记录")
            total_records += self._insert_model_predictions(dataset_name, model_name, model_records, batch_size)

        elapsed = time.time() - start_time
        self.logger.info(f"数据集 {dataset_name} 插入完成: {total_records} 条记录，耗时 {elapsed:.2f}s")

        return total_records

    def _batch_insert_image_info(self, dataset_name: str, image_info_df: pd.DataFrame, batch_size: int) -> int:
        """优化的批量插入图像信息"""
        pool = self._get_connection_pool(dataset_name)

        with pool.get_connection(readonly=False) as conn:
            # 准备数据
            image_data = [(row["image_path"], row["data_source"], row["gt_label"]) for _, row in image_info_df.iterrows()]

            with TransactionManager(conn):
                cursor = conn.cursor()

                # 检查表是否存在
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_info'")
                table_exists = cursor.fetchone() is not None
                if table_exists:
                    # 检查表中是否有数据
                    cursor.execute("SELECT COUNT(*) FROM image_info")
                    existing_count = cursor.fetchone()[0]

                    if existing_count > 0:
                        self.logger.info(f"数据集 {dataset_name} 的 image_info 表已存在 {existing_count} 条记录，跳过插入")
                        return existing_count
                    else:
                        self.logger.info(f"数据集 {dataset_name} 的 image_info 表存在但为空，继续插入数据")
                else:
                    self.logger.info(f"数据集 {dataset_name} 的 image_info 表不存在，创建新表")

                # 创建image_info表，添加更多优化索引
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS image_info (
                        image_path TEXT NOT NULL PRIMARY KEY,
                        data_source TEXT NOT NULL,
                        gt_label INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) WITHOUT ROWID
                """
                )

                # 创建image_info表的索引
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_image_info_source_created 
                    ON image_info(data_source)
                """
                )

                self.logger.info(f"确保 {dataset_name}.image_info 表存在")

                # 分批插入，显示进度
                with tqdm(total=len(image_data), desc=f"插入{dataset_name}的image_info", unit="条") as pbar:
                    for i in range(0, len(image_data), batch_size):
                        batch = image_data[i : i + batch_size]
                        cursor.executemany("INSERT OR REPLACE INTO image_info (image_path, data_source, gt_label) VALUES (?, ?, ?)", batch)
                        pbar.update(len(batch))

                        # 每10个批次中间提交一次（如果支持）
                        if (i // batch_size + 1) % 10 == 0:
                            try:
                                conn.commit()
                                # 重新开始事务
                                if not conn.in_transaction:
                                    cursor.execute("BEGIN IMMEDIATE")
                            except sqlite3.OperationalError:
                                # 如果无法中间提交，继续执行
                                pass

        return len(image_data)

    def _insert_model_predictions(self, dataset_name: str, model_name: str, model_records: pd.DataFrame, batch_size: int) -> int:
        pool = self._get_connection_pool(dataset_name)
        safe_model_name = self._sanitize_name(model_name)

        prediction_data = [(row["image_path"], row["pred_prob"]) for _, row in model_records.iterrows()]
        total_batches = len(prediction_data) // batch_size + (1 if len(prediction_data) % batch_size else 0)

        with pool.get_connection(readonly=False) as conn:
            with TransactionManager(conn):
                cursor = conn.cursor()

                # 检查表是否存在，避免重复创建
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (safe_model_name,))
                table_exists = cursor.fetchone() is not None

                if table_exists:
                    # 查询已存在记录数
                    cursor.execute(f'SELECT COUNT(*) FROM "{safe_model_name}"')
                    existing_count = cursor.fetchone()[0]
                    self.logger.info(f"表 {dataset_name}.{safe_model_name} 已存在 {existing_count} 条记录")
                    if existing_count >= len(prediction_data):
                        self.logger.info(f"表 {dataset_name}.{safe_model_name} 已包含所有记录，跳过插入")
                        return existing_count
                    self.logger.warning(f"表 {dataset_name}.{safe_model_name} 部分记录已存在，将更新或插入新记录")
                else:
                    self.logger.info(f"表 {dataset_name}.{safe_model_name} 不存在，创建新表")
                    cursor.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS "{safe_model_name}" (
                            image_path TEXT NOT NULL PRIMARY KEY,
                            pred_prob REAL NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (image_path) REFERENCES image_info (image_path) ON DELETE CASCADE
                        ) WITHOUT ROWID
                    """
                    )

                    try:
                        self.logger.info(f"创建表 {dataset_name}.{safe_model_name}的索引, 以优化查询性能")
                        # 分开创建索引，提高容错性
                        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{safe_model_name}_image_path ON "{safe_model_name}"(image_path)')
                        cursor.execute(
                            f'CREATE INDEX IF NOT EXISTS idx_{safe_model_name}_image_pred_prob ON "{safe_model_name}"(image_path, pred_prob)'
                        )
                    except sqlite3.Error as e:
                        self.logger.warning(f"创建索引时发生错误: {str(e)}")

            with TransactionManager(conn):
                # 无论表是否已存在，都执行插入或更新操作
                with tqdm(total=total_batches, desc=f"插入{dataset_name}.{safe_model_name}预测", unit="批次") as pbar:
                    for i in range(0, len(prediction_data), batch_size):
                        batch = prediction_data[i : i + batch_size]
                        cursor.executemany(f'INSERT OR REPLACE INTO "{safe_model_name}" (image_path, pred_prob) VALUES (?, ?)', batch)
                        pbar.update(1)

                        # 每5个批次尝试中间提交一次
                        if (i // batch_size + 1) % 5 == 0:
                            conn.commit()
                            cursor.execute("BEGIN IMMEDIATE")

        return len(prediction_data)

    def _get_dataset_predictions(
        self, dataset_name: str, model_names: Optional[List[str]] = None, data_source: Optional[str] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """优化的数据集预测记录查询"""
        try:
            # 获取要查询的表名
            table_names = [self._sanitize_name(name) for name in model_names]
            if not table_names:
                raise ValueError("必须提供至少一个模型名称")

            # 使用更高效的查询策略
            all_results = []
            self.logger.info(f"查询数据集 {dataset_name} 的模型 {model_names} 预测记录")

            # 并发查询多个模型表 -> 顺序查询多个模型表
            for table_name in table_names:
                df = self._query_model_table(dataset_name, table_name, limit)
                all_results.append(df)
            result = pd.concat(all_results, ignore_index=True)
            return result

        except Exception as e:
            self.logger.error(f"查询数据集 {dataset_name} 失败: {e}")
            return pd.DataFrame()

    def _query_model_table(
        self,
        dataset_name: str,
        table_name: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """在线程池中执行的分片查询方法"""
        # 为每个线程获取独立的数据库连接
        pool = self._get_connection_pool(dataset_name)
        base_query = f"""
SELECT
    '{table_name}' as model_name,
    p.*
FROM 
    "{table_name}" p
"""
        with pool.get_connection(readonly=True) as conn:
            # 判断是否需要分片查询
            if limit and limit <= 100000:
                # 小数据量直接查询
                query = f"{base_query} LIMIT {limit}"
                res = pd.read_sql_query(query, conn)
            else:
                # 大数据量或无限制时使用分片查询
                res = self._chunked_query_to_dataframe(conn, dataset_name, table_name, base_query, limit)
        return res

    def _chunked_query_to_dataframe(
        self,
        conn: sqlite3.Connection,
        dataset_name: str,
        table_name: str,
        base_query: str,
        total_limit: Optional[int] = None,
        chunk_size: int = 50000,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """使用生成器和库函数执行分片查询"""

        def get_total_count() -> Optional[int]:
            """获取查询的总记录数"""
            try:
                count_query = f'SELECT COUNT(*) FROM "{table_name}"'
                result = conn.execute(count_query).fetchone()
                total_count = result[0] if result else 0
                return min(total_count, total_limit) if total_limit else total_count
            except Exception as e:
                self.logger.warning(f"无法获取总记录数{dataset_name}-{table_name}: {e}")
                return None

        def query_chunks() -> Iterator[pd.DataFrame]:
            """生成器函数，逐块产生查询结果"""
            offset = 0
            total_read = 0
            total_count = get_total_count() if show_progress else None

            if total_count is None:
                self.logger.warning(f"无法获取总记录数, 直接返回空的pd.Dataframe: {dataset_name}-{table_name}")
                return pd.DataFrame()  # 如果无法获取总数，直接返回空

            pbar = (
                tqdm(total=total_count, desc=f"查询表[{table_name}]IN[{dataset_name}]", unit="行", unit_scale=True, leave=True)
                if show_progress
                else None
            )
            try:
                while True:
                    current_limit = min(chunk_size, total_limit - total_read) if total_limit else chunk_size
                    chunked_query = f"{base_query} LIMIT {current_limit} OFFSET {offset}"

                    try:
                        chunk_df = pd.read_sql_query(chunked_query, conn)
                    except Exception as e:
                        self.logger.error(f"查询分片失败: {e}")
                        break

                    if chunk_df.empty:
                        break

                    yield chunk_df
                    total_read += len(chunk_df)
                    offset += current_limit

                    if pbar:
                        pbar.update(len(chunk_df))
                        if total_limit and total_read >= total_limit:
                            break
            finally:
                if pbar:
                    pbar.close()

        chunks = [chunk for chunk in query_chunks()]
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def get_predictions(
        self,
        dataset_names: Optional[Union[str, List[str]]] = None,
        model_names: Optional[Union[str, List[str]]] = None,
        data_source: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """高性能并发查询预测记录"""
        start_time = time.time()

        # 参数预处理
        def _normalize_names(names, default):
            if names is None:
                return default
            if isinstance(names, str):
                return [names]
            return list(names)

        dataset_names = _normalize_names(dataset_names, [db_file.stem for db_file in self.db_dir.glob("*.db")])
        model_names = _normalize_names(model_names, [])

        if not model_names:
            self.logger.warning("未指定模型名称，查询将失败")
            return pd.DataFrame()

        # 并发查询各数据集
        future_to_name = {}
        for dataset_name in dataset_names:
            future = self._executor.submit(self._get_dataset_predictions, dataset_name, model_names, data_source, limit)
            future_to_name[future] = dataset_name

        all_results = []
        for future in as_completed(future_to_name):
            dataset_name = future_to_name[future]
            try:
                result_df = future.result()
                all_results.append(result_df)
            except Exception:
                self.logger.warning(f"查询数据集 {dataset_name} 失败", exc_info=True)
        final_result = pd.concat(all_results, ignore_index=True)
        elapsed = time.time() - start_time
        self.logger.info(f"并发查询完成，共获取 {len(final_result)} 条记录，耗时 {elapsed:.2f}s")
        return final_result

    def get_image_info(self, dataset_names: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        并发查询各数据库的 image_info 表
        返回: 包含 image_path, data_source, gt_label, dataset_name 的 DataFrame
        """
        start_time = time.time()

        # 参数预处理 - 与 get_predictions 保持一致
        if dataset_names is None:
            dataset_names = [db_file.stem for db_file in self.db_dir.glob("*.db")]
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        else:
            dataset_names = list(dataset_names)

        if not dataset_names:
            self.logger.warning("未发现可用数据库，返回空结果")
            return pd.DataFrame()

        # 单库查询函数
        def _fetch_image_info_for_dataset(dataset_name: str) -> pd.DataFrame:
            # 若数据库文件不存在，则跳过，避免无意创建新库
            db_path = self._get_db_path(dataset_name)
            if not db_path.exists():
                self.logger.warning(f"数据库文件不存在: {db_path}, 跳过")
                return pd.DataFrame()

            pool = self._get_connection_pool(dataset_name)
            with pool.get_connection(readonly=True) as conn:
                # 判断表是否存在
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_info'")
                if not cursor.fetchone():
                    self.logger.warning(f"{dataset_name}.db 中不存在表 image_info，跳过")
                    return pd.DataFrame()

                # 使用分片查询，兼顾大表和内存
                base_query = """
                SELECT
                    image_path,
                    data_source,
                    gt_label
                FROM image_info
                """
                df = self._chunked_query_to_dataframe(
                    conn=conn,
                    dataset_name=dataset_name,
                    table_name="image_info",
                    base_query=base_query,
                    total_limit=None,  # 读取全部
                    chunk_size=50000,
                    show_progress=True,
                )
            if not df.empty:
                df["dataset_name"] = dataset_name
            else:
                raise ValueError(f"数据集{dataset_name}, 未获取到image_info信息")
            return df

        # 并发提交任务
        future_to_dataset = {}
        for name in dataset_names:
            future = self._executor.submit(_fetch_image_info_for_dataset, name)
            future_to_dataset[future] = name

        # 收集结果
        results = []
        for future in as_completed(future_to_dataset):
            name = future_to_dataset[future]
            try:
                df: pd.DataFrame = future.result()
                if not df.empty:
                    results.append(df)
            except Exception as e:
                self.logger.warning(f"查询数据集 {name} 失败: {e}", exc_info=True)

        final_df = pd.concat(results, ignore_index=True)
        elapsed = time.time() - start_time
        self.logger.info(f"并发查询 image_info 完成，共获取 {len(final_df)} 条记录，耗时 {elapsed:.2f}s")
        return final_df

    def insert_predictions(self, validated_records: pd.DataFrame, batch_size: int = 2000) -> int:
        """高性能并发批量插入预测记录"""
        start_time = time.time()

        with LogContext(f"高性能批量插入预测记录 ({len(validated_records)} 条)", level="INFO"):
            # 按数据集分组
            dataset_groups = validated_records.groupby("dataset_name")

            future_to_name = {}
            for dataset_name, dataset_records in dataset_groups:
                future = self._executor.submit(self._insert_dataset_records, dataset_name, dataset_records, batch_size)
                future_to_name[future] = dataset_name

            # 收集结果
            total_inserted = 0
            for future in as_completed(future_to_name):
                try:
                    inserted_count = future.result()
                    total_inserted += inserted_count
                    name = future_to_name[future]
                    self.logger.info(f"数据集 {name} 插入完成: {inserted_count} 条记录")
                except Exception as e:
                    self.logger.error(f"数据集插入失败: {e}")
                    raise

            elapsed = time.time() - start_time
            self.logger.info(f"所有数据插入完成，总计 {total_inserted} 条记录，耗时 {elapsed:.2f}s")
            return total_inserted

    def load_clustering_results(self, dataset_names: List[str], cluster_names: Optional[List[str]] = None) -> pd.DataFrame:
        all_results = []
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if isinstance(cluster_names, str):
            cluster_names = [cluster_names]

        for dataset_name in dataset_names:
            pool = self._get_connection_pool(dataset_name)
            with pool.get_connection(readonly=True) as conn:
                cursor = conn.cursor()

                # 检查表是否存在
                cursor.execute(f'PRAGMA table_info("image_info")')
                columns = {row[1] for row in cursor.fetchall()}

                # 如果 cluster_names 为 None，则加载所有以 cluster_ 开头的列
                if cluster_names is None:
                    cluster_names = [col for col in columns if col.startswith("cluster_")]

                missing_columns = [name for name in cluster_names if name not in columns]
                if missing_columns:
                    self.logger.warning(f"表 {dataset_name}.image_info 中不存在列 {missing_columns}，跳过读取")

                existing_columns = [name for name in cluster_names if name in columns]
                if not existing_columns:
                    self.logger.warning(f"表 {dataset_name}.image_info 中不存在任何指定的聚类列，跳过读取")
                    continue

                # 查询聚类结果
                query = f"""
                SELECT image_path, {', '.join([f'"{name}" AS "{name}"' for name in existing_columns])}
                FROM "image_info"
                """
                try:
                    df = pd.read_sql_query(query, conn)
                    all_results.append(df)
                except Exception as e:
                    self.logger.error(f"读取 {dataset_name}.image_info 的聚类结果失败: {e}")

        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def save_clustering_results(self, df: pd.DataFrame, cluster_name: str, batch_size: int = 1000):
        """
        将聚类结果写入指定模型名的表中

        参数:
            df: 包含 dataset_name, model_name, image_path 和 cluster_name 的 DataFrame
            cluster_name: 聚类列名
            batch_size: 批量写入的大小
        """
        if df.empty:
            self.logger.warning("输入数据为空，跳过保存聚类结果")
            return

        required_columns = {"dataset_name", "model_name", "image_path", cluster_name}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"数据框缺少必需的列: {missing_columns}")

        for dataset_name, group_data in df.groupby("dataset_name"):
            pool = self._get_connection_pool(dataset_name)

            with pool.get_connection(readonly=False) as conn:
                cursor = conn.cursor()

                # 检查并添加聚类列
                cursor.execute(f'PRAGMA table_info("image_info")')
                columns = {row[1] for row in cursor.fetchall()}
                if cluster_name not in columns:
                    self.logger.info(f"添加聚类列 {cluster_name} 到 {dataset_name}.image_info 表")
                    cursor.execute(f'ALTER TABLE "image_info" ADD COLUMN "{cluster_name}" INTEGER')
                    conn.commit()

                # 准备批量更新数据
                clustering_data = [(int(row[cluster_name]), row["image_path"]) for _, row in group_data.iterrows()]
                total_batches = (len(clustering_data) + batch_size - 1) // batch_size

                with TransactionManager(conn):
                    with tqdm(total=total_batches, desc=f"插入 {dataset_name}.image_info 聚类结果", unit="批次") as pbar:
                        for i in range(0, len(clustering_data), batch_size):
                            batch = clustering_data[i : i + batch_size]
                            cursor.executemany(
                                f'UPDATE "image_info" SET "{cluster_name}" = ? WHERE image_path = ?',
                                batch,
                            )
                            pbar.update(1)

                            # 定期提交以避免事务过大
                            if (i // batch_size + 1) % 10 == 0:
                                conn.commit()

    def insert_calibration_results(self, df: pd.DataFrame, batch_size: int = 1000):
        """
        修正版本：顺序保存校准结果，避免数据库锁定问题
        Args:
            df: 必须包含 'dataset_name', 'model_name', 'image_path', 'calibration_prob'
            batch_size: 批量更新的大小
        """
        if df.empty:
            self.logger.warning("输入数据框为空，跳过保存")
            return

        required_columns = ["dataset_name", "model_name", "image_path", "calibration_prob"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据框缺少必需的列: {missing_columns}")

        start_time = time.time()

        # 统计信息
        dataset_count = df["dataset_name"].nunique()
        model_count = df.groupby(["dataset_name", "model_name"]).ngroups

        self.logger.info(f"开始保存校准结果: {dataset_count} 个数据集, {model_count} 个模型表")

        # 修改策略：按数据集顺序处理，每个数据集内并发处理模型
        dataset_groups = df.groupby("dataset_name")
        total_success = 0
        total_failed = 0

        for dataset_name, dataset_group in dataset_groups:
            try:
                # 使用数据库级别的写锁，确保同一时间只有一个线程写入该数据库
                db_lock = self._get_db_write_lock(dataset_name)

                with db_lock:
                    success_count = self._save_calibration_for_dataset_sequential(dataset_name, dataset_group, batch_size)
                    total_success += success_count

            except Exception as e:
                total_failed += 1
                self.logger.error(f"数据集 {dataset_name} 校准结果保存失败: {e}")

        elapsed = time.time() - start_time

        self.logger.info(
            f"校准结果保存完成: 成功 {total_success} 个模型表, "
            f"失败 {total_failed} 个数据集, "
            f"耗时 {elapsed:.2f}s, "
            f"速度 {len(df) / elapsed:.0f} 条/秒"
        )

    def _save_calibration_for_dataset_sequential(self, dataset_name: str, dataset_group: pd.DataFrame, batch_size: int) -> int:
        """
        为单个数据集顺序保存所有模型的校准结果，避免数据库锁定
        """
        pool = self._get_connection_pool(dataset_name)
        model_groups = dataset_group.groupby("model_name")
        success_count = 0

        for model_name, model_group in model_groups:
            try:
                self._save_calibration_for_model_sync(dataset_name, model_name, model_group, batch_size, pool)
                success_count += 1

            except Exception as e:
                self.logger.error(f"{dataset_name}.{model_name} 校准结果保存失败: {e}")
                # 继续处理其他模型，不要因为一个模型失败就停止

        self.logger.info(f"{dataset_name} 校准结果保存完成: {success_count}/{len(model_groups)} 个模型")
        return success_count

    def _save_calibration_for_model_sync(
        self,
        dataset_name: str,
        model_name: str,
        model_group: pd.DataFrame,
        batch_size: int,
        pool: ConnectionPool,
    ):
        """
        同步保存单个模型的校准结果，简化事务管理
        """
        safe_model_name = self._sanitize_name(model_name)

        # 使用重试机制处理数据库锁定
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                with pool.get_connection(readonly=False) as conn:
                    cursor = conn.cursor()

                    # 检查并添加 calibration_prob 列
                    cursor.execute(f'PRAGMA table_info("{safe_model_name}")')
                    columns = [row[1] for row in cursor.fetchall()]

                    if "calibration_prob" not in columns:
                        self.logger.info(f"添加 calibration_prob 列到 {dataset_name}.{safe_model_name} 表")
                        cursor.execute(f'ALTER TABLE "{safe_model_name}" ADD COLUMN calibration_prob REAL DEFAULT NULL')
                        conn.commit()

                    # 准备更新数据
                    update_data = [(float(row["calibration_prob"]), str(row["image_path"])) for _, row in model_group.iterrows()]

                    # 使用更小的批次和更短的事务
                    small_batch_size = min(batch_size, 500)  # 限制批次大小
                    total_batches = len(update_data) // small_batch_size + (1 if len(update_data) % small_batch_size else 0)

                    with tqdm(total=total_batches, desc=f"{dataset_name}.{safe_model_name}", unit="批次", leave=False) as pbar:

                        for i in range(0, len(update_data), small_batch_size):
                            batch = update_data[i : i + small_batch_size]

                            # 每个小批次使用独立的事务
                            with TransactionManager(conn):
                                cursor.executemany(
                                    f'UPDATE "{safe_model_name}" SET calibration_prob = ? WHERE image_path = ?',
                                    batch,
                                )

                            pbar.update(1)

                            # 短暂休息，让其他操作有机会执行
                            if i % (small_batch_size * 5) == 0:
                                time.sleep(0.001)

                    # 成功完成，退出重试循环
                    self.logger.info(f"{dataset_name}.{safe_model_name} 校准结果保存完成: {len(update_data)} 条记录")
                    break

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and retry_count < max_retries - 1:
                    retry_count += 1
                    wait_time = 2**retry_count  # 指数退避
                    self.logger.warning(f"{dataset_name}.{safe_model_name} 数据库被锁定，{wait_time}秒后重试 ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                self.logger.error(f"保存 {dataset_name}.{safe_model_name} 校准结果失败: {e}")
                raise

    def close_all_connections(self):
        """关闭所有连接和线程池"""
        # 关闭线程池
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)

        # 关闭所有连接池
        with self._pools_lock:
            for pool in self._connection_pools.values():
                pool.close_all()
            self._connection_pools.clear()

        self.logger.info("所有数据库连接已关闭")

    def __del__(self):
        """析构函数，确保清理资源"""
        try:
            self.close_all_connections()
        except:
            pass
