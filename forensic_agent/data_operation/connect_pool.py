import sqlite3
from contextlib import contextmanager
from queue import Queue, Empty
from threading import RLock


class ConnectionPool:
    """线程安全的连接池 - 优化版本"""

    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = Queue(maxsize=max_connections)
        self._created_count = 0
        self._lock = RLock()  # 使用可重入锁
        self._closed = False

        # 预创建连接
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self.max_connections):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """创建新的数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0, isolation_level=None)  # 增加超时时间  # 自动提交模式
        conn.row_factory = sqlite3.Row

        # 优化设置 - 针对并发优化
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # 入时不阻塞读取
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")  # 64MB缓存
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 1073741824")  # 256MB内存映射
        conn.execute("PRAGMA optimize")  # 30秒忙等待

        return conn

    @contextmanager
    def get_connection(self, timeout: float = 60.0, readonly: bool = False):
        """获取连接，支持超时和读写分离"""
        if self._closed:
            raise RuntimeError("连接池已关闭")

        conn = None
        try:
            conn = self._pool.get(timeout=timeout)

            # 为只读操作设置优化
            if readonly:
                conn.execute("BEGIN DEFERRED")
            else:
                conn.execute("BEGIN IMMEDIATE")

            yield conn
        except Empty:
            raise TimeoutError(f"获取数据库连接超时 ({timeout}s)")
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            if conn:
                try:
                    if not readonly:
                        conn.commit()
                except:
                    conn.rollback()
                finally:
                    if not self._closed:
                        self._pool.put(conn)

    def close_all(self):
        """关闭所有连接"""
        with self._lock:
            self._closed = True
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break
                except:
                    pass
