import sqlite3


class TransactionManager:
    """事务管理器，避免嵌套事务问题"""

    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self.in_transaction = False

    def __enter__(self):
        # 检查是否已经在事务中
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master LIMIT 1")  # 测试查询

        try:
            # 如果没有在事务中，开始新事务
            if not self.in_transaction:
                cursor.execute("BEGIN IMMEDIATE")
                self.in_transaction = True
        except sqlite3.OperationalError as e:
            if "cannot start a transaction within a transaction" in str(e):
                # 已经在事务中，不需要开始新事务
                self.in_transaction = False
            else:
                raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.in_transaction:
            try:
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
            finally:
                self.in_transaction = False
