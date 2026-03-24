import time
from .logger import get_logger


class ProgressLogger:
    """进度日志器"""

    def __init__(self, total: int, desc: str = "Progress", logger_name: str = None, log_interval: int = 10):
        """初始化进度日志器

        Args:
            total: 总数
            desc: 描述
            logger_name: 日志器名称
            log_interval: 日志间隔（百分比）
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.logger = get_logger(logger_name)
        self.log_interval = log_interval
        self.last_logged_percent = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """更新进度

        Args:
            n: 增量
        """
        self.current = min(self.current + n, self.total)
        percent = (self.current / self.total) * 100

        if percent - self.last_logged_percent >= self.log_interval or percent >= 100:
            elapsed = time.time() - self.start_time

            if percent > 0:
                eta = elapsed * (100 - percent) / percent
                self.logger.info(
                    f"{self.desc}: {percent:.1f}% ({self.current}/{self.total}) " f"- 已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s"
                )
            else:
                self.logger.info(f"{self.desc}: {percent:.1f}% ({self.current}/{self.total})")

            self.last_logged_percent = percent
