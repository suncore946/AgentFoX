"""Loguru logger manager for AgentFoX.

中文说明: 日志默认写入用户配置目录, 不包含图片内容或密钥。
English: Logs are written to the user-configured directory by default and do not
contain image content or credentials.
"""

from loguru import logger
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys
from .base_manager import BaseManager


@dataclass
class LoggerConfig:
    """Logger configuration.

    中文说明: 支持 file_path 或 log_dir + file_name 两种写法。
    English: Supports either file_path or log_dir + file_name.
    """

    name: str = "AgentFoX"
    level: str = "INFO"
    file_path: Optional[str] = None
    log_dir: str = "./logs"
    file_name: str = "agentfox.log"
    format: str = "detailed"
    rotation: str = "10 MB"
    retention: str = "7 days"
    compression: str = "zip"

    def __post_init__(self):
        """Normalize log destination.

        中文说明: 若未显式 file_path, 则由 log_dir/file_name 拼出日志文件路径。
        English: If file_path is not explicit, log_dir/file_name builds the log
        file path.
        """
        self.level = self.level.upper()
        if self.file_path is None and self.log_dir:
            self.file_path = str(Path(self.log_dir) / self.file_name)
        if self.file_path:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def detailed_format(self):
        """Return detailed log format.

        中文说明: 详细格式包含文件名、函数和行号。
        English: Detailed format includes file, function, and line number.
        """
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    @property
    def simple_format(self):
        """Return compact log format.

        中文说明: 简洁格式适合命令行批处理。
        English: Compact format is suitable for CLI batch runs.
        """
        return "<level>{level}</level> | {time:YYYY-MM-DD HH:mm:ss} | {name} | {message}"

    @property
    def log_format(self):
        """Choose the active log format.

        中文说明: format=detailed 时使用详细格式, 否则使用简洁格式。
        English: Uses detailed format when format=detailed, otherwise compact.
        """
        return self.detailed_format if self.format == "detailed" else self.simple_format


class LoggerManager(BaseManager):
    """Manage a shared Loguru logger.

    中文说明: 初始化时移除默认 handler, 防止重复打印。
    English: Removes default handlers during initialization to avoid duplicated
    log lines.
    """

    def __init__(self, config: dict, is_debug: bool = False):
        """Initialize logger from YAML config.

        中文说明: is_debug 会把日志级别提升为 DEBUG。
        English: is_debug raises the log level to DEBUG.
        """
        config = dict(config or {})
        if is_debug:
            config["level"] = "DEBUG"
        else:
            config["level"] = config.get("level", "INFO").upper()
        self.config = LoggerConfig(**config)
        self._logger = self._setup_logger()

    def _setup_logger(self):
        """Configure console and file handlers.

        中文说明: 文件日志可通过 file_path 置空关闭。
        English: File logging can be disabled by setting file_path to null.
        """
        logger.remove()

        logger.add(sys.stdout, format=self.config.log_format, level=self.config.level, enqueue=True)

        if self.config.file_path:
            logger.add(
                self.config.file_path,
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                level=self.config.level,
                format=self.config.log_format,
                enqueue=True,
            )

        logger.configure(extra={"name": self.config.name})
        return logger

    def __getattr__(self, item):
        """Delegate unknown attributes to the underlying logger.

        中文说明: 让调用方可以直接使用 self.logger.info 等 Loguru 方法。
        English: Allows callers to use Loguru methods such as self.logger.info.
        """
        return getattr(self._logger, item)

    @property
    def logger(self):
        """Return the underlying Loguru logger.

        中文说明: 供需要原始 logger 的调用方使用。
        English: Exposed for callers that need the raw logger.
        """
        return self._logger


def get_logger(
    name: str = "AFA",
    level: str = "INFO",
    file_path: str = "./logs/afa.log",
    format: str = "detailed",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
) -> LoggerManager:
    """Create a LoggerManager.

    中文说明: 兼容旧代码的便捷函数。
    English: Convenience helper kept for compatibility with old code.
    """
    config = {
        "name": name,
        "level": level,
        "file_path": file_path,
        "format": format,
        "rotation": rotation,
        "retention": retention,
        "compression": compression,
    }
    return LoggerManager(config)
