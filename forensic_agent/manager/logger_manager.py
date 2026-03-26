from loguru import logger
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys
from .base_manager import BaseManager


@dataclass
class LoggerConfig:
    """日志配置类"""

    name: str = "AFA"
    level: str = "INFO"
    file_path: Optional[str] = "./logs/afa.log"
    format: str = "detailed"  # 支持 detailed 和 simple 两种格式
    rotation: str = "10 MB"
    retention: str = "7 days"
    compression: str = "zip"

    def __post_init__(self):
        self.level = self.level.upper()
        if self.file_path:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def detailed_format(self):
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    @property
    def simple_format(self):
        return "<level>{level}</level> | {time:YYYY-MM-DD HH:mm:ss} | {name} | {message}"

    @property
    def log_format(self):
        return self.detailed_format if self.format == "detailed" else self.simple_format


class LoggerManager(BaseManager):
    """日志管理器 - 直接使用loguru"""

    def __init__(self, config: dict, is_debug: bool = False):
        if is_debug:
            config["level"] = "DEBUG"
        else:
            config["level"] = config.get("level", "INFO").upper()
        self.config = LoggerConfig(**config)
        self._logger = self._setup_logger()

    def _setup_logger(self):
        # 移除默认handler
        logger.remove()

        # 添加控制台handler
        logger.add(sys.stdout, format=self.config.log_format, level=self.config.level, enqueue=True)

        # 添加文件handler
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

        # 绑定名称上下文
        logger.configure(extra={"name": self.config.name})
        return logger

    def __getattr__(self, item):
        return getattr(self._logger, item)

    @property
    def logger(self):
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
    """获取日志记录器的便捷函数"""
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
