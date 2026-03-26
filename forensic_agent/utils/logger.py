import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger
import functools
import time


class LoggerConfig:
    """日志配置管理器"""

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        log_level: str = "INFO",
        console_format: Optional[str] = None,
        file_format: Optional[str] = None,
        max_file_size: str = "10 MB",
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False,
    ):
        """初始化日志配置

        Args:
            log_dir: 日志目录
            log_level: 日志级别
            console_format: 控制台格式
            file_format: 文件格式
            max_file_size: 单个日志文件最大大小
            backup_count: 备份文件数量
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            enable_json: 是否启用JSON格式日志
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level.upper()
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json

        # 默认格式
        if console_format is None:
            self.console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level}</level> | "
                "<cyan>{name}</cyan> | "
                "<cyan>{file}:{line}</cyan> | "
                "<level>{message}</level>"
            )
        else:
            self.console_format = console_format

        if file_format is None:
            self.file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | " "{level} | " "{file}:{line} | " "{name} | " "{message}"
        else:
            self.file_format = file_format

        # JSON格式
        self.json_format = "{time} | {level} | {name} | {file}:{line} |  {message}"

        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup_logger(self) -> None:
        """配置loguru日志器"""
        # 移除默认处理器
        logger.remove()

        # 控制台处理器
        if self.enable_console:
            logger.add(sys.stderr, format=self.console_format, level=self.log_level, colorize=True, backtrace=True, diagnose=True)

        # 文件处理器
        if self.enable_file:
            # 通用日志文件
            logger.add(
                self.log_dir / "app_{time:YYYY-MM-DD}.log",
                format=self.file_format,
                level=self.log_level,
                rotation=self.max_file_size,
                retention=self.backup_count,
                compression="zip",
                backtrace=True,
                diagnose=True,
            )

            # 错误日志文件
            logger.add(
                self.log_dir / "error_{time:YYYY-MM-DD}.log",
                format=self.file_format,
                level="ERROR",
                rotation=self.max_file_size,
                retention=self.backup_count,
                compression="zip",
                backtrace=True,
                diagnose=True,
            )

        # JSON格式日志（用于日志分析）
        if self.enable_json:
            logger.add(
                self.log_dir / "app_{time:YYYY-MM-DD}.json",
                format=self.json_format,
                level=self.log_level,
                rotation=self.max_file_size,
                retention=self.backup_count,
                serialize=True,
                backtrace=True,
                diagnose=True,
            )


# 全局日志配置实例
_logger_config: Optional[LoggerConfig] = None


def setup_logging(
    output_dir: Union[str, Path] = "./",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False,
    **kwargs,
) -> LoggerConfig:
    """设置全局日志配置

    Args:
        log_dir: 日志目录
        log_level: 日志级别
        enable_console: 启用控制台输出
        enable_file: 启用文件输出
        enable_json: 启用JSON日志
        **kwargs: 其他配置参数

    Returns:
        LoggerConfig: 日志配置实例
    """
    global _logger_config
    log_dir = Path(output_dir) / "logs"

    _logger_config = LoggerConfig(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_json=enable_json,
        **kwargs,
    )

    _logger_config.setup_logger()

    logger.info(f"日志系统已初始化 - 级别: {log_level}, 目录: {log_dir}")
    return _logger_config


def get_logger(name: str = None):
    """获取日志器实例

    Args:
        name: 日志器名称

    Returns:
        logger: loguru日志器实例
    """
    if _logger_config is None:
        # 使用默认配置
        setup_logging()
    if name:
        return logger.bind(name=name)
    return logger


def log_execution_time(func=None, *, level: str = "INFO"):
    """装饰器：记录函数执行时间

    Args:
        func: 被装饰的函数
        level: 日志级别

    Returns:
        装饰后的函数
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = get_logger(f.__module__)
            try:
                func_logger.debug(level, f"开始执行 {f.__name__}")
                result = f(*args, **kwargs)
                execution_time = time.time() - start_time
                func_logger.debug(level, f"完成执行 {f.__name__} - 耗时: {execution_time:.3f}秒")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                func_logger.error(f"执行失败 {f.__name__} - 耗时: {execution_time:.3f}秒 - 错误: {e}")
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


class LogContext:
    """日志上下文管理器"""

    def __init__(self, name: str, level: str = "INFO", log_start: bool = True, log_end: bool = True, log_duration: bool = True):
        """初始化日志上下文

        Args:
            name: 上下文名称
            level: 日志级别
            log_start: 是否记录开始
            log_end: 是否记录结束
            log_duration: 是否记录持续时间
        """
        self.name = name
        self.level = level
        self.log_start = log_start
        self.log_end = log_end
        self.log_duration = log_duration
        self.start_time = None
        self.logger = get_logger()

    def __enter__(self):
        self.start_time = time.time()
        if self.log_start:
            self.logger.debug(self.level, f"开始: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.logger.error(f"异常结束: {self.name} - {exc_val} (耗时: {duration:.3f}秒)")
        else:
            if self.log_end:
                if self.log_duration:
                    self.logger.debug(self.level, f"完成: {self.name} (耗时: {duration:.3f}秒)")
                else:
                    self.logger.debug(self.level, f"完成: {self.name}")
