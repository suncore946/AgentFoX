from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger
from .schema import BaseSchema

def skip_auto_register(cls):
    """
    类装饰器：标记该类在自动发现工具时跳过注册

    Usage:
        @skip_auto_register
        class MyTestTool(ForensicToolBase):
            pass
    """
    cls._skip_auto_register = True
    return cls


class ToolsBase(ABC):
    """取证工具抽象基类"""

    _skip_auto_register = False  # 默认不跳过注册
    args_schema = BaseSchema

    def __init__(self, config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        self.config = config or {}
        self.logger = logger

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（必须唯一）"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，包括参数说明"""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs: Any):
        """执行工具 - 模板方法模式。封装验证、日志和错误处理。

        Args:
            *args: 位置参数（传递给 run 和 validate_input）。
            **kwargs: 命名参数（传递给 run 和 validate_input）。

        Returns:
            Dict[str, Any]: 执行结果，包括 tool, tool_result, execution_time, timestamp 等。
        """
        pass