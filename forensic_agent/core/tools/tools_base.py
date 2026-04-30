"""Base classes for AgentFoX tools.

中文说明: 所有工具都实现 name/description/execute, 由 ForensicTools 自动适配到 LangChain。
English: Every tool implements name/description/execute and is adapted to
LangChain by ForensicTools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger

from .schema import BaseSchema


def skip_auto_register(cls):
    """Mark a tool class as excluded from discovery.

    中文说明: 当前最小版很少需要, 但保留给测试或未来扩展使用。
    English: Rarely needed in the minimal release, but kept for tests and future
    extensions.
    """
    cls._skip_auto_register = True
    return cls


class ToolsBase(ABC):
    """Abstract base class for forensic tools.

    中文说明: args_schema 默认从 Agent state 注入 image_path。
    English: args_schema injects image_path from agent state by default.
    """

    _skip_auto_register = False
    args_schema = BaseSchema

    def __init__(self, config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        self.config = config or {}
        self.logger = logger

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name.

        中文说明: 该名称会出现在 Agent 可调用工具列表里。
        English: This name appears in the agent's callable tool list.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description shown to the LLM.

        中文说明: 描述应明确工具用途和返回内容。
        English: The description should state the tool purpose and return shape.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args, **kwargs: Any):
        """Run the tool.

        中文说明: 子类负责参数验证、核心逻辑和错误处理。
        English: Subclasses handle validation, core logic, and error handling.
        """
        raise NotImplementedError
