"""Tool schemas for AgentFoX.

中文说明: 最小版只需要从 LangGraph state 注入 image_path 的基础 schema。
English: The minimal release only needs the base schema that injects image_path
from LangGraph state.
"""

from .base_schema import BaseSchema

__all__ = ["BaseSchema"]
