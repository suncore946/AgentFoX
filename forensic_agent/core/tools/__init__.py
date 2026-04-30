"""Tool package for the minimal AgentFoX runtime.

中文说明: ForensicTools 只会发现配置启用的工具模块。
English: ForensicTools discovers only tool modules enabled by configuration.
"""

from .tools_base import ToolsBase, skip_auto_register

__all__ = ["ToolsBase", "skip_auto_register"]
