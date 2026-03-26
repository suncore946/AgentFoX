"""
面向对象的工具系统 - 重构版本
实现清晰的职责分离和自动工具发现
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from loguru import logger
import pandas as pd
from pydantic import BaseModel

from langchain_core.tools import BaseTool

from .tools import ToolsBase
from ..utils import create_chat_llm
from ..manager.datasets_manager import DatasetsManager

"""
@tool
def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
    '''Do something with state.'''
    if len(state["messages"]) > 2:
        return state["foo"] + str(x)
    else:
        return "not enough messages"

@tool
def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
    '''Do something else with state.'''
    return foo + str(x + 1)
"""


class ToolsAdapter(BaseTool):
    """
    LangChain工具适配器
    """

    def __init__(self, forensic_tool: ToolsBase, args_schema: Optional[Type[BaseModel]]):
        super().__init__(name=forensic_tool.name, description=forensic_tool.description, args_schema=args_schema)
        self._forensic_tool = forensic_tool

    @property
    def tool(self) -> ToolsBase:
        """获取底层取证工具实例"""
        return self._forensic_tool

    def _run(self, **kwargs: Any) -> str:
        """运行工具（同步）"""
        logger.debug(f"执行工具 {self.name}，参数: {kwargs}")
        return self._forensic_tool.execute(**kwargs)


class ForensicTools:
    """工具管理器实现 - 支持自动发现和注册"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        image_manager,
        feature_manager,
        profile_manager,
        dataset_manager: DatasetsManager,
        tools_llm=None,
    ):
        self.config = config or {}
        self._tools: Dict[str, ToolsBase] = {}
        self._langchain_adapters: Dict[str, ToolsAdapter] = {}
        self.image_manager = image_manager
        self.feature_manager = feature_manager
        self.profile_manager = profile_manager
        self.datasets_manager = dataset_manager

        if tools_llm_config := self.config.get("tools_llm", None):
            self.tools_llm = create_chat_llm(tools_llm_config)
        else:
            self.tools_llm = tools_llm

        toggle_to_tool = {
            "open_calibration": ["calibration"],
            "open_clustering": ["clustering_profiles"],
            "open_semantic": ["semantic_analysis"],
            "open_expert": ["expert_analysis"],
        }
        ignored_tools = []
        for config_key, tool_names in toggle_to_tool.items():
            if not self.config.get(config_key, False):
                if isinstance(tool_names, str):
                    tool_names = [tool_names]
                ignored_tools.extend(tool_names)
        self.ignore_tools_name = [name.lower() for name in ignored_tools]

    def auto_discover_and_register(self) -> None:
        """自动发现并注册所有继承ForensicToolBase的工具"""
        from . import tools as tools_package

        tools_path = Path(tools_package.__file__).parent
        logger.info(f"开始自动发现工具，扫描路径: {tools_path}")

        # 扫描 tools 包中的所有模块
        for importer, mod_name, is_pkg in pkgutil.walk_packages(path=[str(tools_path)], prefix=f"{tools_package.__name__}."):
            # 跳过 __pycache__ 和其他非 Python 文件
            if is_pkg or mod_name.endswith(("__init__", "base", "tools_adapter")):
                continue

            # 检查是否需要忽略该模块（忽略规则不区分大小写）
            if any(ignore_name in mod_name.lower() for ignore_name in self.ignore_tools_name):
                logger.warning(f"跳过自动发现工具 (忽略关键字匹配): {mod_name}")
                continue

            # 动态导入模块
            module = importlib.import_module(mod_name)

            # 检查模块中的所有类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # 检查是否继承 ForensicToolBase 且不是基类本身
                if issubclass(obj, ToolsBase) and obj is not ToolsBase and obj.__module__ == mod_name:

                    # 检查是否有跳过自动注册的装饰器标记
                    if getattr(obj, "_skip_auto_register", False):
                        logger.info(f"跳过自动注册工具 (装饰器标记): {obj.__name__}")
                        continue

                    # 获取工具配置
                    tool_config = self._get_tool_config(obj)

                    # 实例化并注册工具
                    tool_instance = obj(
                        **{
                            "config": tool_config,
                            "image_manager": self.image_manager,
                            "feature_manager": self.feature_manager,
                            "profile_manager": self.profile_manager,
                            "datasets_manager": self.datasets_manager,
                            "tools_llm": self.tools_llm,
                        }
                    )
                    self._register_tool(tool_instance)

                    logger.info(f"自动发现并注册工具: {obj.__name__} -> {tool_instance.name}")

        logger.info(f"工具自动发现完成，共注册 {len(self._tools)} 个工具")
        logger.info(f"已注册工具列表: {list(self._tools.keys())}")

    def _get_tool_config(self, tool_class) -> Dict[str, Any]:
        """获取工具配置"""
        # 根据工具类名获取配置键名
        # 获取config中所有小写开头的配置项
        base_config = {k: v for k, v in self.config.items() if k[0].islower()}
        tool_name = tool_class.__name__.replace("Tool", "")
        return base_config | self.config.get(tool_name, {})

    def _register_tool(self, tool: ToolsBase) -> None:
        """注册工具"""
        if not isinstance(tool, ToolsBase):
            raise TypeError(f"工具必须是ToolBase的子类: {tool}")

        self._tools[tool.name] = tool

        # 获取工具的args_schema（如果定义了，用于结构化参数输入）
        args_schema: Optional[Type[BaseModel]] = getattr(tool, "args_schema", None)
        if args_schema:
            logger.debug(f"工具 {tool.name} 注入了args_schema: {args_schema.__name__}")
        else:
            logger.debug(f"工具 {tool.name} 未定义args_schema，使用默认（非结构化输入）")

        # 创建LangChain适配器，并注入args_schema
        self._langchain_adapters[tool.name] = ToolsAdapter(tool, args_schema=args_schema)

        logger.debug(f"注册工具: {tool.name}")

    def get_all_tools(self) -> List[ToolsAdapter]:
        """获取LangChain兼容工具"""
        return list(self._langchain_adapters.values())

    def get_specific_tool(self, tool_name: str) -> Optional[ToolsAdapter]:
        """获取指定名称的工具"""
        return self._langchain_adapters.get(tool_name, None)
