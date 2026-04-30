"""Tool registry for the minimal AgentFoX runtime.

中文说明: 开源最小版默认只注册 semantic_analysis, 不自动导入 expert/calibration/clustering 工具。
English: The minimal open-source runtime registers semantic_analysis by default
and does not import expert, calibration, or clustering tools.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .tools import ToolsBase
from ..manager.datasets_manager import DatasetsManager
from ..utils import create_chat_llm


class ToolsAdapter(BaseTool):
    """Adapt an AgentFoX tool to LangChain.

    中文说明: LangGraph ReAct 节点只识别 LangChain Tool, 因此这里做轻量适配。
    English: LangGraph ReAct nodes consume LangChain tools, so this class
    performs a small adapter step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _forensic_tool: ToolsBase = PrivateAttr()

    def __init__(self, forensic_tool: ToolsBase, args_schema: Optional[Type[BaseModel]]):
        super().__init__(name=forensic_tool.name, description=forensic_tool.description, args_schema=args_schema)
        self._forensic_tool = forensic_tool

    @property
    def tool(self) -> ToolsBase:
        """Return the underlying AgentFoX tool.

        中文说明: 主要用于内部测试和调试。
        English: Mainly used by internal tests and debugging.
        """
        return self._forensic_tool

    def _run(self, **kwargs: Any) -> Any:
        """Run the wrapped tool synchronously.

        中文说明: 当前最小版只实现同步工具调用。
        English: The current minimal runtime only implements synchronous tool
        execution.
        """
        logger.debug(f"Running tool {self.name} with args: {kwargs}")
        return self._forensic_tool.execute(**kwargs)


class ForensicTools:
    """Discover and register enabled tools.

    中文说明: 工具发现会先检查开关, 关闭的工具不会导入, 避免缺少非核心模块时报错。
    English: Discovery checks feature toggles before import, so disabled tools
    are never imported and cannot fail due to removed optional modules.
    """

    MODULE_BY_TOOL = {
        "semantic_analysis": "semantic_analysis_tool",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        image_manager,
        profile_manager,
        dataset_manager: DatasetsManager,
        tools_llm=None,
    ):
        self.config = config or {}
        self.image_manager = image_manager
        self.profile_manager = profile_manager
        self.datasets_manager = dataset_manager
        self._tools: Dict[str, ToolsBase] = {}
        self._langchain_adapters: Dict[str, ToolsAdapter] = {}
        self._tool_classes: Dict[str, Type[ToolsBase]] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}

        llm_config = self.config.get("tools_llm")
        self.tools_llm = create_chat_llm(llm_config) if llm_config else tools_llm

    def _enabled_tool_modules(self) -> set[str]:
        """Return module names allowed by config toggles.

        中文说明: 只有 open_semantic 为 true 时才注册 semantic_analysis。
        English: semantic_analysis is registered only when open_semantic is true.
        """
        enabled = set()
        if self.config.get("open_semantic", True):
            enabled.add(self.MODULE_BY_TOOL["semantic_analysis"])
        return enabled

    def auto_discover_and_register(self) -> None:
        """Discover enabled tool classes from forensic_agent.core.tools.

        中文说明: 发现过程限定在启用模块集合内, 不扫描已删除的实验工具。
        English: Discovery is limited to enabled modules and does not scan
        removed experimental tools.
        """
        from . import tools as tools_package

        enabled_modules = self._enabled_tool_modules()
        tools_path = Path(tools_package.__file__).parent
        for _, mod_name, is_pkg in pkgutil.walk_packages(path=[str(tools_path)], prefix=f"{tools_package.__name__}."):
            short_name = mod_name.rsplit(".", 1)[-1]
            if is_pkg or short_name not in enabled_modules:
                continue
            module = importlib.import_module(mod_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, ToolsBase) and obj is not ToolsBase and obj.__module__ == mod_name:
                    tool_config = self._get_tool_config(obj)
                    self._tool_classes[obj.__name__] = obj
                    self._tool_configs[obj.__name__] = tool_config
                    self._register_tool(self._instantiate_tool(obj, tool_config))
        logger.info(f"Registered tools: {list(self._tools.keys())}")

    def _get_tool_config(self, tool_class: Type[ToolsBase]) -> Dict[str, Any]:
        """Build the config passed to one tool class.

        中文说明: 小写配置作为公共运行时配置, 类名配置作为工具专属配置。
        English: Lowercase config keys are shared runtime config; class-name
        sections are tool-specific config.
        """
        base_config = {key: value for key, value in self.config.items() if key and key[0].islower()}
        tool_name = tool_class.__name__.replace("Tool", "")
        return base_config | self.config.get(tool_name, {})

    def _instantiate_tool(self, tool_class: Type[ToolsBase], tool_config: Dict[str, Any], tools_llm=None) -> ToolsBase:
        """Create a tool instance.

        中文说明: 每个 workflow 可拿到独立工具实例, 避免并发状态互相污染。
        English: Each workflow can receive an independent tool instance to avoid
        shared mutable state across concurrent runs.
        """
        return tool_class(
            config=dict(tool_config),
            image_manager=self.image_manager,
            profile_manager=self.profile_manager,
            datasets_manager=self.datasets_manager,
            tools_llm=self.tools_llm if tools_llm is None else tools_llm,
        )

    def _register_tool(self, tool: ToolsBase) -> None:
        """Register a tool and its LangChain adapter.

        中文说明: args_schema 来自工具类, 用于从 LangGraph state 注入 image_path。
        English: args_schema comes from the tool class and injects image_path
        from LangGraph state.
        """
        args_schema: Optional[Type[BaseModel]] = getattr(tool, "args_schema", None)
        self._tools[tool.name] = tool
        self._langchain_adapters[tool.name] = ToolsAdapter(tool, args_schema=args_schema)

    def get_all_tools(self) -> List[ToolsAdapter]:
        """Return adapters for the default workflow.

        中文说明: ForensicAgent 构图时调用该接口。
        English: ForensicAgent calls this while building its graph.
        """
        return list(self._langchain_adapters.values())

    def get_tools_for_llm(self, tools_llm) -> List[ToolsAdapter]:
        """Return fresh adapters for one workflow LLM.

        中文说明: 多 LLM 配置时每个 workflow 使用自己的 tool LLM。
        English: With multiple LLMs, each workflow uses its own tool LLM.
        """
        adapters = []
        for class_name, tool_class in self._tool_classes.items():
            tool = self._instantiate_tool(tool_class, self._tool_configs[class_name], tools_llm=tools_llm)
            adapters.append(ToolsAdapter(tool, args_schema=getattr(tool, "args_schema", None)))
        return adapters

    def get_specific_tool(self, tool_name: str) -> Optional[ToolsAdapter]:
        """Return one registered tool by name.

        中文说明: 保留该接口供未来高级功能复用。
        English: This interface is kept for future advanced features.
        """
        return self._langchain_adapters.get(tool_name)
