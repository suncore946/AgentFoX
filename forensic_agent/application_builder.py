"""Dependency assembly for the minimal AgentFoX runtime.

中文说明: 这里集中创建配置、日志、数据集、LLM、工具和 Agent, 便于开源用户理解运行链路。
English: This module creates configuration, logging, dataset, LLM, tools, and
agent services in one place so open-source users can inspect the runtime flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

from .core.forensic_agent import ForensicAgent
from .core.forensic_llm import ForensicLLM
from .core.forensic_tools import ForensicTools
from .manager.config_manager import ConfigManager
from .manager.datasets_manager import DatasetsManager
from .manager.image_manager import ImageManager
from .manager.logger_manager import LoggerManager
from .manager.profile_manager import ProfileManager
from .service_container import ServiceContainer


class ApplicationBuilder:
    """Build the dependency container.

    中文说明: 该构建器只注册最小推理所需服务, 不包含训练、标注或画像生成流水线。
    English: This builder registers only services needed for minimal inference,
    excluding training, labeling, or profile-generation pipelines.
    """

    def __init__(self, config_path: str | Path, is_debug: bool = False):
        self.config_path = self._resolve_config_path(config_path)
        self.is_debug = is_debug
        self.container = ServiceContainer()
        self._overrides: Dict[Type[Any], Any] = {}
        self._configured = False

    @staticmethod
    def _resolve_config_path(config_path: str | Path) -> Path:
        """Resolve and validate the YAML config path.

        中文说明: 配置必须显式传入, 避免开源包携带私有默认路径。
        English: The config must be explicit so the open-source package never
        relies on private default paths.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    def override_service(self, service_type: Type[Any], implementation: Any) -> "ApplicationBuilder":
        """Override a service for tests.

        中文说明: 单元测试可替换 LLM 或数据服务, 正常运行无需使用。
        English: Tests may replace the LLM or data service; normal inference
        does not need this hook.
        """
        self._overrides[service_type] = implementation
        return self

    def build(self) -> ServiceContainer:
        """Return a fully configured service container.

        中文说明: 第一次调用时按依赖顺序构建服务, 后续调用复用同一个容器。
        English: The first call builds services in dependency order; later calls
        reuse the same container.
        """
        if not self._configured:
            self._configure_services()
        return self.container

    def _configure_services(self) -> None:
        """Create services in dependency order.

        中文说明: 数据和 profile 服务先于工具创建, 因为 semantic tool 会读写运行时缓存。
        English: Dataset and profile services are created before tools because
        the semantic tool reads and writes runtime caches.
        """
        config_manager = self._get_or_create_service(ConfigManager, lambda: ConfigManager(str(self.config_path)))
        logger = self._get_or_create_service(
            LoggerManager,
            lambda: LoggerManager(config_manager.get_section("logging"), is_debug=self.is_debug),
        )
        image_manager = self._get_or_create_service(
            ImageManager,
            lambda: ImageManager(config_manager.get_section("image_manager")),
        )
        datasets_manager = self._get_or_create_service(
            DatasetsManager,
            lambda: DatasetsManager(config_manager.get_section("datasets")),
        )

        profile_config = (
            config_manager.get_section("profiles", {})
            | config_manager.get_section("agent", {})
            | {"datasets": config_manager.get_section("datasets", {})}
        )
        profile_manager = self._get_or_create_service(ProfileManager, lambda: ProfileManager(profile_config))

        forensic_llm = self._get_or_create_service(ForensicLLM, lambda: ForensicLLM(config_manager.get_section("llm", {})))
        default_llm = forensic_llm.get_pos_llm(0)

        tools_config = (
            config_manager.get_section("tools", {})
            | config_manager.get_section("agent", {})
            | {"feature_extraction": config_manager.get_section("feature_extraction", {})}
        )
        forensic_tools = self._get_or_create_service(
            ForensicTools,
            lambda: ForensicTools(
                config=tools_config,
                image_manager=image_manager,
                profile_manager=profile_manager,
                dataset_manager=datasets_manager,
                tools_llm=default_llm,
            ),
        )
        forensic_tools.auto_discover_and_register()

        self._get_or_create_service(
            ForensicAgent,
            lambda: ForensicAgent(
                config=config_manager.get_section("agent", {}),
                forensic_tool=forensic_tools,
                forensic_llm=forensic_llm,
                image_manager=image_manager,
                is_debug=self.is_debug,
            ),
        )
        logger.info("AgentFoX minimal services configured.")
        self._configured = True

    def _get_or_create_service(self, service_type: Type[Any], factory_func) -> Any:
        """Return an existing service or register a new singleton.

        中文说明: 所有运行时服务使用单例, 避免同一批次重复初始化 LLM 和缓存。
        English: Runtime services are singletons to avoid repeated LLM and cache
        initialization in a batch run.
        """
        if service_type in self._overrides:
            instance = self._overrides[service_type]
            self.container.register_singleton(service_type, instance)
            return instance
        if self.container.has_service(service_type):
            return self.container.get_service(service_type)
        instance = factory_func()
        self.container.register_singleton(service_type, instance)
        return instance
