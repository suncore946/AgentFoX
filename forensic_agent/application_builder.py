"""
统一的应用程序构建器 - 替代多个分散的构建器
使用现代化的依赖注入和工厂模式
合并了ApplicationBuilder和AFABuilder的功能
"""

from __future__ import annotations

from typing import Dict, Any, Type
from pathlib import Path


from .core.forensic_llm import ForensicLLM
from .core.forensic_agent import ForensicAgent
from .core.forensic_tools import ForensicTools
from .manager.datasets_manager import DatasetsManager
from .manager.logger_manager import LoggerManager
from .manager.config_manager import ConfigManager
from .manager.profile_manager import ProfileManager
from .manager.feature_manager import FeatureManager
from .manager.image_manager import ImageManager
from .service_container import ServiceContainer


class ApplicationBuilder:
    def __init__(self, config_path: str | Path = None, is_debug: bool = False):
        """
        初始化构建器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = self._resolve_config_path(config_path)
        self.container = ServiceContainer()
        self._overrides: Dict[Type, Any] = {}
        self._configured = False
        self.forensic_pipeline = None
        self.is_debug = is_debug

    def _resolve_config_path(self, config_path: str | Path = None) -> Path:
        """解析配置文件路径"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "config.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        return config_path

    def override_service(self, service_type: Type, implementation: Any) -> "ApplicationBuilder":
        """
        覆盖服务实现

        Args:
            service_type: 服务类型
            implementation: 实现实例

        Returns:
            构建器实例，支持链式调用
        """
        self._overrides[service_type] = implementation
        return self

    def build(self) -> ServiceContainer:
        """
        构建完整的应用程序容器

        Returns:
            配置好的服务容器
        """
        if not self._configured:
            self._configure_services()

        return self.container

    def _configure_services(self) -> None:
        """统一配置所有服务 - 按依赖顺序一次性配置"""
        # 1. 核心服务配置 - 配置管理器和日志
        config_manager: ConfigManager = self._get_or_create_service(
            ConfigManager,
            lambda: ConfigManager(str(self.config_path)),
        )
        logger_config = config_manager.get_section("logging")
        logger: LoggerManager = self._get_or_create_service(
            LoggerManager,
            lambda: LoggerManager(logger_config, is_debug=self.is_debug),
        )

        # 2. 管理器服务配置
        # 图像管理器
        image_config = config_manager.get_section("image_manager")
        image_manager: ImageManager = self._get_or_create_service(
            ImageManager,
            lambda: ImageManager(image_config),
        )

        # 数据集管理器
        datasets_config = config_manager.get_section("datasets")
        datasets_manager: DatasetsManager = self._get_or_create_service(
            DatasetsManager,
            lambda: DatasetsManager(datasets_config),
        )

        # 模型画像管理器
        model_profile_config = config_manager.get_section("profiles") | config_manager.get_section("agent", {})
        model_profile_manager = self._get_or_create_service(
            ProfileManager,
            lambda: ProfileManager(model_profile_config),
        )
        logger.info("管理器服务配置完成")

        # Agent和工作流服务配置
        # LLM实例
        forensic_llm: ForensicLLM = self._get_or_create_service(
            ForensicLLM,
            lambda: ForensicLLM(config_manager.get_section("llm", {})),
        )

        # 特征管理器
        feature_config = config_manager.get_section("feature_extraction")
        feature_manager: FeatureManager = self._get_or_create_service(
            FeatureManager,
            lambda: FeatureManager(feature_config, image_manager, semantic_llm=forensic_llm.llm),
        )

        # 工具管理器
        tools_config = config_manager.get_section("tools", {}) | config_manager.get_section("agent", {})
        forensic_tools: ForensicTools = self._get_or_create_service(
            ForensicTools,
            lambda: ForensicTools(
                tools_config,
                image_manager,
                feature_manager,
                profile_manager=model_profile_manager,
                tools_llm=forensic_llm.llm,
                dataset_manager=datasets_manager,
            ),
        )
        forensic_tools.auto_discover_and_register()

        # ForensicAgent
        forensic_agent = self._get_or_create_service(
            ForensicAgent,
            lambda: ForensicAgent(
                config=config_manager.get_section("agent", {}),
                forensic_tool=forensic_tools,
                forensic_llm=forensic_llm,
                image_manager=image_manager,
                is_debug=self.is_debug,
            ),
        )
        logger.info("Agent和ForensicOrchestrator服务配置完成")
        logger.info("所有服务配置完成")

        self._configured = True

    def _get_or_create_service(self, service_type: Type, factory_func) -> Any:
        """获取或创建服务实例"""
        # 检查是否有覆盖实现
        if service_type in self._overrides:
            instance = self._overrides[service_type]
            self.container.register_singleton(service_type, instance)
            return instance

        # 检查容器中是否已经存在
        if self.container.has_service(service_type):
            return self.container.get_service(service_type)

        # 创建新实例
        instance = factory_func()
        self.container.register_singleton(service_type, instance)
        return instance
