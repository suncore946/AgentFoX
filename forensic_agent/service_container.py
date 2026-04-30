"""Small dependency injection container.

中文说明: AgentFoX 使用该容器注册运行时单例, 避免全局变量和隐式私有配置。
English: AgentFoX uses this container for runtime singletons, avoiding global
state and implicit private config.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, TypeVar, Type, cast
from .core.core_exceptions import ConfigurationError

T = TypeVar("T")


class ServiceContainer:
    """Register and resolve runtime services.

    中文说明: 最小开源版主要使用 register_singleton/get_service。
    English: The minimal open-source runtime mainly uses register_singleton and
    get_service.
    """

    def __init__(self) -> None:
        self._services: Dict[str, Type[Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Register one singleton instance.

        中文说明: LLM、工具和数据管理器都以单例形式复用。
        English: LLMs, tools, and data managers are reused as singletons.
        """
        key = self._get_service_key(service_type)
        self._singletons[key] = instance

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory.

        中文说明: 当前最小流程很少使用, 保留给测试扩展。
        English: Rarely used in the minimal path and kept for test extensions.
        """
        key = self._get_service_key(service_type)
        self._factories[key] = factory

    def register_transient(self, service_type: Type[T], implementation: Type[T]) -> None:
        """Register a transient implementation.

        中文说明: 每次解析都创建新实例。
        English: Creates a new instance on each resolution.
        """
        key = self._get_service_key(service_type)
        self._services[key] = implementation

    def get_service(self, service_type: Type[T]) -> T:
        """Resolve a service instance.

        中文说明: 未注册服务会抛出 ConfigurationError, 便于定位构建链路缺失。
        English: Missing services raise ConfigurationError so build-chain gaps
        are easy to locate.
        """
        key = self._get_service_key(service_type)

        if key in self._singletons:
            return cast(T, self._singletons[key])

        if key in self._factories:
            factory = self._factories[key]
            instance = factory()
            return cast(T, instance)

        if key in self._services:
            implementation = self._services[key]
            try:
                instance = implementation()
                return cast(T, instance)
            except Exception as e:
                raise ConfigurationError(f"无法创建服务实例: {service_type.__name__}", config_key=key) from e

        raise ConfigurationError(f"未注册的服务类型: {service_type.__name__}", config_key=key)

    def get_all_services(self) -> Dict[str, Any]:
        """Return all registered service instances.

        中文说明: 诊断接口, 默认推理不使用。
        English: Diagnostic helper and not used by default inference.
        """
        result = {}
        for key, instance in self._singletons.items():
            result[key] = instance
        for key, factory in self._factories.items():
            try:
                result[key] = factory()
            except Exception:
                result[key] = None
        for key, implementation in self._services.items():
            try:
                result[key] = implementation()
            except Exception:
                result[key] = None
        return result

    def has_service(self, service_type: Type[T]) -> bool:
        """Return whether a service type has been registered.

        中文说明: ApplicationBuilder 用它避免重复构建单例。
        English: ApplicationBuilder uses this to avoid rebuilding singletons.
        """
        key = self._get_service_key(service_type)
        return key in self._singletons or key in self._factories or key in self._services

    def remove_service(self, service_type: Type[T]) -> None:
        """Remove a registered service.

        中文说明: 主要用于测试覆盖。
        English: Mainly useful for tests.
        """
        key = self._get_service_key(service_type)

        self._singletons.pop(key, None)
        self._factories.pop(key, None)
        self._services.pop(key, None)

    def clear(self) -> None:
        """Clear all registrations.

        中文说明: 不会删除磁盘输出或缓存文件。
        English: Does not delete disk outputs or cache files.
        """
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()

    def get_registered_services(self) -> Dict[str, str]:
        """Return a registration summary.

        中文说明: 只返回服务名和生命周期, 不暴露配置内容。
        English: Returns service names and lifetimes only, not config content.
        """
        result = {}

        for key in self._singletons:
            result[key] = "singleton"

        for key in self._factories:
            result[key] = "factory"

        for key in self._services:
            result[key] = "transient"

        return result

    @staticmethod
    def _get_service_key(service_type: Type[Any]) -> str:
        """Build a stable key for a service type.

        中文说明: 使用 module + class name 防止同名类冲突。
        English: Uses module + class name to avoid collisions between same-name
        classes.
        """
        if hasattr(service_type, "__module__") and hasattr(service_type, "__name__"):
            return f"{service_type.__module__}.{service_type.__name__}"
        return str(service_type)
