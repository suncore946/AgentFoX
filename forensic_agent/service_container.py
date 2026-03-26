"""
服务容器 - 现代化依赖注入实现
"""

from __future__ import annotations

from typing import Dict, Any, Callable, TypeVar, Type, cast
from .core.core_exceptions import ConfigurationError

T = TypeVar("T")


class ServiceContainer:

    def __init__(self) -> None:
        self._services: Dict[str, Type[Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """注册单例服务"""
        key = self._get_service_key(service_type)
        self._singletons[key] = instance

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """注册工厂方法"""
        key = self._get_service_key(service_type)
        self._factories[key] = factory

    def register_transient(self, service_type: Type[T], implementation: Type[T]) -> None:
        """注册瞬态服务"""
        key = self._get_service_key(service_type)
        self._services[key] = implementation

    def get_service(self, service_type: Type[T]) -> T:
        """获取服务实例，提供类型安全保证"""
        key = self._get_service_key(service_type)

        # 首先检查单例
        if key in self._singletons:
            return cast(T, self._singletons[key])

        # 然后检查工厂方法
        if key in self._factories:
            factory = self._factories[key]
            instance = factory()
            return cast(T, instance)

        # 最后检查瞬态服务
        if key in self._services:
            implementation = self._services[key]
            try:
                instance = implementation()
                return cast(T, instance)
            except Exception as e:
                raise ConfigurationError(f"无法创建服务实例: {service_type.__name__}", config_key=key) from e

        raise ConfigurationError(f"未注册的服务类型: {service_type.__name__}", config_key=key)

    def get_all_services(self) -> Dict[str, Any]:
        """获取所有已注册的服务实例（包括单例、工厂、瞬态实例）"""
        result = {}
        # 单例直接返回
        for key, instance in self._singletons.items():
            result[key] = instance
        # 工厂生成实例
        for key, factory in self._factories.items():
            try:
                result[key] = factory()
            except Exception:
                result[key] = None
        # 瞬态实例
        for key, implementation in self._services.items():
            try:
                result[key] = implementation()
            except Exception:
                result[key] = None
        return result

    def has_service(self, service_type: Type[T]) -> bool:
        """检查是否已注册指定服务类型"""
        key = self._get_service_key(service_type)
        return key in self._singletons or key in self._factories or key in self._services

    def remove_service(self, service_type: Type[T]) -> None:
        """移除已注册的服务"""
        key = self._get_service_key(service_type)

        self._singletons.pop(key, None)
        self._factories.pop(key, None)
        self._services.pop(key, None)

    def clear(self) -> None:
        """清空所有注册的服务"""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()

    def get_registered_services(self) -> Dict[str, str]:
        """获取所有已注册服务的摘要信息"""
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
        """获取服务类型的键名"""
        if hasattr(service_type, "__module__") and hasattr(service_type, "__name__"):
            return f"{service_type.__module__}.{service_type.__name__}"
        return str(service_type)
