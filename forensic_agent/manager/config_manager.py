"""
配置管理器 - 统一的配置管理和验证
提供类型安全的配置访问和验证功能
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .base_manager import BaseManager
from ..core.core_exceptions import ConfigurationError


class ConfigManager(BaseManager):
    """
    配置管理器实现
    提供类型安全的配置访问
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._cached_values: Dict[str, Any] = {}  # 缓存频繁访问的值
        self._load_config()

    def _load_config(self) -> None:
        """加载并验证配置文件"""
        if not self.config_path.exists():
            raise ConfigurationError(f"配置文件不存在: {self.config_path}", "CONFIG_FILE_NOT_FOUND")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"配置文件格式错误: {str(e)}", "CONFIG_PARSE_ERROR", e)
        except Exception as e:
            raise ConfigurationError(f"配置加载失败: {str(e)}", "CONFIG_LOAD_ERROR", e)

        self._cached_values.clear()  # 清空缓存

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键和缓存"""
        # 检查缓存
        if key in self._cached_values:
            return self._cached_values[key]

        value = self._get_nested_value(key, default)

        # 缓存常用配置项
        if key.startswith(("system.", "llm.", "vlm.", "models.", "knowledge_base.", "agent.")):
            self._cached_values[key] = value

        return value

    def _get_nested_value(self, key: str, default: Any) -> Any:
        """获取嵌套配置值的内部实现"""
        try:
            keys = key.split(".")
            value = self._config

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """设置配置值并清除相关缓存"""
        keys = key.split(".")
        config = self._config

        # 创建嵌套字典结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

        # 清除相关缓存
        self._clear_cache_for_key(key)

    def _clear_cache_for_key(self, key: str) -> None:
        """清除指定键的缓存"""
        # 清除精确匹配的缓存
        self._cached_values.pop(key, None)

        # 清除以该键为前缀的缓存
        keys_to_remove = [k for k in self._cached_values.keys() if k.startswith(f"{key}.")]
        for k in keys_to_remove:
            self._cached_values.pop(k, None)

    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            self._validate_config()
            return True
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"配置验证失败: {str(e)}", "VALIDATION_ERROR", e)

    def reload(self) -> None:
        """重新加载配置"""
        self._load_config()

    def get_section(self, section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """获取配置段"""
        return self.get(section, default or {})

    def save(self, output_path: Optional[str] = None) -> None:
        """保存配置到文件"""
        output_file = Path(output_path) if output_path else self.config_path

        try:
            # 确保目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=True)
        except Exception as e:
            raise ConfigurationError(f"配置保存失败: {str(e)}", "CONFIG_SAVE_ERROR", e)

    def clear_cache(self) -> None:
        """清空所有缓存"""
        self._cached_values.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {"cached_items": len(self._cached_values), "cache_keys": list(self._cached_values.keys())}
