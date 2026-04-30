"""YAML configuration manager.

中文说明: ConfigManager 只读取用户显式传入的 YAML, 不提供私有默认路径。
English: ConfigManager reads only the user-provided YAML file and does not
provide private default paths.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .base_manager import BaseManager
from ..core.core_exceptions import ConfigurationError


class ConfigManager(BaseManager):
    """Typed access wrapper around a YAML dictionary.

    中文说明: 支持点号路径读取, 例如 llm.model 或 datasets.test_paths。
    English: Supports dotted-key reads such as llm.model or datasets.test_paths.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize and load config.

        中文说明: config_path 必须存在, 以避免开源版本误读本地私有配置。
        English: config_path must exist so the open-source runtime never reads a
        private local default by accident.
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._cached_values: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load YAML from disk.

        中文说明: YAML 解析失败会包装为 ConfigurationError, 便于 CLI 显示清晰错误。
        English: YAML parse failures are wrapped as ConfigurationError for
        clear CLI errors.
        """
        if not self.config_path.exists():
            raise ConfigurationError(f"配置文件不存在: {self.config_path}", "CONFIG_FILE_NOT_FOUND")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"配置文件格式错误: {str(e)}", config_key="PARSE") from e
        except Exception as e:
            raise ConfigurationError(f"配置加载失败: {str(e)}", config_key="LOAD") from e

        self._cached_values.clear()

    def get(self, key: str, default: Any = None) -> Any:
        """Return a config value by dotted key.

        中文说明: 常用路径会被缓存, 但 reload/set 会清空相关缓存。
        English: Common keys are cached, and reload/set clears relevant cache.
        """
        if key in self._cached_values:
            return self._cached_values[key]

        value = self._get_nested_value(key, default)

        if key.startswith(("system.", "llm.", "vlm.", "models.", "knowledge_base.", "agent.")):
            self._cached_values[key] = value

        return value

    def _get_nested_value(self, key: str, default: Any) -> Any:
        """Resolve one dotted config key.

        中文说明: 缺失键返回调用方给出的 default。
        English: Missing keys return the caller-provided default.
        """
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
        """Set a config value in memory.

        中文说明: 该方法不自动写入磁盘, 调用 save 才会落盘。
        English: This does not write to disk until save is called.
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

        self._clear_cache_for_key(key)

    def _clear_cache_for_key(self, key: str) -> None:
        """Clear cached values affected by a key.

        中文说明: 同时清除精确键和子键缓存。
        English: Clears both the exact key and child-key cache entries.
        """
        self._cached_values.pop(key, None)

        keys_to_remove = [k for k in self._cached_values.keys() if k.startswith(f"{key}.")]
        for k in keys_to_remove:
            self._cached_values.pop(k, None)

    def validate(self) -> bool:
        """Validate config.

        中文说明: 当前最小版只保留接口, 具体必需项由各 Manager 初始化时验证。
        English: The minimal release keeps this interface; concrete required
        fields are validated by each Manager during initialization.
        """
        try:
            return True
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"配置验证失败: {str(e)}", "VALIDATION_ERROR", e)

    def reload(self) -> None:
        """Reload YAML from disk.

        中文说明: 测试或服务化场景可用, CLI 通常只加载一次。
        English: Useful for tests or services; CLI usually loads once.
        """
        self._load_config()

    def get_section(self, section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a top-level config section.

        中文说明: 不存在时返回空字典或调用方指定默认值。
        English: Returns an empty dict or caller default when missing.
        """
        return self.get(section, default or {})

    def save(self, output_path: Optional[str] = None) -> None:
        """Save config to YAML.

        中文说明: 默认不在最小运行路径中调用, 避免修改用户模板。
        English: Not used in the default minimal path to avoid modifying user
        templates.
        """
        output_file = Path(output_path) if output_path else self.config_path

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=True)
        except Exception as e:
            raise ConfigurationError(f"配置保存失败: {str(e)}", "CONFIG_SAVE_ERROR", e)

    def clear_cache(self) -> None:
        """Clear all cached values.

        中文说明: 修改配置后可手动调用。
        English: Can be called manually after config mutation.
        """
        self._cached_values.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache diagnostics.

        中文说明: 仅用于调试, 不参与推理。
        English: For debugging only and not used by inference.
        """
        return {"cached_items": len(self._cached_values), "cache_keys": list(self._cached_values.keys())}
