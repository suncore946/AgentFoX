"""AgentFoX exception hierarchy.

中文说明: 最小开源版只保留当前推理链路实际使用的异常类型。
English: The minimal open-source release keeps only exception types used by the
current inference path.
"""

from __future__ import annotations

from typing import Optional


class ForensicTemplateError(Exception):
    """Prompt template error.

    中文说明: 模板文件缺失、为空或编码错误时抛出。
    English: Raised when template files are missing, empty, or mis-encoded.
    """


class AFAException(Exception):
    """Base AgentFoX exception.

    中文说明: error_code 便于 CLI 和日志定位错误来源。
    English: error_code helps CLI output and logs identify the error source.
    """

    def __init__(self, message: str, error_code: Optional[str] = None, *, cause: Optional[Exception] = None) -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return a readable error string.

        中文说明: 有 error_code 时把它放在消息前缀。
        English: Prefixes the message with error_code when present.
        """
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AFAException):
    """Configuration error.

    中文说明: 配置文件缺失、字段不合法或服务未注册时使用。
    English: Used for missing config files, invalid fields, or unregistered
    services.
    """

    def __init__(self, message: str, error_code: Optional[str] = None, *, config_key: Optional[str] = None) -> None:
        """Initialize configuration error.

        中文说明: 兼容旧代码传入 error_code 的位置参数, 新代码可用 config_key。
        English: Compatible with old positional error_code calls while new code
        can use config_key.
        """
        resolved_code = f"CONFIG_ERROR_{config_key}" if config_key else (error_code or "CONFIG_ERROR")
        super().__init__(message, resolved_code)
        self.config_key = config_key


class FeatureExtractionError(AFAException):
    """Feature extraction error.

    中文说明: 轻量图像特征或语义 profile 生成失败时使用。
    English: Used when lightweight image features or semantic profile generation
    fails.
    """

    def __init__(self, message: str, *, feature_type: Optional[str] = None) -> None:
        """Initialize feature extraction error.

        中文说明: feature_type 标记 metadata/spatial/frequency/semantic 等阶段。
        English: feature_type marks stages such as metadata, spatial, frequency,
        or semantic.
        """
        error_code = f"FEATURE_ERROR_{feature_type}" if feature_type else "FEATURE_ERROR"
        super().__init__(message, error_code)
        self.feature_type = feature_type
