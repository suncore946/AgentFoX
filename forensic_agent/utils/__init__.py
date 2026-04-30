"""Shared utility helpers.

中文说明: 工具函数不读取私有路径, API key 只从配置或环境变量读取。
English: Utilities do not read private paths; API keys are read only from config
or environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


def get_base_url(llm_config: dict) -> str:
    """Normalize an OpenAI-compatible base URL.

    中文说明: 配置优先, 其次读取 OPENAI_API_BASE, 最后使用官方默认地址。
    English: Config wins first, then OPENAI_API_BASE, then the official default.
    """
    base_url = (llm_config.get("base_url") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1").strip()
    if not base_url:
        raise ValueError("base_url is empty.")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        base_url = "https://" + base_url

    parsed = urlparse(base_url)
    if not parsed.netloc:
        raise ValueError(f"Invalid base_url host: {base_url}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return f"{parsed.scheme}://{parsed.netloc.rstrip('/')}{path}"


def create_chat_llm(
    llm_config: dict,
    callbacks: Optional[Union[List[BaseCallbackHandler], BaseCallbackHandler]] = None,
) -> ChatOpenAI:
    """Create a chat LLM for tools.

    中文说明: 工具层可复用主 LLM, 也可通过 tools_llm 单独配置。
    English: Tools may reuse the main LLM or use a dedicated tools_llm config.
    """
    if not isinstance(llm_config, dict):
        raise TypeError("llm_config must be a dictionary.")
    model_provider = str(llm_config.get("model_provider", "openai")).lower()
    api_key = llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    llm_kwargs = {
        "model": llm_config.get("model", "gpt-4o-mini"),
        "model_provider": model_provider,
        "temperature": llm_config.get("temperature", 0),
        "max_tokens": llm_config.get("max_tokens", 2048),
        "timeout": llm_config.get("timeout", 60),
    }
    if model_provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible providers.")
        llm_kwargs["api_key"] = api_key
        llm_kwargs["base_url"] = get_base_url(llm_config)
    else:
        llm_kwargs["base_url"] = llm_config.get("base_url")

    if callbacks is not None:
        llm_kwargs["callbacks"] = [callbacks] if isinstance(callbacks, BaseCallbackHandler) else callbacks
    return init_chat_model(**llm_kwargs)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed.

    中文说明: 输出目录统一通过该函数创建。
    English: Output directories are created through this helper.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_json_serialize(obj: Any) -> Any:
    """Serialize common numpy objects.

    中文说明: 用于将特征提取中的 numpy 值写入 JSON。
    English: Used to write numpy values from feature extraction into JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    return obj


def safe_division(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Divide safely.

    中文说明: denominator 为 0 时返回默认值。
    English: Returns the default value when denominator is zero.
    """
    return numerator / denominator if denominator > 0 else default


__all__ = [
    "create_chat_llm",
    "ensure_directory",
    "get_base_url",
    "safe_division",
    "safe_json_serialize",
]
