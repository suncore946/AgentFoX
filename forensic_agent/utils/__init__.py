from .logger import get_logger
from .progress_logger import ProgressLogger


import os
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from urllib.parse import urlparse
from typing import Optional, List, Union
from langchain_core.callbacks import BaseCallbackHandler
import torch
import pynvml
from loguru import logger

from pathlib import Path
from typing import Any
import numpy as np
from loguru import logger


def get_base_url(llm_config: dict) -> str:
    """从配置或环境变量中获取并规范化 base_url

    Args:
        llm_config: LLM配置字典

    Returns:
        str: 规范化后的 base_url

    Raises:
        ValueError: 当 base_url 配置错误或格式不正确时
    """
    # 优先从配置再回退到环境变量（常见命名 OPENAI_API_BASE）
    base_url = llm_config.get("base_url") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
    base_url = base_url.strip()

    if not base_url:
        raise ValueError("base_url 为空")

    # 如果没有 scheme，则默认加 https://
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        base_url = "https://" + base_url

    try:
        parsed = urlparse(base_url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc
        path = parsed.path or ""

        # urlparse 在某些输入下可能把 host 放到 path（但我们已确保带 scheme），再保险检查
        if not netloc:
            # 尝试从 path 中取 host
            parts = path.lstrip("/").split("/", 1)
            netloc = parts[0]
            path = "/" + parts[1] if len(parts) > 1 else ""

        netloc = netloc.rstrip("/")

        # 简单合法性校验：必须有主机名或允许 localhost / ip
        if not netloc or (
            ("." not in netloc and not netloc.startswith("localhost") and not netloc.replace(":", "").replace(".", "").isdigit())
        ):
            raise ValueError(f"无效的 base_url 主机: {netloc}")

        # 清理并保证以 /v1 结尾（避免重复 /v1/v1）
        path = (path.rstrip("/") if path else "").rstrip()
        if not path.endswith("/v1"):
            path = path + "/v1"
        if not path.startswith("/"):
            path = "/" + path

        normalized = f"{scheme}://{netloc}{path}"
        return normalized
    except Exception as e:
        raw_base = llm_config.get("base_url") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
        raise ValueError(f"base_url 配置错误: {raw_base.strip()} -> {e}") from e


def create_chat_llm(llm_config: dict, callbacks: Optional[Union[List[BaseCallbackHandler], BaseCallbackHandler]] = None) -> ChatOpenAI:
    """创建默认的LLM实例（改进版：更稳健的 base_url 规范化与校验，支持 callbacks）

    Args:
        llm_config: LLM配置字典
        callbacks: 回调函数列表或单个回调函数

    Returns:
        ChatOpenAI: 配置好的LLM实例
    """
    model_provider = llm_config.get("model_provider", "openai").lower()

    api_key = llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if model_provider == "openai":
        if not api_key:
            raise ValueError("未检测到OpenAI API Key")
        # 获取并规范化 base_url
        base_url = get_base_url(llm_config)
    else:
        base_url = llm_config.get("base_url")  # 非 OpenAI 模型提供商，直接使用配置的 base_url

    # 处理 callbacks 参数
    processed_callbacks = None
    if callbacks is not None:
        if isinstance(callbacks, BaseCallbackHandler):
            # 单个回调函数转换为列表
            processed_callbacks = [callbacks]
        elif isinstance(callbacks, list):
            # 验证列表中的每个元素都是 BaseCallbackHandler 实例
            for cb in callbacks:
                if not isinstance(cb, BaseCallbackHandler):
                    raise ValueError(f"callbacks 列表中包含非 BaseCallbackHandler 类型: {type(cb)}")
            processed_callbacks = callbacks
        else:
            raise ValueError(f"callbacks 必须是 BaseCallbackHandler 或其列表，got: {type(callbacks)}")

    # 创建 ChatOpenAI 实例的参数
    llm_kwargs = {
        "model": llm_config.get("model", "gpt-4o"),
        "model_provider": model_provider,
        "temperature": llm_config.get("temperature", 0),
        "max_tokens": llm_config.get("max_tokens", 2048),
        "base_url": base_url,
        "timeout": llm_config.get("timeout", 60),
        "api_key": api_key,
    }

    # 如果有 callbacks，添加到参数中
    if processed_callbacks:
        llm_kwargs["callbacks"] = processed_callbacks

    llm = init_chat_model(**llm_kwargs)
    return llm


def get_available_gpus(min_memory_gb=10, max_utilization=80):
    """获取满足条件的可用GPU列表

    Args:
        min_memory_gb (int): 最小可用显存（单位：GB）
        max_utilization (int): 最大占用率（百分比）

    Returns:
        List[int]: 满足条件的 GPU 索引列表
    """
    if not torch.cuda.is_available():
        logger.info("未检测到可用的 CUDA 环境")
        return []

    # 参数合法性检查
    if min_memory_gb <= 0 or max_utilization < 0 or max_utilization > 100:
        raise ValueError("参数 min_memory_gb 和 max_utilization 必须为正数，且 max_utilization 在 0-100 范围内")

    # 初始化 NVIDIA 管理库
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        logger.error(f"无法初始化 NVIDIA 管理库: {e}")
        return []

    available_gpus = []

    try:
        for i in range(torch.cuda.device_count()):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # 转换为 GB 单位
                free_memory_gb = mem_info.free / (1024**3)
                gpu_utilization = util_info.gpu

                # 检查显存和占用率条件
                if free_memory_gb >= min_memory_gb and gpu_utilization <= max_utilization:
                    available_gpus.append(i)
                else:
                    logger.info(f"GPU {i} 不满足条件: 可用显存 {free_memory_gb:.2f} GB, 占用率 {gpu_utilization}%")
            except pynvml.NVMLError as e:
                logger.warning(f"无法获取 GPU {i} 信息: {e}")
            except Exception as e:
                logger.warning(f"GPU {i} 检测时发生未知错误: {e}")
    finally:
        # 释放 NVIDIA 管理库
        pynvml.nvmlShutdown()

    if not available_gpus:
        logger.info("所有 GPU 均不满足条件")
    return available_gpus


def ensure_directory(path: str) -> Path:
    """确保目录存在"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_json_serialize(obj: Any) -> Any:
    """安全的JSON序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj


def safe_division(numerator: float, denominator: float, default: float = np.nan) -> float:
    """安全除法，避免除零错误"""
    return numerator / denominator if denominator > 0 else default
