from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import gc
import psutil
import pandas as pd
from loguru import logger


class MemoryManager:
    """内存管理器类"""

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """
        获取当前内存使用信息

        Returns:
            Dict: 包含内存信息的字典
        """
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
        }

    @staticmethod
    def log_memory_usage(prefix: str = ""):
        """记录当前内存使用情况"""
        info = MemoryManager.get_memory_info()
        logger.debug(
            f"{prefix}内存使用: {info['used_gb']:.2f}/{info['total_gb']:.2f} GB ({info['percent']:.1f}%), 可用: {info['available_gb']:.2f} GB"
        )

    @staticmethod
    def force_gc():
        """强制垃圾回收"""
        collected = gc.collect()
        logger.debug(f"垃圾回收清理了 {collected} 个对象")
        return collected

    @staticmethod
    def check_memory_pressure(threshold_gb: float = 1.0) -> bool:
        """
        检查内存压力

        Args:
            threshold_gb: 内存阈值（GB），低于此值认为内存紧张

        Returns:
            bool: True表示内存紧张
        """
        info = MemoryManager.get_memory_info()
        is_pressure = info["available_gb"] < threshold_gb
        if is_pressure:
            logger.warning(f"内存紧张: 可用内存仅 {info['available_gb']:.2f} GB")
        return is_pressure

    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame, string_threshold: float = 0.5) -> pd.DataFrame:
        """
        优化DataFrame内存使用

        Args:
            df: 要优化的DataFrame
            string_threshold: 字符串列转换为category的阈值（唯一值比例）

        Returns:
            优化后的DataFrame
        """
        if df.empty:
            return df

        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # 优化字符串列
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].dtype == "object":
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < string_threshold:
                    df[col] = df[col].astype("category")

        # 优化数值列
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(
            f"DataFrame内存优化: {original_memory:.2f} MB -> {optimized_memory:.2f} MB (节省 {(1-optimized_memory/original_memory)*100:.1f}%)"
        )

        return df
