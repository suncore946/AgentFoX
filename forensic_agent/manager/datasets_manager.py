"""Minimal CSV dataset loader for AgentFoX inference.

中文说明: 开源版只要求用户提供包含 image_path 和 gt_label 的 CSV, 不依赖私有数据库。
English: The open-source version only requires a CSV with image_path and
gt_label, and does not depend on private databases.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger


class DatasetsManager:
    """Load and normalize test data.

    中文说明: 该类把一个或多个测试 CSV 合并为统一 DataFrame, 并处理相对图片路径。
    English: This class merges one or more test CSV files into a single
    DataFrame and resolves relative image paths.
    """

    REQUIRED_COLUMNS = {"image_path", "gt_label"}

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("datasets config must be a dictionary.")
        self.config = config
        self.test_paths = self._normalize_test_paths(config.get("test_paths"))
        if not self.test_paths:
            raise ValueError("datasets.test_paths is required in the open-source minimal runtime.")

        self.image_root = Path(config["image_root"]).expanduser() if config.get("image_root") else None
        self.runtime_cache_dir = self._resolve_runtime_cache_dir()
        self.runtime_cache_dir.mkdir(parents=True, exist_ok=True)

        self._detail_data = self._load_test_data()
        self._clustering_data = pd.DataFrame({"image_path": self._detail_data["image_path"].drop_duplicates()})
        self._val_data = pd.DataFrame()

    @property
    def detail_data(self) -> pd.DataFrame:
        """Return per-image rows for batch inference.

        中文说明: 最小 test 只按 image_path 去重, 不要求 model_name/expert 结果。
        English: Minimal test deduplicates by image_path and does not require
        model_name or expert predictions.
        """
        return self._detail_data.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    @property
    def clustering_data(self) -> pd.DataFrame:
        """Return placeholder clustering data.

        中文说明: clustering 在最小配置中关闭, 这里保留空壳以兼容运行时接口。
        English: Clustering is disabled in minimal config; this placeholder
        keeps the runtime interface stable.
        """
        return self._clustering_data.copy()

    @property
    def val_data(self) -> pd.DataFrame:
        """Return validation data placeholder.

        中文说明: 最小推理不需要验证集。
        English: Minimal inference does not need a validation set.
        """
        return self._val_data.copy()

    @property
    def primary_test_path(self) -> Path:
        """Return the first configured CSV path.

        中文说明: 运行时缓存默认放在第一个 CSV 的同级目录。
        English: Runtime cache defaults to the parent directory of the first
        CSV file.
        """
        return Path(self.test_paths[0])

    @staticmethod
    def _normalize_test_paths(test_paths) -> list[str]:
        """Normalize datasets.test_paths to a non-empty list.

        中文说明: 同时支持字符串和字符串列表。
        English: Both a single string and a list of strings are supported.
        """
        if isinstance(test_paths, (str, Path)):
            return [str(test_paths)]
        if isinstance(test_paths, Iterable):
            return [str(path) for path in test_paths if str(path).strip()]
        return []

    def _resolve_runtime_cache_dir(self) -> Path:
        """Choose where runtime semantic caches are stored.

        中文说明: 用户可配置 runtime_cache_dir; 否则使用 CSV 同级目录下的 .agentfox_cache。
        English: Users may configure runtime_cache_dir; otherwise `.agentfox_cache`
        beside the CSV is used.
        """
        if self.config.get("runtime_cache_dir"):
            return Path(self.config["runtime_cache_dir"]).expanduser()
        return self.primary_test_path.expanduser().parent / ".agentfox_cache"

    @staticmethod
    @lru_cache(maxsize=65536)
    def _normalize_image_path_str(path_str: str) -> str:
        """Normalize a local path without requiring it to exist.

        中文说明: 不强制图片存在, 让配置检查和单元测试可以先验证 CSV 解析。
        English: The image does not need to exist during path normalization, so
        config checks and unit tests can validate CSV parsing first.
        """
        if "://" in path_str:
            return path_str
        return Path(path_str).expanduser().resolve(strict=False).as_posix()

    @staticmethod
    def normalize_image_path(image_path) -> str:
        """Normalize an arbitrary image path value.

        中文说明: 空值返回空字符串, URL 原样保留。
        English: Empty values become an empty string, and URLs are preserved.
        """
        if image_path is None:
            return ""
        try:
            if pd.isna(image_path):
                return ""
        except Exception:
            pass
        path_str = str(image_path).strip()
        if not path_str:
            return ""
        return DatasetsManager._normalize_image_path_str(path_str)

    @staticmethod
    def candidate_image_paths(image_path) -> list[str]:
        """Return raw and normalized path candidates.

        中文说明: profile/cache 查询时同时尝试原始路径和规范化路径。
        English: Profile/cache lookups try both the raw path and normalized path.
        """
        raw = str(image_path).strip() if image_path is not None else ""
        normalized = DatasetsManager.normalize_image_path(raw)
        return [path for path in dict.fromkeys([raw, normalized]) if path]

    def _resolve_image_path(self, raw_path: str, csv_path: Path) -> str:
        """Resolve one CSV image_path entry.

        中文说明: 相对路径优先基于 datasets.image_root, 否则基于 CSV 所在目录。
        English: Relative paths are resolved against datasets.image_root first,
        otherwise against the CSV parent directory.
        """
        raw_path = str(raw_path).strip()
        if "://" in raw_path:
            return raw_path
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return self.normalize_image_path(path)
        base_dir = self.image_root or csv_path.parent
        return self.normalize_image_path(base_dir / path)

    def _load_test_data(self) -> pd.DataFrame:
        """Load and merge configured test CSV files.

        中文说明: 缺少必需列会立即报错, 防止后续 Agent 运行时才失败。
        English: Missing required columns fail fast before the agent starts.
        """
        frames: list[pd.DataFrame] = []
        for raw_csv_path in self.test_paths:
            csv_path = Path(raw_csv_path).expanduser()
            if not csv_path.exists():
                raise FileNotFoundError(f"Test CSV not found: {csv_path}")
            data = pd.read_csv(csv_path)
            missing = self.REQUIRED_COLUMNS - set(data.columns)
            if missing:
                raise ValueError(f"Missing required column(s) in {csv_path}: {sorted(missing)}")
            if data.empty:
                logger.warning(f"Test CSV is empty: {csv_path}")
                continue

            data = data.copy()
            data["image_path"] = data["image_path"].map(lambda value: self._resolve_image_path(value, csv_path))
            data["gt_label"] = pd.to_numeric(data["gt_label"], errors="raise").astype(int)
            if "dataset_name" not in data.columns:
                data["dataset_name"] = csv_path.stem
            frames.append(data[["image_path", "gt_label", "dataset_name"]])

        if not frames:
            raise ValueError("No valid rows were loaded from datasets.test_paths.")
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["image_path"], keep="last")

    def get_image_and_label(self) -> dict[str, dict[str, int]]:
        """Return labels keyed by normalized image path.

        中文说明: 保留该接口是为了兼容 Agent 工具层的状态读取。
        English: This interface is kept for compatibility with the agent tool layer.
        """
        labels = self.detail_data[["image_path", "gt_label"]].set_index("image_path")["gt_label"].to_dict()
        return {path: {"gt_label": int(label)} for path, label in labels.items()}
