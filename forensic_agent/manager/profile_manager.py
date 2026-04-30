"""Runtime semantic profile cache.

中文说明: 最小开源版不附带实验 profile JSON, 只在运行时缓存语义分析结果。
English: The minimal open-source release does not ship experiment profile JSON;
it only caches semantic analysis results at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .base_manager import BaseManager
from .datasets_manager import DatasetsManager


class ProfileManager(BaseManager):
    """Manage runtime semantic profiles.

    中文说明: semantic_profiles 可选; 未配置时会自动写到测试 CSV 同级缓存目录。
    English: semantic_profiles is optional; when omitted, results are written to
    the cache directory beside the test CSV.
    """

    def __init__(self, models_config: dict | None):
        self.config = models_config or {}
        self.logger = logger
        self.semantic_profiles_path = self._resolve_optional_path(self.config.get("semantic_profiles"))
        self._semantic_profiles: dict[str, Any] = self._load_semantic_profiles()

    @staticmethod
    def _resolve_optional_path(path_value: str | Path | None) -> Path | None:
        """Resolve an optional path.

        中文说明: 空配置保留为 None, 由 ensure_profile_paths 根据数据集补齐。
        English: Empty values remain None and are later filled by
        ensure_profile_paths based on the dataset.
        """
        if not path_value:
            return None
        return Path(path_value).expanduser()

    @property
    def semantic_profiles(self) -> dict[str, Any]:
        """Return loaded semantic profile cache.

        中文说明: 返回内存缓存, 调用方不要直接修改。
        English: Returns the in-memory cache; callers should not mutate it directly.
        """
        return self._semantic_profiles

    def ensure_profile_paths(self, datasets_manager: DatasetsManager | None = None, kind: str | None = None) -> None:
        """Ensure the runtime semantic cache path exists.

        中文说明: kind 参数保留为兼容接口, 最小版只支持 semantic。
        English: The kind argument is kept for interface compatibility; the
        minimal version only supports semantic profiles.
        """
        if self.semantic_profiles_path is None and datasets_manager is not None:
            self.semantic_profiles_path = datasets_manager.runtime_cache_dir / "semantic_profiles.json"
        if kind in {None, "semantic"}:
            self._semantic_profiles = self._load_semantic_profiles()

    def _load_semantic_profiles(self) -> dict[str, Any]:
        """Load cached semantic profiles if present.

        中文说明: 缓存文件不存在是正常情况, 首次运行会按需生成。
        English: A missing cache file is normal; the first run generates entries
        on demand.
        """
        if self.semantic_profiles_path and self.semantic_profiles_path.exists():
            with open(self.semantic_profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        return {}

    @staticmethod
    def _is_valid_semantic_profile(profile: Any) -> bool:
        """Validate the semantic profile shape.

        中文说明: 只接受 reporter 和 semantic tool 需要的三个核心字段。
        English: Only the three fields required by the reporter and semantic
        tool are accepted.
        """
        return isinstance(profile, dict) and {"observations", "detected_anomalies", "pred_label"}.issubset(profile)

    def get_semantic_profile(self, image_path: str) -> dict[str, Any]:
        """Return the cached semantic profile for an image.

        中文说明: 同时尝试原始路径和规范化路径, 便于兼容不同 CSV 写法。
        English: Both raw and normalized path candidates are tried to support
        different CSV path styles.
        """
        for candidate in DatasetsManager.candidate_image_paths(image_path):
            profile = self._semantic_profiles.get(candidate)
            if self._is_valid_semantic_profile(profile):
                return profile
        return {}

    def save_runtime_semantic_profile(
        self,
        image_path: str,
        semantic_profile: dict[str, Any],
        datasets_manager: DatasetsManager | None = None,
    ) -> None:
        """Persist one semantic profile.

        中文说明: 写入前会再次加载缓存, 避免同一批次重复覆盖其他样本结果。
        English: The cache is reloaded before writing so a batch run does not
        overwrite results from other samples.
        """
        if not self._is_valid_semantic_profile(semantic_profile):
            self.logger.warning(f"Invalid semantic profile skipped for image: {image_path}")
            return
        self.ensure_profile_paths(datasets_manager=datasets_manager, kind="semantic")
        if self.semantic_profiles_path is None:
            return

        key = DatasetsManager.normalize_image_path(image_path)
        self._semantic_profiles[key] = semantic_profile
        self.semantic_profiles_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.semantic_profiles_path.with_suffix(self.semantic_profiles_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._semantic_profiles, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.semantic_profiles_path)
