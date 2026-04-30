"""Image feature preparation for semantic analysis.

中文说明: 该处理器汇总轻量图像特征并调用 VLM 生成语义鉴伪 profile。
English: This processor gathers lightweight image features and calls a VLM to
produce a semantic forensic profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from ..core.core_exceptions import FeatureExtractionError
from ..expert_features.frequency_feature import FrequencyFeatureExtractor
from ..expert_features.metadata_feature import MetadataExtractor
from ..expert_features.spatial_feature import SpatialFeatureExtractor
from ..manager.base_manager import BaseManager
from ..manager.image_manager import ImageManager
from .semantic_forgery_tracking_processor import SemanticForgeryTrackingProcessor


class ImageFeatProcessor(BaseManager):
    """Build semantic-analysis inputs for one image.

    中文说明: 特征只作为 VLM 提示的辅助上下文, 不需要训练权重或私有资源。
    English: Features are only auxiliary context for the VLM prompt and do not
    require training weights or private resources.
    """

    DEFAULT_SPATIAL_CONFIG = {"enable_lbp": True, "enable_glcm": True, "enable_edge": True}
    DEFAULT_FREQUENCY_CONFIG = {
        "enable_fft": True,
        "enable_dct": True,
        "enable_wavelet": False,
        "wavelet_type": "db4",
        "wavelet_levels": 3,
    }

    def __init__(self, config: dict[str, Any], image_manager: ImageManager, semantic_llm) -> None:
        self.config = config or {}
        self.logger = logger
        self.image_manager = image_manager
        feature_config = self.config.get("feature_extraction", self.config)

        self.metadata_extractor = MetadataExtractor()
        self.spatial_extractor = SpatialFeatureExtractor(feature_config.get("spatial", self.DEFAULT_SPATIAL_CONFIG))
        self.frequency_extractor = FrequencyFeatureExtractor(feature_config.get("frequency", self.DEFAULT_FREQUENCY_CONFIG))

        prompt_path = self.config.get("prompt_path")
        if not prompt_path:
            raise ValueError("SemanticAnalysis.prompt_path is required.")
        self.forensic_traces = SemanticForgeryTrackingProcessor(prompt_path=prompt_path, forensic_llm=semantic_llm)

    def run(self, image_path: str | Path, *args, **kwargs):
        """Generate a semantic forensic profile.

        中文说明: 返回值是 (profile, image), profile 至少包含 observations/detected_anomalies/pred_label。
        English: Returns (profile, image); profile contains at least
        observations, detected_anomalies, and pred_label.
        """
        image_base64, image, image_format = self.image_manager.get_base64(image_path, is_resize=False)
        image_features = {
            "metadata": self.metadata_extractor.extract_features(image_path=image_path, image=image),
            "spatial": self.spatial_extractor.extract_features(image=image),
            "frequency": self.frequency_extractor.extract_features(image=image),
        }
        result = self.forensic_traces.run(
            image=image,
            image_path=Path(image_path),
            image_base64=image_base64,
            image_format=image_format,
            image_features=image_features,
        )
        if isinstance(result, dict) and "error" in result:
            raise FeatureExtractionError(result["error"])
        return result, image
