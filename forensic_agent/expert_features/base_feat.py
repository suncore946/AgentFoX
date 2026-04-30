"""Lightweight feature extractor interfaces.

中文说明: 这里的 expert_features 不是外部专家模型, 只提供语义提示用的轻量统计特征。
English: These expert_features are not external expert models; they provide
lightweight statistical features for semantic prompts only.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import numpy as np
from PIL import Image


class FeatureExtractorBase(ABC):
    """Base class for lightweight feature extractors.

    中文说明: 子类应只读取当前图片, 不加载训练权重或私有资源。
    English: Subclasses should read only the current image and must not load
    training weights or private resources.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize extractor config.

        中文说明: config 只包含开源 YAML 中的轻量参数。
        English: config contains only lightweight parameters from the open-source
        YAML.
        """
        self.config = config
        self.feature_type = self.__class__.__name__.lower().replace("extractor", "")

    @abstractmethod
    def extract_features(
        self,
        image: Image.Image,
        image_base64: str,
        image_format,
        *args,
        **kwargs,
    ):
        """Extract features from one image.

        中文说明: 返回值必须可被 CustomJsonEncoder 转为 JSON。
        English: Return values must be serializable by CustomJsonEncoder.
        """
        pass
