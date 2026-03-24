from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import numpy as np
from PIL import Image


class FeatureExtractorBase(ABC):
    """统计特征提取器抽象基类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化统计特征提取器

        Args:
            config: 特征提取配置
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
        """
        提取特征的抽象方法

        Args:
            image_path: 图像路径

        Returns:
            Tuple[特征向量, 元数据]

        Raises:
            FeatureExtractionError: 特征提取失败时抛出
        """
        pass
