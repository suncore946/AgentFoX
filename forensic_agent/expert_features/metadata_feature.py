from typing import Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from .base_feat import FeatureExtractorBase


class MetadataExtractor(FeatureExtractorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_features(self, image: Image.Image, image_path: Path, *args, **kwargs) -> Dict[str, Any]:
        """
        提取图像的基本特征：图像尺寸、压缩比、模糊系数
        """
        file_size = Path(image_path).stat().st_size
        compression_ratio = self._calculate_compression_ratio(image, file_size)
        laplacian_score = self._calculate_laplacian(image)
        return {"img_size": image.size, "compression_ratio": float(compression_ratio), "laplacian": float(laplacian_score)}

    def _calculate_compression_ratio(self, image: Image.Image, file_size: int) -> float:
        """计算压缩比"""
        try:
            width, height = image.size

            # 简化的字节数计算
            mode_to_bytes = {"L": 1, "LA": 2, "P": 1, "PA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "HSV": 3, "LAB": 3}

            bytes_per_pixel = mode_to_bytes.get(image.mode, 3)

            # 大多数情况下假设8位深度
            uncompressed_size = width * height * bytes_per_pixel

            if file_size > 0:
                compression_ratio = uncompressed_size / file_size
                return round(compression_ratio, 4)
            else:
                return 0.0

        except Exception:
            return None  # 保持一致性

    def _calculate_laplacian(self, image: Image.Image) -> float:
        """计算模糊系数, 拉普拉斯方差"""
        try:
            # 先将PIL Image转换为numpy数组
            image_array = np.array(image)

            # 如果是彩色图像，先转为灰度
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array

            # 应用拉普拉斯算子
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = round(laplacian.var(), 4)
            return blur_score
        except Exception:
            return None  # 保持一致性
