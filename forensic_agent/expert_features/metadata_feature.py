"""Metadata and simple image statistics.

中文说明: 这些特征只来自图像文件本身, 不访问外部模型或私有数据库。
English: These features come only from the image file itself and do not access
external models or private databases.
"""

from typing import Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from .base_feat import FeatureExtractorBase


class MetadataExtractor(FeatureExtractorBase):
    """Extract basic metadata-like statistics.

    中文说明: 输出用于给 VLM 提供尺寸、压缩和模糊程度上下文。
    English: Outputs size, compression, and blur context for the VLM prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_features(self, image: Image.Image, image_path: Path, *args, **kwargs) -> Dict[str, Any]:
        """Extract image size, compression ratio, and Laplacian blur.

        中文说明: image_path 仅用于读取文件大小。
        English: image_path is used only to read file size.
        """
        file_size = Path(image_path).stat().st_size
        compression_ratio = self._calculate_compression_ratio(image, file_size)
        laplacian_score = self._calculate_laplacian(image)
        return {"img_size": image.size, "compression_ratio": float(compression_ratio), "laplacian": float(laplacian_score)}

    def _calculate_compression_ratio(self, image: Image.Image, file_size: int) -> float:
        """Estimate compression ratio.

        中文说明: 这是近似统计量, 只作为语义提示辅助信息。
        English: This is an approximate statistic and only auxiliary prompt
        context.
        """
        try:
            width, height = image.size

            mode_to_bytes = {"L": 1, "LA": 2, "P": 1, "PA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "HSV": 3, "LAB": 3}

            bytes_per_pixel = mode_to_bytes.get(image.mode, 3)

            uncompressed_size = width * height * bytes_per_pixel

            if file_size > 0:
                compression_ratio = uncompressed_size / file_size
                return round(compression_ratio, 4)
            else:
                return 0.0

        except Exception:
            return None

    def _calculate_laplacian(self, image: Image.Image) -> float:
        """Compute Laplacian variance as a blur cue.

        中文说明: 值越低通常表示图像越模糊。
        English: Lower values usually indicate a blurrier image.
        """
        try:
            image_array = np.array(image)

            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = round(laplacian.var(), 4)
            return blur_score
        except Exception:
            return None
