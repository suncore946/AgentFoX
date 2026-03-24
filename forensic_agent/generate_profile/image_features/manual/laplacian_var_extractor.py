import numpy as np
from typing import Union, List
from pathlib import Path
import cv2

from ..base_feature_extractor import BaseFeatureExtractor
from ....utils.image_utils import load_image, to_grayscale


class LaplacianVarExtractor(BaseFeatureExtractor):
    """拉普拉斯方差特征提取器

    衡量图像的清晰度/锐度，值越大表示图像越清晰
    """

    name = "LaplacianVar"
    description = "Image clarity/sharpness metric based on the variance of the Laplacian operator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_required_dependencies(self) -> List[str]:
        return ["opencv"]

    def extract(self, images: List) -> List:
        """提取拉普拉斯方差特征"""
        results = []
        for image in images:
            img = load_image(image)
            gray = to_grayscale(img)

            # 计算拉普拉斯算子
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            # 计算方差
            results.append(np.var(laplacian))
        return results
