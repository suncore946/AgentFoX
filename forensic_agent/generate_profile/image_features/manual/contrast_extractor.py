import numpy as np
from typing import Union, List
from pathlib import Path
from ..base_feature_extractor import BaseFeatureExtractor
from ....utils.image_utils import load_image, to_grayscale


class ContrastExtractor(BaseFeatureExtractor):
    """对比度特征提取器

    基于95%分位数和5%分位数的差值计算对比度
    """

    name = "Contrast_P95_P5"
    description = "图像对比度，基于95%和5%分位数的差值"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_required_dependencies(self) -> List[str]:
        return []

    def extract(self, images: List) -> List:
        """提取对比度特征"""
        contrasts = []
        for image in images:
            img = load_image(image)
            gray = to_grayscale(img)

            # 计算5%和95%分位数
            p5 = np.percentile(gray, 5)
            p95 = np.percentile(gray, 95)

            # 计算对比度
            contrast = p95 - p5
            contrasts.append(contrast)
        return contrasts
