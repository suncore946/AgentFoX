import numpy as np
from typing import Union, List
from pathlib import Path
from ..base_feature_extractor import BaseFeatureExtractor
from ....utils.image_utils import load_image


class ColorfulnessExtractor(BaseFeatureExtractor):
    """颜色丰富度特征提取器, 基于Hasler和Susstrunk的颜色丰富度度量方法"""

    name = "Colorfulness"
    description = "图像颜色丰富度，基于RGB通道标准差和均值"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_required_dependencies(self) -> List[str]:
        return []

    def extract(self, images: List) -> List[float]:
        """提取颜色丰富度特征"""
        results = []

        for image in images:
            img = load_image(image)

            # 如果图像加载失败或为空，跳过
            if img is None:
                results.append(0.0)
                continue

            # 如果是灰度图，颜色丰富度为0
            if len(img.shape) == 2:
                results.append(0.0)
                continue

            # 确保是3通道RGB图像
            if img.shape[2] < 3:
                results.append(0.0)
                continue

            # 提取RGB通道
            R = img[:, :, 0].astype(np.float64)
            G = img[:, :, 1].astype(np.float64)
            B = img[:, :, 2].astype(np.float64)

            # 计算对手色彩空间
            rg = R - G
            yb = 0.5 * (R + G) - B

            # 计算标准差和均值
            sigma_rg = np.std(rg)
            sigma_yb = np.std(yb)
            mu_rg = np.mean(rg)
            mu_yb = np.mean(yb)

            # 计算颜色丰富度
            sigma_rgyb = np.sqrt(sigma_rg**2 + sigma_yb**2)
            mu_rgyb = np.sqrt(mu_rg**2 + mu_yb**2)

            colorfulness = sigma_rgyb + 0.3 * mu_rgyb

            results.append(float(colorfulness))

        return results
