import numpy as np
from typing import Union, List
from pathlib import Path
from ..base_feature_extractor import BaseFeatureExtractor
from ....utils.image_utils import load_image, to_grayscale


class HFLFRatioExtractor(BaseFeatureExtractor):
    """高低频比率特征提取器

    计算图像高频成分与低频成分的比率，反映图像的细节丰富程度
    """

    name = "HF_LF_Ratio"
    description = "High-to-low frequency energy ratio of the image, reflecting level of fine detail"

    def __init__(self, cutoff_ratio: float = 0.3, **kwargs):
        """
        Args:
            cutoff_ratio: 高低频分割比例，默认0.3表示30%的频率作为分割点
        """
        super().__init__(cutoff_ratio=cutoff_ratio, **kwargs)
        self.cutoff_ratio = cutoff_ratio

    def get_required_dependencies(self) -> List[str]:
        return []  # 只需要numpy

    def extract(self, images: List) -> List:
        """提取高低频比率特征"""
        ratios = []  # 用于存储每张图像的高低频比率

        for image in images:
            img = load_image(image)
            gray = to_grayscale(img).astype(np.float64)

            # FFT变换
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            # 获取频域中心
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            # 创建频率掩码
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
            max_distance = np.sqrt(center_h**2 + center_w**2)

            # 分割高频和低频
            cutoff_distance = self.cutoff_ratio * max_distance

            lf_mask = distances <= cutoff_distance
            hf_mask = distances > cutoff_distance

            # 计算能量
            lf_energy = np.sum(magnitude[lf_mask] ** 2)
            hf_energy = np.sum(magnitude[hf_mask] ** 2)

            # 计算比率，避免除零
            if lf_energy < 1e-10:
                ratio = float("inf") if hf_energy > 0 else 1.0
            else:
                ratio = hf_energy / lf_energy

            ratios.append(ratio)  # 将比率添加到结果列表中

        return ratios
