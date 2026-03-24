import numpy as np
from typing import List
from ..base_feature_extractor import BaseFeatureExtractor
from scipy import fftpack
from ....utils.image_utils import load_image, to_grayscale


class JPEGQExtractor(BaseFeatureExtractor):
    """JPEG压缩质量估计特征提取器

    估计图像的JPEG压缩质量因子，值越大表示压缩程度越低（质量越好）
    """

    name = "JPEG_Q"
    description = "JPEG压缩质量估计，基于量化表分析"

    def __init__(self, **kwargs):
        super().__init__(name="JPEG_Q", **kwargs)

    def get_required_dependencies(self) -> List[str]:
        return ["scipy"]

    def extract(self, images: List) -> List:
        """估计JPEG压缩质量

        基于DCT系数的分布特性来估计压缩质量
        """
        qualities = []

        for image in images:
            img = load_image(image)
            gray = to_grayscale(img).astype(np.float64)

            # 8x8分块DCT变换
            h, w = gray.shape

            # 确保尺寸是8的倍数
            h_blocks = h // 8
            w_blocks = w // 8

            if h_blocks == 0 or w_blocks == 0:
                qualities.append(50.0)  # 默认质量值
                continue

            # 提取8x8块进行DCT
            dct_coeffs = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]

                    # DCT变换
                    dct_block = fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")
                    dct_coeffs.append(dct_block.flatten())

            if not dct_coeffs:
                qualities.append(50.0)
                continue

            # 分析DCT系数的量化特性
            all_coeffs = np.concatenate(dct_coeffs)

            # 计算系数的分布特性
            # 1. 非零系数比例
            nonzero_ratio = np.count_nonzero(np.abs(all_coeffs) > 1e-6) / len(all_coeffs)

            # 2. 高频系数的能量
            hf_energy = np.mean(np.abs(all_coeffs[len(all_coeffs) // 2 :]))

            # 3. 系数的标准差
            coeff_std = np.std(all_coeffs)

            # 估计质量因子（经验公式）
            quality = min(100, max(1, 100 * nonzero_ratio * (1 + hf_energy / coeff_std)))

            qualities.append(quality)

        return qualities
