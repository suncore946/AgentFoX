import numpy as np
from typing import Union, List
from pathlib import Path
import cv2
from scipy import ndimage
from ..base_feature_extractor import BaseFeatureExtractor
from ....utils.image_utils import load_image, to_grayscale


class EdgeDensityExtractor(BaseFeatureExtractor):
    """边缘密度特征提取器

    计算图像中边缘像素的密度，反映图像的结构复杂度
    """

    name = "EdgeDensity"
    description = "Image edge pixel density, reflecting structural complexity"

    def __init__(self, threshold: float = 0.1, **kwargs):
        """
        Args:
            threshold: 边缘检测阈值，相对于最大梯度的比例
        """
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold

    def get_required_dependencies(self) -> List[str]:
        return ["opencv", "scipy"]

    def extract(
        self,
        images: List,
        use_canny: bool = True,
        use_sobel: bool = False,
    ) -> List:
        """提取边缘密度特征
        1. Canny 边缘检测
        2. Sobel 算子 + 自适应阈值

        Args:
            images: 图像列表
            use_canny: 是否使用 Canny 边缘检测
            use_sobel: 是否使用 Sobel 算子

        Returns:
            List: 每张图像的边缘密度特征列表
        """
        edge_densities = []

        for image in images:
            img = load_image(image)
            gray = to_grayscale(img).astype(np.float64)

            if use_canny:
                # 使用Canny边缘检测
                sigma = 0.33
                median = np.median(gray)
                lower = int(max(0, (1.0 - sigma) * median))
                upper = int(min(255, (1.0 + sigma) * median))
                edges = cv2.Canny(gray.astype(np.uint8), lower, upper)
                edge_density = np.sum(edges > 0) / edges.size
            elif use_sobel:
                # 使用Sobel算子
                sobel_x = ndimage.sobel(gray, axis=1)
                sobel_y = ndimage.sobel(gray, axis=0)
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

                # 自适应阈值
                threshold_value = self.threshold * np.max(gradient_magnitude)
                edges = gradient_magnitude > threshold_value
                edge_density = np.sum(edges) / edges.size
            else:
                raise ValueError("Either 'use_canny' or 'use_sobel' must be True.")

            edge_densities.append(edge_density)

        return edge_densities
