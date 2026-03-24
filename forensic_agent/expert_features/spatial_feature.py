import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
from typing import Dict, Any, Tuple, Union
from pathlib import Path
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

from ..core.core_exceptions import FeatureExtractionError
from .base_feat import FeatureExtractorBase


class SpatialFeatureExtractor(FeatureExtractorBase):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.enable_lbp = config.get("enable_lbp", True)
        self.enable_glcm = config.get("enable_glcm", True)
        self.enable_edge = config.get("enable_edge", True)

    def extract_features(self, image: Image.Image, *args, **kwargs):
        """
        提取空间域特征

        Args:
            image: PIL图像对象

        Returns:
            Tuple[特征向量, 元数据]
        """
        gray_image = np.array(image.convert("L"))  # 转换为灰度图像
        try:
            features = {}

            # 提取LBP特征
            if self.enable_lbp:
                lbp_features = self._extract_lbp_features(gray_image)
                features["lbp"] = {
                    "features": lbp_features,
                    "meta": {"radius": 3, "n_points": 8 * 3, "method": "uniform"},
                }
            # 提取GLCM特征
            if self.enable_glcm:
                glcm_features = self._extract_glcm_features(gray_image)
                features["glcm"] = {
                    "features": glcm_features,
                    "meta": {"distances": [1, 2, 3], "angles": [0, 45, 90, 135], "levels": 64},
                }
            # 提取边缘特征
            if self.enable_edge:
                edge_features = self._extract_edge_features(gray_image)
                features["edge"] = {
                    "features": edge_features,
                    "meta": {"canny_thresholds": [50, 150], "direction_bins": 8},
                }
            return features

        except Exception as e:
            raise FeatureExtractionError(f"空间域特征提取失败: {e}", feature_type="spatial") from e

    def _extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """提取局部二值模式(LBP)特征

        优先使用scikit-image的标准LBP实现，如果不可用则使用基于梯度的简化版本

        Args:
            image: 输入的灰度图像

        Returns:
            归一化的LBP特征向量
        """
        # LBP参数配置
        radius = 3
        n_points = 8 * radius  # 24个采样点
        method = "uniform"  # 使用均匀模式

        # 计算LBP
        lbp = local_binary_pattern(image, n_points, radius, method)

        # 计算LBP直方图
        n_bins = n_points + 2  # uniform模式的bin数量
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # 归一化处理
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-8  # 避免除零

        return hist

    def _extract_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """提取灰度共生矩阵(GLCM)特征"""

        # GLCM参数
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        levels = 64

        # 量化图像，确保值在0到levels-1范围内
        quantized = (image / 255.0 * (levels - 1)).astype(np.uint8)
        # 进一步确保没有超出范围的值
        quantized = np.clip(quantized, 0, levels - 1)

        # 计算GLCM
        glcm = graycomatrix(quantized, distances=distances, angles=np.deg2rad(angles), levels=levels, symmetric=True, normed=True)

        # 提取GLCM属性
        properties = ["contrast", "dissimilarity", "homogeneity", "energy"]
        features = []

        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            features.extend(prop_values.ravel())

        return np.array(features)

    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """提取边缘特征"""
        # Canny边缘检测
        edges = cv2.Canny(image, 50, 150)

        # 计算边缘密度
        edge_density = np.sum(edges > 0) / edges.size

        # 计算边缘方向直方图
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # 只考虑边缘像素的梯度方向
        edge_mask = edges > 0
        if np.sum(edge_mask) > 0:
            edge_directions = np.arctan2(grad_y[edge_mask], grad_x[edge_mask])
            direction_hist, _ = np.histogram(edge_directions, bins=8, range=(-np.pi, np.pi))
            direction_hist = direction_hist.astype(float) / (direction_hist.sum() + 1e-8)
        else:
            direction_hist = np.zeros(8)

        # 合并特征
        edge_features = np.array([edge_density])
        edge_features = np.concatenate([edge_features, direction_hist])

        return edge_features
