"""Spatial-domain lightweight feature extraction.

中文说明: 提取 LBP、GLCM 和边缘统计, 用作语义分析提示上下文。
English: Extracts LBP, GLCM, and edge statistics as context for semantic
analysis prompts.
"""

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
    """Extract spatial texture and edge cues.

    中文说明: 不使用任何训练权重, 只计算传统图像统计量。
    English: Uses no trained weights and computes only classical image
    statistics.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enabled spatial features.

        中文说明: 用户可在 feature_extraction.spatial 中关闭某些统计项以加速。
        English: Users can disable individual statistics under
        feature_extraction.spatial for speed.
        """
        self.enable_lbp = config.get("enable_lbp", True)
        self.enable_glcm = config.get("enable_glcm", True)
        self.enable_edge = config.get("enable_edge", True)

    def extract_features(self, image: Image.Image, *args, **kwargs):
        """Extract spatial-domain features.

        中文说明: 输出结构包含 features 和 meta, 便于 VLM 理解每组统计的含义。
        English: Output contains features and meta so the VLM can interpret each
        statistic group.
        """
        gray_image = np.array(image.convert("L"))
        try:
            features = {}

            if self.enable_lbp:
                lbp_features = self._extract_lbp_features(gray_image)
                features["lbp"] = {
                    "features": lbp_features,
                    "meta": {"radius": 3, "n_points": 8 * 3, "method": "uniform"},
                }
            if self.enable_glcm:
                glcm_features = self._extract_glcm_features(gray_image)
                features["glcm"] = {
                    "features": glcm_features,
                    "meta": {"distances": [1, 2, 3], "angles": [0, 45, 90, 135], "levels": 64},
                }
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
        """Extract Local Binary Pattern features.

        中文说明: 使用 scikit-image 标准 LBP 实现并返回归一化直方图。
        English: Uses scikit-image's standard LBP implementation and returns a
        normalized histogram.
        """
        radius = 3
        n_points = 8 * radius
        method = "uniform"

        lbp = local_binary_pattern(image, n_points, radius, method)

        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        hist = hist.astype(float)
        hist /= hist.sum() + 1e-8

        return hist

    def _extract_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """Extract GLCM texture features.

        中文说明: 量化到 64 级灰度以控制计算量。
        English: Quantizes to 64 gray levels to control computation cost.
        """

        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        levels = 64

        quantized = (image / 255.0 * (levels - 1)).astype(np.uint8)
        quantized = np.clip(quantized, 0, levels - 1)

        glcm = graycomatrix(quantized, distances=distances, angles=np.deg2rad(angles), levels=levels, symmetric=True, normed=True)

        properties = ["contrast", "dissimilarity", "homogeneity", "energy"]
        features = []

        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            features.extend(prop_values.ravel())

        return np.array(features)

    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Canny edge density and direction histogram.

        中文说明: 这些统计帮助 VLM 判断纹理边界是否自然。
        English: These statistics help the VLM reason about naturalness of
        texture boundaries.
        """
        edges = cv2.Canny(image, 50, 150)

        edge_density = np.sum(edges > 0) / edges.size

        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        edge_mask = edges > 0
        if np.sum(edge_mask) > 0:
            edge_directions = np.arctan2(grad_y[edge_mask], grad_x[edge_mask])
            direction_hist, _ = np.histogram(edge_directions, bins=8, range=(-np.pi, np.pi))
            direction_hist = direction_hist.astype(float) / (direction_hist.sum() + 1e-8)
        else:
            direction_hist = np.zeros(8)

        edge_features = np.array([edge_density])
        edge_features = np.concatenate([edge_features, direction_hist])

        return edge_features
