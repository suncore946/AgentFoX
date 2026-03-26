import numpy as np
from typing import Union, List
from pathlib import Path
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from ..base_feature_extractor import BaseFeatureExtractor
from ....utils.image_utils import load_image, to_grayscale


class SRMExtractor(BaseFeatureExtractor):
    """SRM残差共生特征提取器

    基于空间丰富模型(Spatial Rich Model)计算图像残差的共生矩阵特征，
    """

    name = "SRM"
    description = "Applies Spatial Rich Model (SRM) filters to extract residual images through convolution operations, then computes Gray Level Co-occurrence Matrix (GLCM) on the residuals to derive texture statistics including contrast, correlation, energy, and homogeneity. These features capture subtle texture variations and are widely used in steganalysis and tampering detection"

    def __init__(
        self,
        residual_type: str = "spam14h",
        distances: List[int] = [1, 2],
        angles: List[float] = [0, 45, 90, 135],
        levels: int = 32,
        **kwargs,
    ):
        """
        Args:
            residual_type: 残差类型 ('spam14h', 'spam14v', 'minmax21', 'minmax41')
            distances: 共生矩阵计算距离列表
            angles: 共生矩阵计算角度列表(度)
            levels: 灰度级数 (建议使用较小值如32以提高计算效率)
        """
        params = {
            "name": "SRM",
            "residual_type": residual_type,
            "distances": distances,
            "angles": angles,
            "levels": levels,
        }
        params.update(kwargs)

        super().__init__(**params)
        self.residual_type = residual_type
        self.distances = distances
        # 修正：保持角度为度数，graycomatrix函数需要弧度
        self.angles_deg = angles
        self.angles = [np.radians(angle) for angle in angles]
        self.levels = levels

        # 预定义SRM滤波器
        self._init_srm_filters()

    def _init_srm_filters(self):
        """初始化SRM滤波器"""
        self.filters = {
            "spam14h": np.array(
                [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float64
            )
            / 12.0,
            "spam14v": np.array(
                [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float64
            ).T
            / 12.0,
            "minmax21": np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64) / 4.0,
            "minmax41": np.array(
                [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float64
            )
            / 4.0,
        }

    def _compute_residual(self, image: np.ndarray) -> np.ndarray:
        """计算SRM残差，使用scipy库函数"""
        if self.residual_type not in self.filters:
            raise ValueError(f"Unsupported residual type: {self.residual_type}")

        filter_kernel = self.filters[self.residual_type]

        # 使用scipy.ndimage进行卷积，镜像边界处理
        residual = ndimage.convolve(image.astype(np.float64), filter_kernel, mode="mirror")

        # 修正：使用更鲁棒的量化方法
        # 将残差映射到[0, levels-1]范围
        residual_min = np.min(residual)
        residual_max = np.max(residual)

        if residual_max > residual_min:
            residual_normalized = (residual - residual_min) / (residual_max - residual_min)
            residual_quantized = (residual_normalized * (self.levels - 1)).astype(np.uint8)
        else:
            # 如果残差为常数，设为0
            residual_quantized = np.zeros_like(residual, dtype=np.uint8)

        return residual_quantized

    def _compute_glcm_features(self, residual: np.ndarray) -> List[float]:
        """使用skimage库计算GLCM特征"""
        features = []

        try:
            # 使用skimage的graycomatrix计算GLCM
            glcm = graycomatrix(residual, distances=self.distances, angles=self.angles, levels=self.levels, symmetric=True, normed=True)

            # 使用skimage的graycoprops计算特征
            properties = ["contrast", "correlation", "energy", "homogeneity"]

            for prop in properties:
                try:
                    feature_values = graycoprops(glcm, prop)
                    # 展平特征值并添加到特征列表
                    features.extend(feature_values.flatten().tolist())
                except Exception as e:
                    # 如果某个特征计算失败，用0填充
                    features.extend([0.0] * (len(self.distances) * len(self.angles)))

        except Exception as e:
            # 如果整个GLCM计算失败，返回零特征
            num_features = len(["contrast", "correlation", "energy", "homogeneity"]) * len(self.distances) * len(self.angles)
            features = [0.0] * num_features

        return features

    def extract(self, images: List[Union[str, Path, np.ndarray]]) -> List[List[float]]:
        """批量提取SRM残差共生特征

        Args:
            images: 图片路径、Path对象或ndarray列表

        Returns:
            每张图片的特征向量列表
        """
        results = []
        for image in images:
            img = load_image(image)
            gray = to_grayscale(img)

            # 计算残差
            residual = self._compute_residual(gray)

            # 计算GLCM特征
            features = self._compute_glcm_features(residual)

            # 确保特征向量不包含NaN或无穷大
            features = [float(f) if np.isfinite(f) else 0.0 for f in features]

            results.append(features)
        return results

    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        names = []
        properties = ["contrast", "correlation", "energy", "homogeneity"]

        for prop in properties:
            for distance in self.distances:
                for angle_deg in self.angles_deg:
                    names.append(f"{self.residual_type}_{prop}_d{distance}_a{int(angle_deg)}")

        return names

    def get_required_dependencies(self) -> List[str]:
        return ["scipy", "scikit-image", "scikit-learn", "numpy"]
