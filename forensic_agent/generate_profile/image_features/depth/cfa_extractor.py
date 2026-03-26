import numpy as np
from typing import Union, List, Tuple
from pathlib import Path
from scipy import ndimage, fftpack
from skimage.feature import graycomatrix, graycoprops
from ..base_feature_extractor import BaseFeatureExtractor
from ....utils.image_utils import load_image, to_grayscale


class CFAExtractor(BaseFeatureExtractor):
    """CFA (Color Filter Array) 特征提取器

    基于CFA插值痕迹检测图像是否经过重采样处理，
    用于图像伪造检测和真实性验证
    """

    name = "CFA"
    description = "Image features based on Color Filter Array interpolation artifacts. Detects CFA interpolation patterns by applying multiple filters (Laplacian, Prewitt, etc.) and extracts statistical features (mean, variance, skewness, kurtosis, zero-crossing rate), texture features (GLCM contrast, correlation, energy, homogeneity), block-wise features (statistics of block variance and mean), and periodicity features (FFT frequency domain analysis of 2x2 periodic patterns and energy distribution)"

    def __init__(
        self,
        filter_types: List[str] = ["laplacian", "prewitt_h", "prewitt_v"],
        periodicity_analysis: bool = True,
        texture_analysis: bool = True,
        statistical_analysis: bool = True,
        block_size: int = 64,
        **kwargs,
    ):
        """
        Args:
            filter_types: 使用的滤波器类型
            periodicity_analysis: 是否进行周期性分析
            texture_analysis: 是否进行纹理分析
            statistical_analysis: 是否进行统计分析
            block_size: 分块分析的块大小
        """
        params = {
            "name": "CFA",
            "description": "Image features based on Color Filter Array interpolation artifacts. Detects CFA interpolation patterns by applying multiple filters (Laplacian, Prewitt, etc.) and extracts statistical features (mean, variance, skewness, kurtosis, zero-crossing rate), texture features (GLCM contrast, correlation, energy, homogeneity), block-wise features (statistics of block variance and mean), and periodicity features (FFT frequency domain analysis of 2x2 periodic patterns and energy distribution)",
            "filter_types": filter_types,
            "periodicity_analysis": periodicity_analysis,
            "texture_analysis": texture_analysis,
            "statistical_analysis": statistical_analysis,
            "block_size": block_size,
        }
        params.update(kwargs)  # 如果kwargs中有这些参数的值，会覆盖上面的默认值

        super().__init__(**params)
        self.filter_types = filter_types
        self.periodicity_analysis = periodicity_analysis
        self.texture_analysis = texture_analysis
        self.statistical_analysis = statistical_analysis
        self.block_size = block_size

        # 初始化滤波器
        self._init_filters()

    def _init_filters(self):
        """初始化CFA检测滤波器"""
        self.filters = {
            # 拉普拉斯滤波器 - 检测插值痕迹
            "laplacian": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64),
            # Prewitt水平边缘检测
            "prewitt_h": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64),
            # Prewitt垂直边缘检测
            "prewitt_v": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64),
            # CFA特定滤波器 - 检测2x2重复模式
            "cfa_2x2": np.array([[1, -1], [-1, 1]], dtype=np.float64),
            # 高频滤波器
            "high_pass": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64) / 8.0,
        }

    def _apply_filter(self, image: np.ndarray, filter_name: str) -> np.ndarray:
        """应用指定滤波器"""
        if filter_name not in self.filters:
            raise ValueError(f"Unknown filter: {filter_name}")

        kernel = self.filters[filter_name]
        filtered = ndimage.convolve(image.astype(np.float64), kernel, mode="reflect")

        return filtered

    def _extract_periodicity_features(self, image: np.ndarray) -> List[float]:
        """提取周期性特征 - 检测CFA模式的周期性"""
        features = []

        try:
            # FFT分析检测周期性
            fft_image = fftpack.fft2(image)
            fft_magnitude = np.abs(fft_image)

            # 检测2x2周期性（CFA的基本模式）
            h, w = image.shape

            # 在频域中查找2x2周期对应的峰值
            period_2_h = fft_magnitude[h // 2, :]
            period_2_v = fft_magnitude[:, w // 2]

            # 统计特征
            features.extend(
                [np.mean(period_2_h), np.std(period_2_h), np.max(period_2_h), np.mean(period_2_v), np.std(period_2_v), np.max(period_2_v)]
            )

            # 计算频域能量分布
            low_freq_energy = np.sum(fft_magnitude[: h // 4, : w // 4])
            mid_freq_energy = np.sum(fft_magnitude[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4])
            high_freq_energy = np.sum(fft_magnitude[3 * h // 4 :, 3 * w // 4 :])

            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
            if total_energy > 0:
                features.extend([low_freq_energy / total_energy, mid_freq_energy / total_energy, high_freq_energy / total_energy])
            else:
                features.extend([0.0, 0.0, 0.0])

        except Exception as e:
            # 如果FFT分析失败，返回零特征
            features = [0.0] * 9

        return features

    def _extract_texture_features(self, filtered_image: np.ndarray) -> List[float]:
        """提取纹理特征"""
        features = []

        try:
            # 量化滤波后的图像
            img_min, img_max = np.min(filtered_image), np.max(filtered_image)
            if img_max > img_min:
                normalized = (filtered_image - img_min) / (img_max - img_min)
                quantized = (normalized * 31).astype(np.uint8)  # 32灰度级
            else:
                quantized = np.zeros_like(filtered_image, dtype=np.uint8)

            # 计算GLCM
            distances = [1, 2]
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

            glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

            # 提取GLCM属性
            properties = ["contrast", "correlation", "energy", "homogeneity"]
            for prop in properties:
                prop_values = graycoprops(glcm, prop)
                features.extend([np.mean(prop_values), np.std(prop_values)])

        except Exception as e:
            # 如果纹理分析失败，返回零特征
            features = [0.0] * 8  # 4个属性 * 2个统计量

        return features

    def _extract_statistical_features(self, filtered_image: np.ndarray) -> List[float]:
        """提取统计特征"""
        features = []

        # 基本统计量
        features.extend(
            [
                np.mean(filtered_image),
                np.std(filtered_image),
                np.var(filtered_image),
                np.min(filtered_image),
                np.max(filtered_image),
                np.median(filtered_image),
            ]
        )

        # 高阶矩
        from scipy import stats

        features.extend([stats.skew(filtered_image.flatten()), stats.kurtosis(filtered_image.flatten())])

        # 零交叉率（检测插值痕迹的重要指标）
        zero_crossings_h = np.sum(np.diff(np.sign(filtered_image), axis=1) != 0)
        zero_crossings_v = np.sum(np.diff(np.sign(filtered_image), axis=0) != 0)
        total_pixels = filtered_image.size

        features.extend([zero_crossings_h / total_pixels, zero_crossings_v / total_pixels])

        return features

    def _extract_block_features(self, image: np.ndarray) -> List[float]:
        """分块提取特征 - 检测局部CFA不一致性"""
        h, w = image.shape
        features = []

        # 分块统计
        block_variances = []
        block_means = []

        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                block = image[i : i + self.block_size, j : j + self.block_size]
                block_variances.append(np.var(block))
                block_means.append(np.mean(block))

        if block_variances:
            features.extend([np.mean(block_variances), np.std(block_variances), np.mean(block_means), np.std(block_means)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return features

    def extract(self, images: List[Union[str, Path, np.ndarray]]) -> np.ndarray:
        """提取CFA特征

        Returns:
            CFA特征向量 (np.ndarray)
        """
        all_features = []
        for image in images:
            img = load_image(image)
            gray = to_grayscale(img)

            features = []

            # 对每种滤波器提取特征
            for filter_name in self.filter_types:
                filtered = self._apply_filter(gray, filter_name)

                # 统计特征
                if self.statistical_analysis:
                    stat_features = self._extract_statistical_features(filtered)
                    features.extend(stat_features)

                # 纹理特征
                if self.texture_analysis:
                    texture_features = self._extract_texture_features(filtered)
                    features.extend(texture_features)

                # 分块特征
                block_features = self._extract_block_features(filtered)
                features.extend(block_features)

            # 周期性特征（只对原图计算一次）
            if self.periodicity_analysis:
                periodicity_features = self._extract_periodicity_features(gray)
                features.extend(periodicity_features)

            # 确保特征向量不包含NaN或无穷大
            features = [float(f) if np.isfinite(f) else 0.0 for f in features]
            all_features.append(features)

        return np.array(all_features)

    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        names = []

        # 每个滤波器的特征名称
        for filter_name in self.filter_types:
            if self.statistical_analysis:
                stat_names = [
                    f"cfa_{filter_name}_mean",
                    f"cfa_{filter_name}_std",
                    f"cfa_{filter_name}_var",
                    f"cfa_{filter_name}_min",
                    f"cfa_{filter_name}_max",
                    f"cfa_{filter_name}_median",
                    f"cfa_{filter_name}_skew",
                    f"cfa_{filter_name}_kurtosis",
                    f"cfa_{filter_name}_zero_cross_h",
                    f"cfa_{filter_name}_zero_cross_v",
                ]
                names.extend(stat_names)

            if self.texture_analysis:
                properties = ["contrast", "correlation", "energy", "homogeneity"]
                for prop in properties:
                    names.extend([f"cfa_{filter_name}_{prop}_mean", f"cfa_{filter_name}_{prop}_std"])

            # 分块特征
            block_names = [
                f"cfa_{filter_name}_block_var_mean",
                f"cfa_{filter_name}_block_var_std",
                f"cfa_{filter_name}_block_mean_mean",
                f"cfa_{filter_name}_block_mean_std",
            ]
            names.extend(block_names)

        # 周期性特征
        if self.periodicity_analysis:
            periodicity_names = [
                "cfa_period_2_h_mean",
                "cfa_period_2_h_std",
                "cfa_period_2_h_max",
                "cfa_period_2_v_mean",
                "cfa_period_2_v_std",
                "cfa_period_2_v_max",
                "cfa_low_freq_energy",
                "cfa_mid_freq_energy",
                "cfa_high_freq_energy",
            ]
            names.extend(periodicity_names)

        return names

    def get_required_dependencies(self) -> List[str]:
        return ["scipy", "scikit-image", "scikit-learn", "numpy"]
