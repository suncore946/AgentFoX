import numpy as np
import pywt
from scipy import fftpack
from typing import Dict, Any, Tuple, List
import warnings
from PIL import Image
from ..core.core_exceptions import FeatureExtractionError
from .base_feat import FeatureExtractorBase


class FrequencyFeatureExtractor(FeatureExtractorBase):
    """频域特征提取器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化频域特征提取器

        Args:
            config: 频域分析配置
        """
        # 基本配置
        self.enable_fft = config.get("enable_fft", True)
        self.enable_dct = config.get("enable_dct", True)
        self.enable_wavelet = config.get("enable_wavelet", True)

        # 小波配置
        self.wavelet_type = config.get("wavelet_type", "db4")
        self.wavelet_levels = config.get("wavelet_levels", 3)

        # DCT配置
        self.dct_block_size = config.get("dct_block_size", 8)
        self.max_dct_blocks = config.get("max_dct_blocks", 1000)  # 限制DCT块数量

        # FFT配置
        self.radial_bins = config.get("radial_bins", 20)  # 固定径向bin数量

        # 噪声鲁棒性配置
        self.noise_robustness = config.get("noise_robustness", None)

        # 参数验证
        self._validate_config()

    def _validate_config(self):
        """验证配置参数"""
        # 验证小波类型
        if self.wavelet_type not in pywt.wavelist():
            raise ValueError(f"Invalid wavelet type: {self.wavelet_type}")

        # 验证小波级数
        if not isinstance(self.wavelet_levels, int) or self.wavelet_levels < 1:
            raise ValueError(f"Wavelet levels must be positive integer, got: {self.wavelet_levels}")

        # 验证DCT块大小
        if self.dct_block_size not in [4, 8, 16, 32]:
            warnings.warn(f"DCT block size {self.dct_block_size} is uncommon, recommended: 8")

    def extract_features(self, image: Image.Image, *args, **kwargs) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """
        提取频域特征

        Args:
            image: 图像数组 (H, W) 或 (H, W, C)

        Returns:
            Tuple[特征向量字典, 元数据]
        """
        try:
            image = np.array(image.convert("L"))  # 转换为灰度图像

            # 预处理图像
            features = {}

            # FFT特征
            if self.enable_fft:
                fft_features, fft_meta = self._extract_fft_features(image)
                features["fft"] = {
                    "features": fft_features,
                    "meta": fft_meta,
                }

            # DCT特征
            if self.enable_dct:
                dct_features, dct_meta = self._extract_dct_features(image)
                features["dct"] = {
                    "features": dct_features,
                    "meta": dct_meta,
                }

            # 小波特征
            if self.enable_wavelet:
                wavelet_features, wavelet_meta = self._extract_wavelet_features(image)
                features["wavelet"] = {
                    "features": wavelet_features,
                    "meta": wavelet_meta,
                }

            # 噪声鲁棒性测试
            if self.noise_robustness:
                robustness_features, robustness_meta = self._extract_robustness_features(image)
                features["robustness"] = {"features": robustness_features, "meta": robustness_meta}

            return features

        except Exception as e:
            raise FeatureExtractionError(f"频域特征提取失败: {e}")

    def _extract_fft_features(self, image) -> Tuple[List[float], Dict[str, Any]]:
        """提取FFT特征"""
        # 2D FFT
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)

        # 计算功率谱
        power_spectrum = magnitude_spectrum**2
        log_power_spectrum = np.log(power_spectrum + 1e-8)

        features = []

        # 基本统计特征
        features.extend(
            [
                float(np.mean(log_power_spectrum)),
                float(np.std(log_power_spectrum)),
                float(np.median(log_power_spectrum)),
                float(np.max(log_power_spectrum)),
                float(np.min(log_power_spectrum)),
                float(np.percentile(log_power_spectrum, 25)),
                float(np.percentile(log_power_spectrum, 75)),
                float(np.percentile(log_power_spectrum, 90)),
            ]
        )

        # 高低频能量比
        h, w = image.shape
        center_h, center_w = h // 2, w // 2

        # 定义多个频率带
        total_energy = float(np.sum(power_spectrum))
        frequency_bands = []

        for radius_ratio in [0.1, 0.25, 0.5]:  # 不同半径的圆形区域
            radius = int(min(h, w) * radius_ratio / 2)
            y, x = np.ogrid[:h, :w]
            mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius**2
            band_energy = float(np.sum(power_spectrum * mask))
            frequency_bands.append(float(band_energy / total_energy))

        features.extend(frequency_bands)

        # 方向性分析 - 计算不同方向的能量
        angles = [0, 45, 90, 135]  # 度
        directional_energies = []

        for angle in angles:
            # 创建方向掩码
            angle_rad = np.radians(angle)
            y, x = np.ogrid[:h, :w]
            x_centered = x - center_w
            y_centered = y - center_h

            # 计算每个像素相对于中心的角度
            pixel_angles = np.arctan2(y_centered, x_centered)
            # 创建角度带掩码 (±22.5度范围)
            angle_width = np.radians(22.5)
            mask = np.abs(pixel_angles - angle_rad) <= angle_width
            # 处理角度环绕
            mask |= np.abs(pixel_angles - angle_rad + 2 * np.pi) <= angle_width
            mask |= np.abs(pixel_angles - angle_rad - 2 * np.pi) <= angle_width

            directional_energy = float(np.sum(power_spectrum * mask))
            directional_energies.append(float(directional_energy / total_energy))

        features.extend(directional_energies)

        # 径向功率谱 - 固定bin数量
        radial_profile = self._get_radial_profile(power_spectrum, center_h, center_w)
        features.extend(radial_profile)

        # 频谱熵
        normalized_spectrum = power_spectrum / total_energy
        spectral_entropy = float(-np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-8)))
        features.append(float(spectral_entropy))

        metadata = {
            "total_energy": float(total_energy),
            "spectral_entropy": float(spectral_entropy),
            "frequency_bands": frequency_bands,
            "directional_energies": directional_energies,
            "image_shape": list(image.shape),
        }

        return features, metadata

    def _extract_dct_features(self, image) -> Tuple[List[float], Dict[str, Any]]:
        """提取DCT特征"""
        h, w = image.shape
        block_size = self.dct_block_size

        # 填充图像到block_size的倍数
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size

        if pad_h > 0 or pad_w > 0:
            padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded_image = image

        new_h, new_w = padded_image.shape

        # 限制处理的块数量以控制内存和计算时间
        total_blocks = (new_h // block_size) * (new_w // block_size)
        if total_blocks > self.max_dct_blocks:
            # 随机采样块
            np.random.seed(42)  # 确保可重现性
            sample_indices = np.random.choice(total_blocks, self.max_dct_blocks, replace=False)
            sample_indices = set(sample_indices)
        else:
            sample_indices = None

        dct_coeffs = []
        block_idx = 0

        for i in range(0, new_h, block_size):
            for j in range(0, new_w, block_size):
                if sample_indices is None or block_idx in sample_indices:
                    block = padded_image[i : i + block_size, j : j + block_size]
                    dct_block = fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")
                    dct_coeffs.append(dct_block.flatten())
                block_idx += 1

        if not dct_coeffs:
            return [0.0] * 25, {"error": "No DCT blocks extracted"}

        dct_coeffs = np.array(dct_coeffs)
        features = []

        # DCT系数统计
        features.extend(
            [
                float(np.mean(dct_coeffs)),
                float(np.std(dct_coeffs)),
                float(np.median(dct_coeffs)),
                float(np.percentile(dct_coeffs, 25)),
                float(np.percentile(dct_coeffs, 75)),
                float(np.percentile(dct_coeffs, 10)),
                float(np.percentile(dct_coeffs, 90)),
            ]
        )

        # 零系数比例和稀疏性分析
        zero_coeff_ratio = np.sum(np.abs(dct_coeffs) < 1e-6) / dct_coeffs.size
        near_zero_ratio = np.sum(np.abs(dct_coeffs) < 1e-3) / dct_coeffs.size
        features.extend([float(zero_coeff_ratio), float(near_zero_ratio)])

        # 基于位置的系数分析
        # DC分量 (0,0位置)
        dc_coeffs = dct_coeffs[:, 0]

        # 低频AC分量 (位置1-9)
        low_freq_indices = [1, 8, 16, 9, 2, 3, 10, 17, 24][: min(9, dct_coeffs.shape[1] - 1)]
        low_freq_coeffs = dct_coeffs[:, low_freq_indices] if len(low_freq_indices) > 0 else dct_coeffs[:, 1 : min(10, dct_coeffs.shape[1])]

        # 高频系数 (剩余位置)
        high_freq_start = min(10, dct_coeffs.shape[1])
        high_freq_coeffs = dct_coeffs[:, high_freq_start:] if high_freq_start < dct_coeffs.shape[1] else np.array([[0]])

        features.extend(
            [
                float(np.mean(np.abs(dc_coeffs))),
                float(np.std(dc_coeffs)),
                float(np.mean(np.abs(low_freq_coeffs))),
                float(np.std(low_freq_coeffs)),
                float(np.mean(np.abs(high_freq_coeffs))) if high_freq_coeffs.size > 0 else 0.0,
                float(np.std(high_freq_coeffs)) if high_freq_coeffs.size > 0 else 0.0,
            ]
        )

        # 能量集中度分析
        total_energy = float(np.sum(dct_coeffs**2))
        dc_energy = float(np.sum(dc_coeffs**2))
        low_freq_energy = float(np.sum(low_freq_coeffs**2))
        high_freq_energy = float(np.sum(high_freq_coeffs**2)) if high_freq_coeffs.size > 0 else 0.0

        features.extend(
            [
                float(dc_energy / (total_energy + 1e-8)),
                float(low_freq_energy / (total_energy + 1e-8)),
                float(high_freq_energy / (total_energy + 1e-8)),
                float((dc_energy + low_freq_energy) / (total_energy + 1e-8)),  # 综合低频能量比
            ]
        )

        # 系数分布特征
        coeff_hist, _ = np.histogram(dct_coeffs.flatten(), bins=50, density=True)
        coeff_entropy = float(-np.sum(coeff_hist * np.log2(coeff_hist + 1e-8)))
        features.append(float(coeff_entropy))

        metadata = {
            "zero_coeff_ratio": float(zero_coeff_ratio),
            "near_zero_ratio": float(near_zero_ratio),
            "total_blocks": len(dct_coeffs),
            "block_size": block_size,
            "coeff_entropy": float(coeff_entropy),
            "energy_distribution": {
                "dc_ratio": float(dc_energy / (total_energy + 1e-8)),
                "low_freq_ratio": float(low_freq_energy / (total_energy + 1e-8)),
                "high_freq_ratio": float(high_freq_energy / (total_energy + 1e-8)),
            },
        }

        return features, metadata

    def _extract_wavelet_features(self, image) -> Tuple[List[float], Dict[str, Any]]:
        """提取小波特征"""
        features = []

        try:
            # 多级小波分解
            coeffs = pywt.wavedec2(image, self.wavelet_type, level=self.wavelet_levels)
        except Exception as e:
            # 如果分解失败（图像太小），降低分解级数
            max_level = pywt.dwt_max_level(min(image.shape), self.wavelet_type)
            actual_levels = min(self.wavelet_levels, max_level)
            if actual_levels < 1:
                return [0.0] * 20, {"error": f"Image too small for wavelet decomposition: {list(image.shape)}"}
            coeffs = pywt.wavedec2(image, self.wavelet_type, level=actual_levels)

        # 分析每一级的子带
        level_energies = []
        level_entropies = []
        level_stats = []

        for level, coeff_level in enumerate(coeffs):
            if level == 0:
                # 第0级是近似系数(LL)
                ll = coeff_level
                energy = float(np.sum(ll**2))
                entropy = self._calculate_entropy(ll)

                stats = [
                    float(np.mean(ll)),
                    float(np.std(ll)),
                    float(np.median(ll)),
                    float(np.percentile(ll, 25)),
                    float(np.percentile(ll, 75)),
                ]

                features.extend(stats)
                features.extend([float(energy), float(entropy)])

                level_energies.append(float(energy))
                level_entropies.append(float(entropy))
                level_stats.append({"level": level, "type": "LL", "stats": stats})

            else:
                # 其他级是细节系数(LH, HL, HH)
                lh, hl, hh = coeff_level

                for subband, name in [(lh, "LH"), (hl, "HL"), (hh, "HH")]:
                    energy = float(np.sum(subband**2))
                    entropy = self._calculate_entropy(subband)

                    stats = [
                        float(np.mean(subband)),
                        float(np.std(subband)),
                    ]

                    features.extend(stats)
                    features.extend([float(energy), float(entropy)])

                    level_energies.append(float(energy))
                    level_entropies.append(float(entropy))
                    level_stats.append({"level": level, "type": name, "stats": stats})

        # 全局特征
        total_energy = float(sum(level_energies))
        mean_entropy = float(np.mean(level_entropies))
        std_entropy = float(np.std(level_entropies))

        # 能量分布特征
        energy_ratios = [float(e / (total_energy + 1e-8)) for e in level_energies]

        features.extend(
            [
                total_energy,
                mean_entropy,
                std_entropy,
                float(np.max(energy_ratios)),
                float(np.min(energy_ratios)),
                float(np.std(energy_ratios)),
            ]
        )

        # 跨尺度相关性分析
        if len(level_energies) > 1:
            energy_correlation = float(np.corrcoef(level_energies[:-1], level_energies[1:])[0, 1]) if len(level_energies) > 2 else 0.0
        else:
            energy_correlation = 0.0

        features.append(energy_correlation)

        metadata = {
            "wavelet_type": self.wavelet_type,
            "levels": len(coeffs) - 1,  # 实际使用的级数
            "total_energy": float(total_energy),
            "level_energies": [float(e) for e in level_energies],
            "level_entropies": [float(e) for e in level_entropies],
            "energy_correlation": float(energy_correlation),
            "level_stats": level_stats,
        }

        return features, metadata

    def _extract_robustness_features(self, image) -> Tuple[List[float], Dict[str, Any]]:
        """提取噪声鲁棒性特征"""
        noise_config = self.noise_robustness
        noise_types = noise_config.get("noise_types", ["gaussian"])
        intensity_range = noise_config.get("intensity_range", [0.01, 0.05, 0.1])

        # 计算原始图像的频域特征
        original_fft = np.fft.fft2(image)
        original_energy = float(np.sum(np.abs(original_fft) ** 2))

        # 计算原始图像的功率谱熵
        original_power = np.abs(original_fft) ** 2
        original_power_norm = original_power / np.sum(original_power)
        original_entropy = float(-np.sum(original_power_norm * np.log2(original_power_norm + 1e-8)))

        robustness_scores = []
        energy_changes = []
        entropy_changes = []

        for noise_type in noise_types:
            for intensity in intensity_range:
                try:
                    # 添加噪声
                    if noise_type == "gaussian":
                        noise = np.random.normal(0, intensity, image.shape)
                        noisy_image = np.clip(image + noise, 0, 1)
                    elif noise_type == "salt_pepper":
                        noisy_image = image.copy()
                        num_pixels = int(intensity * image.size)
                        if num_pixels > 0:
                            coords = (np.random.randint(0, image.shape[0], num_pixels), np.random.randint(0, image.shape[1], num_pixels))
                            noisy_image[coords] = np.random.choice([0, 1], num_pixels)
                    elif noise_type == "uniform":
                        noise = np.random.uniform(-intensity, intensity, image.shape)
                        noisy_image = np.clip(image + noise, 0, 1)
                    else:
                        continue

                    # 计算加噪后的频域特征
                    noisy_fft = np.fft.fft2(noisy_image)
                    noisy_energy = float(np.sum(np.abs(noisy_fft) ** 2))

                    noisy_power = np.abs(noisy_fft) ** 2
                    noisy_power_norm = noisy_power / np.sum(noisy_power)
                    noisy_entropy = float(-np.sum(noisy_power_norm * np.log2(noisy_power_norm + 1e-8)))

                    # 计算变化量
                    energy_change = abs(noisy_energy - original_energy) / (original_energy + 1e-8)
                    entropy_change = abs(noisy_entropy - original_entropy) / (original_entropy + 1e-8)

                    # 计算鲁棒性得分 (变化越小，鲁棒性越好)
                    energy_robustness = 1.0 / (1.0 + energy_change)
                    entropy_robustness = 1.0 / (1.0 + entropy_change)
                    combined_robustness = (energy_robustness + entropy_robustness) / 2.0

                    robustness_scores.append(float(combined_robustness))
                    energy_changes.append(float(energy_change))
                    entropy_changes.append(float(entropy_change))

                except Exception as e:
                    # 如果某种噪声类型失败，跳过
                    continue

        if not robustness_scores:
            # 如果所有测试都失败了，返回默认值
            features = [0.5, 0.2, 0.0, 1.0, 0.1, 0.05, 0.0, 0.2]
        else:
            features = [
                float(np.mean(robustness_scores)),
                float(np.std(robustness_scores)),
                float(np.min(robustness_scores)),
                float(np.max(robustness_scores)),
                float(np.mean(energy_changes)),
                float(np.std(energy_changes)),
                float(np.mean(entropy_changes)),
                float(np.std(entropy_changes)),
            ]

        metadata = {
            "noise_types": noise_types,
            "intensity_range": intensity_range,
            "robustness_scores": [float(s) for s in robustness_scores],
            "energy_changes": [float(c) for c in energy_changes],
            "entropy_changes": [float(c) for c in entropy_changes],
            "original_energy": float(original_energy),
            "original_entropy": float(original_entropy),
        }

        return features, metadata

    def _get_radial_profile(self, image: np.ndarray, center_h: int, center_w: int) -> List[float]:
        """计算径向功率谱 - 固定bin数量"""
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

        max_radius = min(center_h, center_w)
        if max_radius < self.radial_bins:
            # 如果图像太小，使用较少的bin
            actual_bins = max_radius
        else:
            actual_bins = self.radial_bins

        radial_profile = []
        bin_size = float(max_radius / actual_bins) if actual_bins > 0 else 1.0

        for i in range(actual_bins):
            r_min = float(i * bin_size)
            r_max = float((i + 1) * bin_size)
            mask = (r >= r_min) & (r < r_max)

            if np.sum(mask) > 0:
                radial_profile.append(float(np.mean(image[mask])))
            else:
                radial_profile.append(0.0)

        # 补齐到固定长度
        while len(radial_profile) < self.radial_bins:
            radial_profile.append(0.0)

        return radial_profile[: self.radial_bins]

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """计算熵"""
        data = data.flatten()

        # 使用自适应bin数量
        n_bins = min(50, max(10, len(data) // 100))

        try:
            hist, _ = np.histogram(data, bins=n_bins, density=True)
            hist = hist[hist > 0]  # 移除零值

            if len(hist) == 0:
                return 0.0

            entropy = float(-np.sum(hist * np.log2(hist + 1e-8)))
            return entropy
        except Exception:
            return 0.0
