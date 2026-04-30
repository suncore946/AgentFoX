"""Frequency-domain lightweight feature extraction.

中文说明: 这些传统频域统计只作为 VLM 语义分析的辅助上下文, 不加载检测模型权重。
English: These classical frequency statistics are only auxiliary context for
VLM semantic analysis and do not load detector weights.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from PIL import Image
from scipy import fftpack

from ..core.core_exceptions import FeatureExtractionError
from .base_feat import FeatureExtractorBase

try:
    import pywt
except ImportError:
    pywt = None


class FrequencyFeatureExtractor(FeatureExtractorBase):
    """Extract compact FFT/DCT/wavelet cues.

    中文说明: 输出保持固定结构, 方便写入 JSON 并注入 prompt。
    English: Outputs keep a stable structure so they can be written to JSON and
    injected into prompts.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize enabled feature groups.

        中文说明: max_dct_blocks 限制大图计算量, 避免最小 test 运行时内存暴涨。
        English: max_dct_blocks bounds work on large images and prevents memory
        spikes during minimal test runs.
        """
        super().__init__(config)
        config = config or {}
        self.enable_fft = bool(config.get("enable_fft", True))
        self.enable_dct = bool(config.get("enable_dct", True))
        self.enable_wavelet = bool(config.get("enable_wavelet", True))
        self.wavelet_type = config.get("wavelet_type", "db4")
        self.wavelet_levels = int(config.get("wavelet_levels", 3))
        self.dct_block_size = int(config.get("dct_block_size", 8))
        self.max_dct_blocks = int(config.get("max_dct_blocks", 512))
        self.radial_bins = int(config.get("radial_bins", 16))
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate feature parameters.

        中文说明: 配置错误在初始化时暴露, 不等到图片分析中途失败。
        English: Invalid config fails during initialization rather than halfway
        through image analysis.
        """
        if self.enable_wavelet and pywt is None:
            raise ValueError("PyWavelets is required when frequency.enable_wavelet is true.")
        if self.enable_wavelet and self.wavelet_type not in pywt.wavelist():
            raise ValueError(f"Invalid wavelet type: {self.wavelet_type}")
        if self.wavelet_levels < 1:
            raise ValueError("wavelet_levels must be positive.")
        if self.dct_block_size not in {4, 8, 16, 32}:
            raise ValueError("dct_block_size must be one of 4, 8, 16, 32.")
        if self.max_dct_blocks < 1:
            raise ValueError("max_dct_blocks must be positive.")

    def extract_features(self, image: Image.Image, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Extract frequency features from one PIL image.

        中文说明: 所有 numpy 值都会转成 Python float/list, 避免缓存序列化失败。
        English: All numpy values are converted to Python float/list values to
        avoid cache serialization failures.
        """
        try:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            features: Dict[str, Dict[str, Any]] = {}
            if self.enable_fft:
                features["fft"] = self._extract_fft(gray)
            if self.enable_dct:
                features["dct"] = self._extract_dct(gray)
            if self.enable_wavelet:
                features["wavelet"] = self._extract_wavelet(gray)
            return features
        except Exception as exc:
            raise FeatureExtractionError(f"Frequency feature extraction failed: {exc}", feature_type="frequency") from exc

    def _extract_fft(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract FFT spectrum statistics.

        中文说明: 汇总全局功率、低/中/高频能量占比和径向功率曲线。
        English: Summarizes global power, low/mid/high frequency ratios, and a
        radial power curve.
        """
        spectrum = np.fft.fftshift(np.fft.fft2(image))
        power = np.abs(spectrum) ** 2
        total = float(power.sum() + 1e-8)
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        max_radius = float(max(1, min(center_h, center_w)))

        band_ratios: List[float] = []
        previous = 0.0
        for edge in (0.15, 0.35, 1.0):
            mask = (radius >= previous * max_radius) & (radius < edge * max_radius)
            band_ratios.append(float(power[mask].sum() / total))
            previous = edge

        radial_profile = []
        for index in range(self.radial_bins):
            low = index / self.radial_bins * max_radius
            high = (index + 1) / self.radial_bins * max_radius
            mask = (radius >= low) & (radius < high)
            radial_profile.append(float(power[mask].mean()) if np.any(mask) else 0.0)

        log_power = np.log(power + 1e-8)
        return {
            "features": [
                float(log_power.mean()),
                float(log_power.std()),
                float(np.median(log_power)),
                *band_ratios,
                *radial_profile,
            ],
            "meta": {
                "band_order": ["low", "middle", "high"],
                "band_ratios": band_ratios,
                "radial_bins": self.radial_bins,
                "image_shape": [int(h), int(w)],
            },
        }

    def _extract_dct(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract bounded DCT block statistics.

        中文说明: 按固定随机种子采样块, 保证结果可复现且不会遍历超大图片全部块。
        English: Blocks are sampled with a fixed seed for reproducibility and to
        avoid scanning every block in very large images.
        """
        block = self.dct_block_size
        h, w = image.shape
        pad_h = (block - h % block) % block
        pad_w = (block - w % block) % block
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect") if pad_h or pad_w else image
        new_h, new_w = padded.shape
        block_coords = [(i, j) for i in range(0, new_h, block) for j in range(0, new_w, block)]
        if len(block_coords) > self.max_dct_blocks:
            rng = np.random.default_rng(42)
            selected = rng.choice(len(block_coords), size=self.max_dct_blocks, replace=False)
            block_coords = [block_coords[int(idx)] for idx in selected]

        coeffs = []
        for i, j in block_coords:
            patch = padded[i : i + block, j : j + block]
            dct_patch = fftpack.dct(fftpack.dct(patch.T, norm="ortho").T, norm="ortho")
            coeffs.append(dct_patch.reshape(-1))
        coeff_array = np.asarray(coeffs, dtype=np.float32)
        if coeff_array.size == 0:
            return {"features": [0.0] * 8, "meta": {"sampled_blocks": 0, "block_size": block}}

        abs_coeff = np.abs(coeff_array)
        dc = abs_coeff[:, 0]
        low = abs_coeff[:, 1 : min(10, abs_coeff.shape[1])]
        high = abs_coeff[:, min(10, abs_coeff.shape[1]) :]
        total_energy = float((coeff_array**2).sum() + 1e-8)
        low_energy = float((low**2).sum()) if low.size else 0.0
        high_energy = float((high**2).sum()) if high.size else 0.0

        return {
            "features": [
                float(coeff_array.mean()),
                float(coeff_array.std()),
                float(np.median(coeff_array)),
                float((abs_coeff < 1e-6).mean()),
                float(dc.mean()),
                float(low.mean()) if low.size else 0.0,
                float(high.mean()) if high.size else 0.0,
                float(low_energy / total_energy),
                float(high_energy / total_energy),
            ],
            "meta": {
                "sampled_blocks": len(block_coords),
                "block_size": block,
                "total_blocks_after_padding": int((new_h // block) * (new_w // block)),
            },
        }

    def _extract_wavelet(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract compact wavelet subband statistics.

        中文说明: 如果图片尺寸不足, 自动降低分解层数而不是报错退出。
        English: If the image is too small, the decomposition level is reduced
        instead of failing the run.
        """
        max_level = pywt.dwt_max_level(min(image.shape), self.wavelet_type)
        level = max(1, min(self.wavelet_levels, max_level))
        coeffs = pywt.wavedec2(image, self.wavelet_type, level=level)
        subband_stats = []
        energy_values = []

        for index, coeff in enumerate(coeffs):
            if index == 0:
                subbands = [("LL", coeff)]
            else:
                lh, hl, hh = coeff
                subbands = [("LH", lh), ("HL", hl), ("HH", hh)]
            for name, subband in subbands:
                energy = float(np.sum(subband**2))
                energy_values.append(energy)
                subband_stats.extend([float(np.mean(subband)), float(np.std(subband)), energy])

        total = float(sum(energy_values) + 1e-8)
        energy_ratios = [float(value / total) for value in energy_values]
        return {
            "features": [*subband_stats, float(np.mean(energy_ratios)), float(np.std(energy_ratios))],
            "meta": {
                "wavelet_type": self.wavelet_type,
                "levels": int(level),
                "subband_count": len(energy_values),
                "energy_ratios": energy_ratios,
            },
        }
