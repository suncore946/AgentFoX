"""
多模态特征提取器服务 - 完整版本
支持多模态特征提取：空域、频域、小波、元数据和语义特征
提供类型安全和异步支持，集成指纹提取功能
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List, Union
from PIL import Image
from loguru import logger


from ..core.core_exceptions import FeatureExtractionError

from ..expert_features.spatial_feature import SpatialFeatureExtractor
from ..expert_features.frequency_feature import FrequencyFeatureExtractor
from ..expert_features.metadata_feature import MetadataExtractor
from ..manager.image_manager import ImageManager
from .base_manager import BaseManager

from ..processor.semantic_forgery_tracking_processor import SemanticForgeryTrackingProcessor
from ..utils import create_chat_llm


class FeatureManager(BaseManager):

    # 支持的特征类型常量
    SUPPORTED_FEATURES = ["spatial", "frequency", "semantic"]

    def __init__(self, config: Dict[str, Any], image_manager: ImageManager, semantic_llm=None) -> None:
        self.config = config
        self.logger = logger
        self.image_manager = image_manager

        # 特征开关配置
        self.enable_metadata = config.get("enable_metadata", True)
        self.enable_spatial = config.get("enable_spatial", True)
        self.enable_frequency = config.get("enable_frequency", True)

        self.metadata_extractor = MetadataExtractor()
        self.spatial_extractor = SpatialFeatureExtractor(config["spatial"])
        self.frequency_extractor = FrequencyFeatureExtractor(config["frequency"])

        # 初始化ChatOpenAI
        semantic_config = config.get("semantic_llm", {})
        llm_config = semantic_config.get("llm_config", {})
        if llm_config:
            self.semantic_llm = create_chat_llm(llm_config)
        elif semantic_llm:
            self.semantic_llm = semantic_llm
        else:
            raise ValueError("Either llm_config or semantic_llm must be provided.")

        # 获取伪造痕迹的内容并进行分析
        self.forensic_traces = SemanticForgeryTrackingProcessor(prompt_path=semantic_config["prompt_path"], forensic_llm=self.semantic_llm)

    def run(self, image_path: Union[str, Path], *args, **kwargs):
        """
        Args:
            image_path: 图像路径
            feature_types: 需要提取的特征类型列表

        Returns:
            特征提取结果
        """
        image_base64, image, image_format = self.image_manager.get_base64(image_path, is_resize=False)

        image_metadata = self.metadata_extractor.extract_features(image_path=image_path, image=image)
        spatial_features = self.spatial_extractor.extract_features(image=image)
        frequency_features = self.frequency_extractor.extract_features(image=image)
        ret = self.forensic_traces.run(
            **{
                "image": image,
                "image_path": image_path,
                "image_base64": image_base64,
                "image_format": image_format,
                "image_features": {
                    "metadata": image_metadata,
                    "spatial": spatial_features,
                    "frequency": frequency_features,
                },
            }
        )
        if "error" in ret:
            raise FeatureExtractionError(ret["error"])
        return ret, image

    def _aggregate_features(self, features: Dict[str, Any]) -> np.ndarray:
        """聚合所有特征为固定维度向量"""
        try:
            feature_vectors = []

            for feature_type, feature_data in features.items():
                if isinstance(feature_data, dict):
                    vector = self._dict_to_vector(feature_data, feature_type)
                    feature_vectors.append(vector)
                elif isinstance(feature_data, np.ndarray):
                    feature_vectors.append(feature_data.flatten())
                elif isinstance(feature_data, (list, tuple)):
                    feature_vectors.append(np.array(feature_data))

            if feature_vectors:
                combined = np.concatenate(feature_vectors)

                if len(combined) > self.fingerprint_dim:
                    combined = combined[: self.fingerprint_dim]
                elif len(combined) < self.fingerprint_dim:
                    padding = np.zeros(self.fingerprint_dim - len(combined))
                    combined = np.concatenate([combined, padding])

                return combined
            else:
                return np.zeros(self.fingerprint_dim)

        except Exception:
            return np.random.normal(0, 0.1, self.fingerprint_dim)

    def _dict_to_vector(self, feature_dict: Dict[str, Any], feature_type: str) -> np.ndarray:
        """将字典特征转换为向量"""
        try:
            vector_parts = []

            if feature_type == "metadata":
                vector_parts.extend(
                    [
                        feature_dict.get("file_size", 0) / 1000000,
                        feature_dict.get("width", 0) / 1000,
                        feature_dict.get("height", 0) / 1000,
                        float(feature_dict.get("has_exif", 0)),
                        feature_dict.get("compression_ratio", 0),
                    ]
                )
            elif feature_type == "spatial":
                vector_parts.extend(
                    [
                        feature_dict.get("noise_variance", 0),
                        feature_dict.get("edge_density", 0),
                        feature_dict.get("texture_energy", 0),
                        feature_dict.get("contrast", 0),
                        feature_dict.get("sharpness", 0),
                    ]
                )
            elif feature_type == "frequency":
                vector_parts.extend(
                    [
                        feature_dict.get("high_freq_energy", 0),
                        feature_dict.get("low_freq_energy", 0),
                        feature_dict.get("freq_entropy", 0),
                        feature_dict.get("dct_variance", 0),
                    ]
                )
            elif feature_type == "wavelet":
                coeffs = feature_dict.get("coefficients", [])
                if coeffs:
                    vector_parts.extend(coeffs[:10])
                else:
                    vector_parts.extend([0] * 10)
            elif feature_type == "vlm":
                vector_parts.extend(
                    [
                        feature_dict.get("semantic_anomaly_score", 0),
                        feature_dict.get("style_consistency", 0),
                        feature_dict.get("object_coherence", 0),
                    ]
                )

            return np.array(vector_parts, dtype=np.float32)

        except Exception:
            return np.zeros(10, dtype=np.float32)

    def _generate_type_hint(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """基于特征生成类型提示"""
        type_hint = {"suspected_type": "unknown", "confidence": 0.5, "indicators": []}

        try:
            # 分析频域特征
            freq_features = features.get("frequency", {})
            if isinstance(freq_features, np.ndarray) and len(freq_features) > 7:
                high_freq_ratio = freq_features[7]  # 高频/低频能量比
            else:
                high_freq_ratio = freq_features.get("high_freq_energy", 0.5) if isinstance(freq_features, dict) else 0.5

            if high_freq_ratio < 0.3:
                type_hint["suspected_type"] = "diffusion"
                type_hint["confidence"] = 0.7
                type_hint["indicators"].append("low_high_freq_energy")
            elif high_freq_ratio > 0.7:
                type_hint["suspected_type"] = "real"
                type_hint["confidence"] = 0.6
                type_hint["indicators"].append("high_freq_energy")

            # 分析VLM特征
            vlm_features = features.get("vlm", {})
            semantic_anomaly = vlm_features.get("has_semantic_anomaly", False)

            if semantic_anomaly:
                type_hint["suspected_type"] = "gan"
                type_hint["confidence"] = max(type_hint["confidence"], 0.8)
                type_hint["indicators"].append("semantic_anomaly")

            return type_hint

        except Exception:
            return type_hint

    def load_image(self, image_info: Union[str, Path, Image.Image, np.ndarray], mode: str = "grayscale") -> np.ndarray:
        """
        加载图像，支持路径、PIL.Image、np.ndarray输入

        Args:
            image_info: 图像路径、PIL.Image对象或np.ndarray
            mode: 加载模式 ('grayscale', 'rgb', 'bgr')

        Returns:
            加载的图像数组

        Raises:
            FeatureExtractionError: 图像加载失败
        """
        try:
            # 路径类型
            if isinstance(image_info, (str, Path)):
                if mode == "grayscale":
                    image = cv2.imread(str(image_info), cv2.IMREAD_GRAYSCALE)
                elif mode == "rgb":
                    image = cv2.imread(str(image_info), cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
                elif mode == "bgr":
                    image = cv2.imread(str(image_info), cv2.IMREAD_COLOR)
                else:
                    raise ValueError(f"不支持的图像模式: {mode}")
                if image is None:
                    raise FeatureExtractionError(f"无法加载图像: {image_info}")
            # PIL.Image类型
            elif isinstance(image_info, Image.Image):
                if mode == "grayscale":
                    image = np.array(image_info.convert("L"))
                elif mode == "rgb":
                    image = np.array(image_info.convert("RGB"))
                elif mode == "bgr":
                    image = np.array(image_info.convert("RGB"))[..., ::-1]
                else:
                    raise ValueError(f"不支持的图像模式: {mode}")
            # ndarray类型
            elif isinstance(image_info, np.ndarray):
                if mode == "grayscale":
                    if image_info.ndim == 3:
                        image = cv2.cvtColor(image_info, cv2.COLOR_BGR2GRAY)
                    else:
                        image = image_info
                elif mode == "rgb":
                    if image_info.ndim == 2:
                        image = cv2.cvtColor(image_info, cv2.COLOR_GRAY2RGB)
                    elif image_info.shape[2] == 3:
                        image = cv2.cvtColor(image_info, cv2.COLOR_BGR2RGB)
                    else:
                        image = image_info
                elif mode == "bgr":
                    if image_info.ndim == 2:
                        image = cv2.cvtColor(image_info, cv2.COLOR_GRAY2BGR)
                    else:
                        image = image_info
                else:
                    raise ValueError(f"不支持的图像模式: {mode}")
            else:
                raise FeatureExtractionError("不支持的输入类型")
            if not (isinstance(image, np.ndarray) and image.size > 0):
                raise FeatureExtractionError("图像数据无效")
            return image
        except Exception as e:
            raise FeatureExtractionError(f"图像加载失败: {e}")
