import json
import numpy as np
import torch
from PIL import Image
from typing import Union, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import os
from transformers import CLIPProcessor, CLIPModel
from safetensors.torch import load_file
from ..base_feature_extractor import BaseFeatureExtractor
from .clip_model import CLIPBinaryClassifier


class CLIPExtractor(BaseFeatureExtractor):
    """CLIP图像特征提取器
    使用Transformers库的CLIP模型提取图像特征
    """

    description = "Extracts intermediate layer feature vectors from a CLIP vision encoder retrained on AIGC and authentic images. The model processes images through a Vision Transformer architecture, capturing the penultimate layer's hidden states as deep feature representations that encapsulate rich visual semantic information"

    name = "CLIP"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        model_path: Optional[str] = None,
        device_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_path is None:
            model_path = "outputs/retrain_model/clip_classifier/best_model_epoch_1_acc_94.52915528604004.pth"
        assert os.path.isfile(model_path), f"模型文件不存在: {model_path}"
        self.model_path = model_path
        self.model_name = model_name
        self.device = torch.device(f"cuda:{device_id}" if device_id is not None and torch.cuda.is_available() else "cpu")
        # 加载模型和处理器
        self._processor = None
        self._model = None

    @property
    def processor(self) -> CLIPProcessor:
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> CLIPBinaryClassifier:
        if self._model is None:
            self._model = CLIPBinaryClassifier(model_name=self.model_name, get_intermediate=True)
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
            logger.info(f"Model weights loaded from {self.model_path}")
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._model.to(self.device)
        return self._model

    def _preprocess_images(self, imgs: List[Union[str, Path, np.ndarray, Image.Image]]) -> torch.Tensor:
        """预处理图像"""
        processed_images = []

        for img_input in imgs:
            if isinstance(img_input, (str, Path)):
                img = Image.open(img_input).convert("RGB")
            elif isinstance(img_input, Image.Image):
                img = img_input.convert("RGB")
            elif isinstance(img_input, np.ndarray):
                img = self._numpy_to_pil(img_input)
            else:
                raise ValueError(f"不支持的图像格式: {type(img_input)}")

            processed_images.append(img)

        inputs = self.processor(images=processed_images, return_tensors="pt", padding=True)
        return inputs.pixel_values

    def extract(self, imgs: Union[List[Union[str, Path, np.ndarray]], str, Path, np.ndarray]) -> np.ndarray:
        """提取CLIP图像特征"""
        # 确保输入是列表
        if not isinstance(imgs, list):
            imgs = [imgs]

        # 预处理图像
        pixel_values = self._preprocess_images(imgs)
        pixel_values = pixel_values.to(self.device)

        # 提取特征
        with torch.no_grad():
            intermediate_features = self.model.extract_intermediate_features(pixel_values)

        # 转换为numpy数组并返回
        features = intermediate_features.detach().cpu().numpy()
        return features

    def get_required_dependencies(self) -> List[str]:
        return ["torch", "torchvision", "transformers", "pillow", "safetensors"]
