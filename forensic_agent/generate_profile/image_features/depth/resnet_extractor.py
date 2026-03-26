import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional, Union, List
from pathlib import Path
from loguru import logger
from ..base_feature_extractor import BaseFeatureExtractor


class ResNetExtractor(BaseFeatureExtractor):
    name = "ResNet"
    description = "基于ResNet的深度特征提取"

    def __init__(
        self,
        model_name: str = "resnet50",
        device_id: Optional[int] = None,  # 新增：指定GPU设备ID
        use_pretrained: bool = True,
        feature_layer: str = "avgpool",
        **kwargs,
    ):
        """
        初始化ResNet特征提取器

        Args:
            model_name: ResNet模型名称 (resnet18, resnet34, resnet50, resnet101, resnet152)
            use_pretrained: 是否使用预训练权重
            feature_layer: 提取特征的层名称 (fc, avgpool, layer4等)
        """
        # super().__init__(name=f"ResNet-{model_name}", **kwargs)

        super().__init__(name="ResNet", **kwargs)
        self.model_name = model_name
        self.feature_layer = feature_layer

        # 设备选择逻辑
        if device_id is not None and torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{device_id}")
                print(f"{self.name} 使用GPU设备: cuda:{device_id}")
            else:
                print(f"指定的GPU设备 {device_id} 不存在，回退到CPU")
                self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"{self.name} 使用默认GPU设备")
        else:
            self.device = torch.device("cpu")
            print(f"{self.name} 使用CPU设备")

        # 加载预训练模型
        self.model = self._load_model(model_name, use_pretrained)
        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # 特征存储
        self.features = None
        # 注册特征提取钩子
        self._register_hook()

    def _load_model(self, model_name: str, use_pretrained: bool) -> nn.Module:
        """加载ResNet模型"""
        model_dict = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        if model_name not in model_dict:
            raise ValueError(f"不支持的模型: {model_name}")

        if use_pretrained:
            weights = "IMAGENET1K_V1" if hasattr(models, "ResNet50_Weights") else True
            model = model_dict[model_name](weights=weights)
        else:
            model = model_dict[model_name](weights=None)

        return model

    def _hook_fn(self, module, input, output):
        """钩子函数，改为类方法避免序列化问题"""
        self.features = output.detach()

    def _register_hook(self):
        """注册前向传播钩子函数"""
        # 根据指定层注册钩子
        if self.feature_layer == "fc":
            self.model.fc.register_forward_hook(self._hook_fn)
        elif self.feature_layer == "avgpool":
            self.model.avgpool.register_forward_hook(self._hook_fn)
        elif self.feature_layer == "layer4":
            self.model.layer4.register_forward_hook(self._hook_fn)
        elif self.feature_layer == "layer3":
            self.model.layer3.register_forward_hook(self._hook_fn)
        else:
            # 默认使用avgpool层
            self.model.avgpool.register_forward_hook(self._hook_fn)

    def extract(self, imgs: List[Union[str, Path, np.ndarray]]) -> np.ndarray:
        """提取ResNet特征（支持批处理）"""
        if not isinstance(imgs, list):
            imgs = [imgs]

        # 预处理所有图像
        img_tensors = []
        for img_inputs in imgs:
            # 加载和预处理图像
            if isinstance(img_inputs, (str, Path)):
                pil_img = Image.open(img_inputs).convert("RGB")
            elif isinstance(img_inputs, np.ndarray):
                # 将numpy数组转换为PIL图像
                if img_inputs.dtype != np.uint8:
                    img_inputs = (img_inputs * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_inputs)
            elif isinstance(img_inputs, Image.Image):
                img_inputs = img_inputs.convert("RGB")
            else:
                raise ValueError("不支持的图像格式")

            # 应用预处理变换
            img_tensor = self.transform(pil_img)
            img_tensors.append(img_tensor)

        # 创建批处理张量
        batch_tensor = torch.stack(img_tensors).to(self.device)

        # 批量提取特征
        all_features = self.model(batch_tensor).detach().cpu().numpy()
        return all_features[0]

    def get_required_dependencies(self) -> List[str]:
        return ["torch", "torchvision", "pillow"]

    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        feature_dims = {
            "resnet18": {"fc": 1000, "avgpool": 512, "layer4": 512},
            "resnet34": {"fc": 1000, "avgpool": 512, "layer4": 512},
            "resnet50": {"fc": 1000, "avgpool": 2048, "layer4": 2048},
            "resnet101": {"fc": 1000, "avgpool": 2048, "layer4": 2048},
            "resnet152": {"fc": 1000, "avgpool": 2048, "layer4": 2048},
        }

        return feature_dims.get(self.model_name, {}).get(self.feature_layer, 2048)
