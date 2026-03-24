import torch
import torch.nn as nn
from transformers import CLIPModel
from typing import Optional, Tuple, Dict, Any
import os


class CLIPBinaryClassifier(nn.Module):
    """基于Transformers CLIP的二分类检测器"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        train_image_encoder: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        get_intermediate: bool = False,
    ):
        """
        初始化CLIP二分类器

        Args:
            clip_model: 预加载的CLIP模型
            model_name: CLIP模型名称
            device: 设备
            train_image_encoder: 是否训练image encoder
            dropout_rate: dropout比率
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.model_name = model_name
        self.train_image_encoder = train_image_encoder
        self.hidden_dim = hidden_dim
        self.criterion = nn.CrossEntropyLoss()

        # 加载CLIP模型
        self.clip_model = CLIPModel.from_pretrained(model_name)

        # 获取配置和特征维度
        self.config = self.clip_model.config
        self.feature_dim = self.config.projection_dim

        # 创建分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),  # 二分类
        )

        # 设置参数训练策略
        self._setup_parameter_training()

        self.get_intermediate = get_intermediate

    def _setup_parameter_training(self):
        """设置参数训练策略"""
        # 首先冻结所有CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 确保分类头的所有参数都是可训练的
        for param in self.classifier.parameters():
            param.requires_grad = True

        if self.train_image_encoder:
            # 只解冻image encoder (vision_model)
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = True

        # 可选：只训练最后几层
        # self._unfreeze_last_layers(num_layers=2)

    def _unfreeze_last_layers(self, num_layers: int = 2):
        """解冻vision模型的最后几层"""
        if hasattr(self.clip_model.vision_model, "encoder"):
            encoder_layers = self.clip_model.vision_model.encoder.layers
            # 解冻最后num_layers层
            for layer in encoder_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def get_trainable_parameters(self) -> Dict[str, Any]:
        """获取可训练参数统计信息"""
        stats = {
            "total_params": 0,
            "trainable_params": 0,
            "vision_params": 0,
            "vision_trainable_params": 0,
            "classifier_params": 0,
            "classifier_trainable_params": 0,
            "trainable_layers": [],
        }

        for name, param in self.named_parameters():
            stats["total_params"] += param.numel()

            if param.requires_grad:
                stats["trainable_params"] += param.numel()
                stats["trainable_layers"].append(name)

            # 统计vision部分参数
            if "clip_model.vision_model" in name:
                stats["vision_params"] += param.numel()
                if param.requires_grad:
                    stats["vision_trainable_params"] += param.numel()

            # 统计分类头参数
            if "classifier" in name:
                stats["classifier_params"] += param.numel()
                if param.requires_grad:
                    stats["classifier_trainable_params"] += param.numel()

        stats["trainable_ratio"] = stats["trainable_params"] / stats["total_params"] if stats["total_params"] > 0 else 0
        return stats

    def print_parameter_stats(self):
        """打印参数统计信息"""
        stats = self.get_trainable_parameters()

        print("=" * 50)
        print(f"模型参数统计 - {self.model_name}")
        print("=" * 50)
        print(f"总参数数: {stats['total_params']:,}")
        print(f"可训练参数数: {stats['trainable_params']:,}")
        print(f"可训练参数占比: {stats['trainable_ratio']:.2%}")
        print(f"Vision模型总参数: {stats['vision_params']:,}")
        print(f"Vision模型可训练参数: {stats['vision_trainable_params']:,}")
        print(f"分类头总参数: {stats['classifier_params']:,}")
        print(f"分类头可训练参数: {stats['classifier_trainable_params']:,}")

        if stats["classifier_params"] > 0 and stats["classifier_trainable_params"] == 0:
            print("⚠️  警告：分类头参数存在但不可训练！")
        elif stats["classifier_trainable_params"] > 0:
            print("✅ 分类头参数正常可训练")

        print("\n可训练的层:")
        for layer_name in stats["trainable_layers"][:10]:  # 只显示前10个
            print(f"  - {layer_name}")
        if len(stats["trainable_layers"]) > 10:
            print(f"  ... 还有 {len(stats['trainable_layers']) - 10} 个层")
        print("=" * 50)

    def extract_intermediate_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """提取分类器第一层（ReLU前）的特征"""
        # 使用CLIP提取图像特征
        image_features = self.clip_model.get_image_features(pixel_values)

        # 通过分类器的第一层（Linear + ReLU）
        x = self.classifier[0](image_features)  # Linear layer
        # 不经过ReLU，直接返回线性变换的结果
        return x

    def forward(
        self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """前向传播"""
        # 提取CLIP图像特征
        image_features = self.clip_model.get_image_features(pixel_values)

        # 检查特征是否包含异常值
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            raise ValueError("CLIP图像特征包含NaN或inf值")

        # 通过分类头
        logits = self.classifier(image_features)

        if self.get_intermediate:
            # 提取中间特征（可选）
            with torch.no_grad():
                intermediate_features = self.extract_intermediate_features(pixel_values)
        else:
            intermediate_features = None

        # 计算损失
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return logits, loss, intermediate_features

    def freeze_clip_model(self):
        """冻结CLIP模型"""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def unfreeze_clip_model(self):
        """解冻CLIP模型"""
        for param in self.clip_model.parameters():
            param.requires_grad = True

    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "train_image_encoder": self.train_image_encoder,
            "device": str(self.device),
        }

    def save_model(self, save_directory: str):
        """保存模型"""
        from safetensors.torch import save_file
        import json

        os.makedirs(save_directory, exist_ok=True)

        # 保存模型权重，并在元数据中包含model_name
        metadata = {"model_name": self.model_name}
        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"), metadata=metadata)

        # 保存配置
        config = self.get_config()
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_directory: str, **kwargs):
        """从目录加载模型"""
        import json
        from safetensors.torch import load_file

        # 加载配置
        config_path = os.path.join(model_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            kwargs.update(config)

        # 创建模型实例
        model = cls(**kwargs)

        # 加载权重
        weights_path = os.path.join(model_directory, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict, strict=False)

        return model
