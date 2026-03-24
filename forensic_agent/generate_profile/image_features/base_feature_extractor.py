import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from PIL import Image
from pathlib import Path


class BaseFeatureExtractor(ABC):
    """特征提取器基础类

    定义了特征提取的标准接口和通用方法
    每个具体的特征提取器都应该继承此类
    """

    name: str = "BaseFeature"
    description: str = "Base feature extractor"

    def __init__(self, name: str = None, **kwargs):
        """初始化特征提取器

        Args:
            name: 特征名称，应该是唯一的
            description: 特征描述
            **kwargs: 其他配置参数
        """
        if name:
            self.name = name

        self.config = kwargs
        # 性能统计
        self._extract_count = 0
        self._total_time = 0.0
        self._last_error = None

    def _numpy_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """将numpy数组转换为PIL图像"""
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return Image.fromarray(img_array)
        elif len(img_array.shape) == 2:
            return Image.fromarray(img_array).convert("RGB")
        else:
            raise ValueError(f"不支持的图像形状: {img_array.shape}")

    @abstractmethod
    def extract(self, image: Union[str, Path, np.ndarray]) -> float:
        """提取特征值

        Args:
            image: 输入图像，可以是文件路径或numpy数组

        Returns:
            float: 特征值

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass

    @abstractmethod
    def get_required_dependencies(self) -> list:
        """获取所需的依赖项列表

        Returns:
            list: 依赖项列表，如 ['opencv', 'PIL', 'scipy']
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """获取提取器的统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        avg_time = self._total_time / self._extract_count if self._extract_count > 0 else 0

        return {
            "name": self.name,
            "extract_count": self._extract_count,
            "total_time": self._total_time,
            "average_time": avg_time,
            "last_error": self._last_error,
            "config": self.config,
        }

    def reset_statistics(self):
        """重置统计信息"""
        self._extract_count = 0
        self._total_time = 0.0
        self._last_error = None

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}({self.description})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"
