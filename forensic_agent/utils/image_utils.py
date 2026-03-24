from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


def load_image(image: Union[str, Path, np.ndarray], return_pil=False):
    """加载图像为numpy数组

    Args:
        image: 输入图像
        fix_icc_profile: 是否修复有问题的ICC配置文件 (解决libpng iCCP警告)

    Returns:
        np.ndarray: 图像数组 (H, W, C) 或 (H, W)

    Raises:
        ValueError: 图像加载失败
    """
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        if return_pil:
            return image
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            return img_array[:, :, :3]  # 返回RGB通道
        return img_array

    # 从文件路径加载
    image_path = Path(image)
    if not image_path.exists():
        raise ValueError(f"图像文件不存在: {image_path}")

    # 使用PIL/Pillow读取图像(它会正确处理ICC配置文件)
    pil_image = Image.open(str(image_path))
    if return_pil:
        return pil_image

    img_array = np.array(pil_image)
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        return img_array[:, :, :3]  # 返回RGB通道
    return img_array


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """转换为灰度图

    Args:
        image: 输入图像 (H, W, C) 或 (H, W)

    Returns:
        np.ndarray: 灰度图 (H, W)
    """
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB
            # 使用标准的RGB到灰度转换权重
            return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        elif image.shape[2] == 4:  # RGBA
            rgb = image[..., :3]
            return np.dot(rgb, [0.2989, 0.5870, 0.1140])
        else:
            # 单通道或其他情况，取第一个通道
            return image[..., 0]
    else:
        raise ValueError(f"不支持的图像维度: {image.shape}")
