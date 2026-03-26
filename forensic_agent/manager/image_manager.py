import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
from loguru import logger
import numpy as np
import requests
from PIL import Image, ImageOps

from .base_manager import BaseManager


class ImageManager(BaseManager):

    def __init__(self, config: dict):
        """
        初始化ImageProcessor类。
        """
        self.max_width = config.get("max_width", 128)
        self.max_height = config.get("max_height", 128)
        self.font_path = config.get("font_path", "./resource/front/MSYH.TTC")
        self.font_size = config.get("font_size", 40)
        # 新增变量：是否等比缩放
        self.maintain_aspect_ratio = config.get("maintain_aspect_ratio", True)

    def reduce_image(self, image, target_width=None, target_height=None):
        # 获取原始图像的宽度和高度
        original_width, original_height = image.size

        # 确定目标尺寸：优先使用传入的参数，其次使用实例默认值
        max_width = target_width if target_width is not None else self.max_width
        max_height = target_height if target_height is not None else self.max_height

        if max_width is None and max_height is None:
            return image, 1

        # 非等比缩放
        if not self.maintain_aspect_ratio:
            new_width = max_width or original_width
            new_height = max_height or original_height
            width_ratio = new_width / original_width
            height_ratio = new_height / original_height
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # 返回宽、高缩放比例
            return resized_image, (width_ratio, height_ratio)

        # 等比缩放：如果原始图像已经符合要求，无需缩放
        if (max_width is None or original_width <= max_width) and (max_height is None or original_height <= max_height):
            return image, 1

        # 计算宽度和高度的缩放比例
        width_ratio = max_width / original_width if max_width is not None else float("inf")
        height_ratio = max_height / original_height if max_height is not None else float("inf")

        # 使用较小的比例进行缩放，保证宽度和高度都不超过最大值
        scale_ratio = min(width_ratio, height_ratio)

        # 计算新的宽度和高度
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # 使用LANCZOS进行高质量缩放
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return resized_image, scale_ratio

    def get_base64(
        self,
        src_image,
        is_resize=True,
        is_center_crop=False,
        target_width=None,
        target_height=None,
    ):
        if isinstance(src_image, str) or isinstance(src_image, Path):
            loaded_image = self.load_image(src_image)[0]
        elif isinstance(src_image, np.ndarray):
            loaded_image = Image.fromarray(src_image)
        else:
            loaded_image = src_image

        format_info = loaded_image.format
        if format_info is None:
            raise ValueError("无法识别图像格式，请确保输入的图像具有有效的格式信息。")

        if is_resize:
            trans_image, _ = self.reduce_image(loaded_image, target_width, target_height)
            logger.debug(f"Resized image to: {trans_image.size}")
        else:
            trans_image = loaded_image

        if is_center_crop and target_width and target_height:
            # 使用 ImageOps.fit 进行中心裁剪
            trans_image = ImageOps.fit(trans_image, (target_width, target_height), method=Image.Resampling.LANCZOS)
            logger.debug(f"Center cropped image to: {trans_image.size}")

        with BytesIO() as buffered:
            trans_image.save(buffered, format=format_info)
            trans_base64 = base64.b64encode(buffered.getvalue()).decode()

        return trans_base64, trans_image, format_info

    def base64_to_image(self, b64_str: str) -> Image.Image:
        """
        将 Base64 字符串解码为 PIL Image
        """
        b64_str = b64_str.split(",")[-1]  # 去掉前缀部分
        data = base64.b64decode(b64_str)
        return Image.open(BytesIO(data))

    def load_image(self, image_file: Path, image_type="RGB"):
        """
        加载并预处理图片
        """
        image_file = Path(image_file)
        if self.is_url(str(image_file)):
            # 将图片下载到本地, 后进行处理
            src_image = Image.open(requests.get(image_file, stream=True).raw).convert(image_type)
        else:
            src_image = Image.open(image_file)
        trans_image, scale_ratio = self.reduce_image(src_image)
        return src_image, trans_image, scale_ratio

    def process_mask(self, src_mask, target_mask, scale_ratio):
        def read_mask(mask):
            if isinstance(mask, (str, Path)):
                return self.load_image(mask)[0]
            return mask

        # 读取mask图像
        src_mask_img = read_mask(src_mask)
        target_mask_img = read_mask(target_mask)

        # 确保图像大小相同
        if src_mask_img.shape != target_mask_img.shape:
            raise ValueError("Source and target masks must have the same dimensions.")

        # 合并target_mask和target_mask_img, 要求相同的像素保留, 不同的像素取大值
        combined_mask = np.maximum(src_mask_img, target_mask_img)

        # 如果需要缩放比例，可以在此处应用缩放
        if scale_ratio != 1.0:
            new_size = (int(combined_mask.shape[1] * scale_ratio), int(combined_mask.shape[0] * scale_ratio))
            combined_mask = cv2.resize(combined_mask, new_size, interpolation=cv2.INTER_NEAREST)

        # 转为PIL图像
        combined_mask = Image.fromarray(combined_mask)
        return combined_mask

    @staticmethod
    def is_url(path: str):
        """判断路径是否为 URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def combine_images(src_img, mask_img):
        # mask_img是灰度图像, src_img是RGB图像
        # mask_img中为白色区域的保留src_img内容,黑色区域的去除src_img内容
        src_img = src_img.convert("RGBA")
        mask_img = mask_img.convert("L")

        # 创建一个新的图像，白色区域的alpha值为255，黑色区域的alpha值为0
        mask_rgba = Image.new("L", mask_img.size)
        mask_rgba.putdata([255 if pixel == 255 else 0 for pixel in mask_img.getdata()])

        # 使用mask_rgba作为掩码合成src_img和透明背景
        combined_image = Image.composite(src_img, Image.new("RGBA", src_img.size, (0, 0, 0, 0)), mask_rgba)

        return combined_image

    @staticmethod
    def get_dataset(dataset_dir: Path):
        """
        获取文件夹下的所有图片文件及其对应的掩码文件

        Args:
            dataset_dir (Path): 数据集目录路径

        Returns:
            list: 包含[图像文件路径, 掩码文件路径]对的列表
        """
        dataset_dir = Path(dataset_dir)
        # 常见的图像文件扩展名
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]

        # 收集所有图像文件（包括大写后缀）
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_dir.glob(f"*{ext}"))
            image_files.extend(dataset_dir.glob(f"*{ext.upper()}"))
        return image_files

    @staticmethod
    def download_image(url: str, save_path: Path):
        """
        下载图片并保存到指定路径
        """
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Image downloaded and saved to {save_path}")
        else:
            print(f"Failed to download image from {url}")
