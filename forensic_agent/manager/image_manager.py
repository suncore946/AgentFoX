"""Image loading and encoding helpers.

中文说明: ImageManager 只处理用户提供的本地图片或 URL, 不包含默认资源路径。
English: ImageManager handles user-provided local images or URLs only and does
not contain bundled resource paths.
"""

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
    """Load, resize, and encode images for LLM input.

    中文说明: 这里不会缓存图片内容, 每次调用按需读取并返回内存对象。
    English: Image content is not cached here; each call reads on demand and
    returns in-memory objects.
    """

    def __init__(self, config: dict):
        """Initialize image limits.

        中文说明: max_width/max_height 只用于工具提示中的压缩视图, 原图文件不会被改写。
        English: max_width/max_height only affect the compressed view sent to
        tools; source image files are never modified.
        """
        config = config or {}
        self.max_width = config.get("max_width", 128)
        self.max_height = config.get("max_height", 128)
        self.font_path = config.get("font_path")
        self.font_size = config.get("font_size", 40)
        self.maintain_aspect_ratio = config.get("maintain_aspect_ratio", True)

    def reduce_image(self, image, target_width=None, target_height=None):
        """Resize an image within configured bounds.

        中文说明: 默认等比缩放, 避免改变取证目标的几何比例。
        English: Aspect ratio is preserved by default to avoid distorting
        forensic geometry cues.
        """
        original_width, original_height = image.size

        max_width = target_width if target_width is not None else self.max_width
        max_height = target_height if target_height is not None else self.max_height

        if max_width is None and max_height is None:
            return image, 1

        if not self.maintain_aspect_ratio:
            new_width = max_width or original_width
            new_height = max_height or original_height
            width_ratio = new_width / original_width
            height_ratio = new_height / original_height
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return resized_image, (width_ratio, height_ratio)

        if (max_width is None or original_width <= max_width) and (max_height is None or original_height <= max_height):
            return image, 1

        width_ratio = max_width / original_width if max_width is not None else float("inf")
        height_ratio = max_height / original_height if max_height is not None else float("inf")

        scale_ratio = min(width_ratio, height_ratio)

        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

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
        """Return base64, transformed image, and format.

        中文说明: LLM/VLM 输入需要 data URL, 因此这里统一保留图片格式并编码。
        English: LLM/VLM input needs data URLs, so this preserves image format
        and performs encoding in one place.
        """
        if isinstance(src_image, str) or isinstance(src_image, Path):
            loaded_image = self.load_image(src_image)[0]
        elif isinstance(src_image, np.ndarray):
            loaded_image = Image.fromarray(src_image)
        else:
            loaded_image = src_image

        format_info = loaded_image.format
        if format_info is None:
            format_info = "PNG"

        if is_resize:
            trans_image, _ = self.reduce_image(loaded_image, target_width, target_height)
            logger.debug(f"Resized image to: {trans_image.size}")
        else:
            trans_image = loaded_image

        if is_center_crop and target_width and target_height:
            trans_image = ImageOps.fit(trans_image, (target_width, target_height), method=Image.Resampling.LANCZOS)
            logger.debug(f"Center cropped image to: {trans_image.size}")

        with BytesIO() as buffered:
            trans_image.save(buffered, format=format_info)
            trans_base64 = base64.b64encode(buffered.getvalue()).decode()

        return trans_base64, trans_image, format_info

    def base64_to_image(self, b64_str: str) -> Image.Image:
        """Decode base64 into a PIL Image.

        中文说明: 支持带 data URL 前缀的字符串。
        English: Supports strings with a data URL prefix.
        """
        b64_str = b64_str.split(",")[-1]
        data = base64.b64decode(b64_str)
        return Image.open(BytesIO(data))

    def load_image(self, image_file: Path, image_type="RGB"):
        """Load one image from disk or URL.

        中文说明: 本地图片用 PIL 直接打开以保留 format 信息; URL 图片会转为 RGB。
        English: Local images are opened directly to preserve format metadata;
        URL images are converted to RGB.
        """
        image_file = Path(image_file)
        if self.is_url(str(image_file)):
            src_image = Image.open(requests.get(image_file, stream=True).raw).convert(image_type)
        else:
            src_image = Image.open(image_file)
        trans_image, scale_ratio = self.reduce_image(src_image)
        return src_image, trans_image, scale_ratio

    def process_mask(self, src_mask, target_mask, scale_ratio):
        """Combine two masks.

        中文说明: 最小 test 流程不使用该方法, 保留给未来分割类扩展。
        English: The minimal test path does not use this method; it is kept for
        future segmentation-style extensions.
        """
        def read_mask(mask):
            if isinstance(mask, (str, Path)):
                return self.load_image(mask)[0]
            return mask

        src_mask_img = read_mask(src_mask)
        target_mask_img = read_mask(target_mask)

        if src_mask_img.shape != target_mask_img.shape:
            raise ValueError("Source and target masks must have the same dimensions.")

        combined_mask = np.maximum(src_mask_img, target_mask_img)

        if scale_ratio != 1.0:
            new_size = (int(combined_mask.shape[1] * scale_ratio), int(combined_mask.shape[0] * scale_ratio))
            combined_mask = cv2.resize(combined_mask, new_size, interpolation=cv2.INTER_NEAREST)

        combined_mask = Image.fromarray(combined_mask)
        return combined_mask

    @staticmethod
    def is_url(path: str):
        """Return whether a path string is a URL.

        中文说明: URL 图片不会按本地路径解析。
        English: URL images are not resolved as local paths.
        """
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def combine_images(src_img, mask_img):
        """Apply a binary mask to an image.

        中文说明: 白色区域保留, 黑色区域透明。
        English: White areas are kept and black areas become transparent.
        """
        src_img = src_img.convert("RGBA")
        mask_img = mask_img.convert("L")

        mask_rgba = Image.new("L", mask_img.size)
        mask_rgba.putdata([255 if pixel == 255 else 0 for pixel in mask_img.getdata()])

        combined_image = Image.composite(src_img, Image.new("RGBA", src_img.size, (0, 0, 0, 0)), mask_rgba)

        return combined_image

    @staticmethod
    def get_dataset(dataset_dir: Path):
        """List common image files in a directory.

        中文说明: 该工具只做文件枚举, 不读取图片内容。
        English: This helper only enumerates files and does not read image
        content.
        """
        dataset_dir = Path(dataset_dir)
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]

        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_dir.glob(f"*{ext}"))
            image_files.extend(dataset_dir.glob(f"*{ext.upper()}"))
        return image_files

    @staticmethod
    def download_image(url: str, save_path: Path):
        """Download an image to a user-specified path.

        中文说明: 不在默认流程中调用, 避免隐式联网。
        English: Not called in the default flow, avoiding implicit network use.
        """
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Image downloaded and saved to {save_path}")
        else:
            print(f"Failed to download image from {url}")
