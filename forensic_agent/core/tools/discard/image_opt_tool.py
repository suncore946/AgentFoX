from pathlib import Path
from typing import Annotated
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from langgraph.prebuilt import InjectedState, ToolNode

from ..tools_base import ToolsBase
from ....manager.image_manager import ImageManager


class ImageToBase64Tool(ToolsBase):
    def __init__(self, config, image_manager: ImageManager, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.image_manager = image_manager
        self.save_path = config.get("save_path", "./output.json")

    @property
    def name(self) -> str:
        return "image_to_base64"

    @property
    def description(self) -> str:
        return "Convert image to base64 encoded string, input is image file path, output contains base64 encoded string"

    def execute(self, **kwargs):
        params = self.args_schema.model_validate(kwargs)
        image_path = Path(params.get_image_path())
        assert image_path.is_file(), f"输入路径 {image_path} 不是一个正确的图片路径"

        with Image.open(image_path) as src_img:
            img_type = src_img.format.lower()
            logger.info(f"将图像 {image_path}（类型：{img_type}）转换为base64编码")
            base64_str, _, _ = self.image_manager.get_base64(src_image=image_path, is_resize=True, target_width=224, target_height=224)
        return base64_str
