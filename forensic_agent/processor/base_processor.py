# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..processor.base_prompt import ErrorDetail
from ..manager.image_manager import ImageManager
from ..utils import create_chat_llm


class BaseProcessor(ABC):
    """图像处理基类，提取通用功能"""

    def __init__(self, config: dict, processor_name: str, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or create_chat_llm(config)
        self.image_manager = ImageManager(config.get("ImageManager", config))
        self.processor_name = processor_name

    def load_human_msg(
        self,
        image_path: Path = None,
        text_content: str = None,
        image_base64: Optional[str] = None,
        image_format: Optional[str] = None,
        *args,
        **kwargs,
    ) -> HumanMessage:
        """加载图像数据并构建消息"""
        content = []
        # 要求image_path或text_content至少有一个不为None
        assert image_path is not None or text_content is not None, "image_path or text_content must be provided"

        if image_path is not None:
            if image_base64 is None:
                src_img, _, _ = self.image_manager.load_image(image_path)
                image_base64, _, format_info = self.image_manager.get_base64(src_image=src_img, is_resize=True)
            else:
                format_info = image_format or Path(image_path).suffix.lstrip(".").upper()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{format_info};base64,{image_base64}",
                    },
                }
            )
        elif image_base64 is not None:
            format_info = image_format or "JPEG"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{format_info};base64,{image_base64}",
                    },
                }
            )
        if text_content:
            content.append(
                {
                    "type": "text",
                    "text": text_content,
                }
            )

        return HumanMessage(role="user", content=content)

    @abstractmethod
    def process_file(self, *args, **kwargs) -> Optional[dict]:
        """处理单个文件的抽象方法，子类必须实现"""
        pass

    def pre_hook(self, image_path: Path, *args, **kwargs) -> None:
        """预处理钩子方法，可选实现"""
        # logger.info(f"执行任务 [{self.processor_name}]: [{image_path}]")
        pass

    def post_hook(self, image_path: Path, result: dict, *args, **kwargs) -> None:
        """后处理钩子方法，可选实现"""
        # logger.info(f"任务执行完毕 [{self.processor_name}]: [{image_path}]")
        pass

    def run(self, image_path: Path = None, image_base64: Optional[str] = None, image_format=None, *args, **kwargs) -> dict:
        """运行处理流程"""
        self.pre_hook(image_path, image_base64)
        kwargs.update(
            {
                "image_path": image_path,
                "image_base64": image_base64,
                "image_format": image_format,
            }
        )
        analysis_result = self.process_file(*args, **kwargs)
        if isinstance(analysis_result, ErrorDetail):
            ret = analysis_result.model_dump()
        elif isinstance(analysis_result, BaseModel):
            ret = analysis_result.model_dump()
            ret = ret[0] if len(ret) == 1 else ret
        else:
            ret = analysis_result
        self.post_hook(image_path, ret)
        return ret
