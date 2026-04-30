"""Base processor utilities for image-aware LLM calls.

中文说明: BaseProcessor 负责把图片和文本打包成 LangChain HumanMessage, 不持久化图片内容。
English: BaseProcessor packages images and text into LangChain HumanMessage
objects and does not persist image content.
"""

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
    """Common image processor base class.

    中文说明: 子类只需要实现 process_file, run 会统一处理前后钩子和返回值规范化。
    English: Subclasses only implement process_file; run handles hooks and
    return-value normalization.
    """

    def __init__(self, config: dict, processor_name: str, llm: Optional[ChatOpenAI] = None):
        """Initialize processor dependencies.

        中文说明: 若外部未传入 LLM, 会根据配置创建工具层 LLM。
        English: If no LLM is passed in, a tool-level LLM is created from config.
        """
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
        """Build a HumanMessage from image and text inputs.

        中文说明: 支持传入已有 base64, 避免同一张图片重复读取和编码。
        English: Existing base64 input is accepted to avoid reading and encoding
        the same image repeatedly.
        """
        content = []
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
        """Process one input item.

        中文说明: 子类在这里实现具体的特征提取或 LLM 分析。
        English: Subclasses implement concrete feature extraction or LLM
        analysis here.
        """
        pass

    def pre_hook(self, image_path: Path, *args, **kwargs) -> None:
        """Optional pre-processing hook.

        中文说明: 默认不做任何副作用。
        English: No side effects by default.
        """
        pass

    def post_hook(self, image_path: Path, result: dict, *args, **kwargs) -> None:
        """Optional post-processing hook.

        中文说明: 默认不写缓存, 缓存由 ProfileManager 统一负责。
        English: Does not write cache by default; ProfileManager owns caching.
        """
        pass

    def run(self, image_path: Path = None, image_base64: Optional[str] = None, image_format=None, *args, **kwargs) -> dict:
        """Run the processor pipeline.

        中文说明: Pydantic 返回值会转成普通 dict, 便于 JSON 缓存。
        English: Pydantic outputs are converted to dicts for JSON caching.
        """
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
