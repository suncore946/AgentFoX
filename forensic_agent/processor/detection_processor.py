# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLLM
from .base_prompt import BaseParser
from .base_processor import BaseProcessor


class DetectionRes(BaseModel):
    pred_label: int = Field(
        description="Classification label: 0 indicates authentic (camera-captured), 1 indicates forged (AI-generated or AI-manipulated)",
        required=True,
    )
    reason_for_thinking: str = Field(
        description="Detailed explanation of the forensic assessment, including supporting evidence and analysis rationale", required=True
    )


class DetectionPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = self.__class__.__name__
        super().__init__(llm, prompt_path, DetectionRes, store=store)


class DetectionProcessor(BaseProcessor):
    # 支持的图片后缀
    USER_PROMPT_NONE = "Please analyze whether the provided image is AI-generated or natural. Please do not describe any elements that do not exist in the picture. This might be because the picture has undergone certain cropping, but it does not affect the authenticity verification of the image."
    PROCESS_NAME = "detection_reasoning_analysis"

    def __init__(self, config: dict = {}, store=None, prompt_path: str = None, llm=None):
        """初始化图像分析器"""
        super().__init__(config, self.PROCESS_NAME, llm=llm)
        self.store = store if store is not None else {}
        if prompt_path is None:
            if "prompt_path" in config and isinstance(config["prompt_path"], str):
                prompt_path = config["prompt_path"]
            elif "prompt_path" in config.get("llm"):
                prompt_path = config["llm"]["prompt_path"]
            elif "prompt_path" in config:
                prompt_path = config["prompt_path"]
        self.complex_reasoning = DetectionPrompt(self.llm, store=self.store, prompt_path=prompt_path)

    def process_file(self, image_path: Path, image_base64: None, *args, **kwargs) -> Any:
        """执行图像分析"""
        image_path = Path(image_path)
        human_prompt = self.load_human_msg(image_path, self.USER_PROMPT_NONE, image_base64=image_base64)
        session_id = hashlib.sha256(image_path.as_posix().encode()).hexdigest()
        analysis_result = self.complex_reasoning.run(human_prompt, session_id)
        return analysis_result
