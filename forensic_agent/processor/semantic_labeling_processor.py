# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Any
from enum import Enum
import json
from typing import Any, Dict
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLLM

from .base_processor import BaseProcessor
from .base_prompt import BaseParser
from ..utils.custom_json_encoder import CustomJsonEncoder


class ForensicTracesModel(BaseModel):
    observations: str = Field(..., description="Brief, objective description of key scene elements and cues used.")
    detected_anomalies: str = Field(..., description="Detailed description of detected anomalies or inconsistencies.")
    limitations: str = Field(..., description="Note any factors that reduce reliability (resolution, occlusion, ambiguity).")
    pred_label: int = Field(..., description="1 indicates the AIGC Image and 0 indicates the natural Image.")


class ForensicTracesPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = self.__class__.__name__
        super().__init__(llm, prompt_path, ForensicTracesModel, store=store)


class SemanticLabelingProcessor(BaseProcessor):
    # 支持的图片后缀
    USER_PROMPT = """
Please analyze this image, focusing on forensic traces. The following are the relevant features of this image:
{IMAGE_FEATURES}

This is a {IMAGE_LABEL} image. Based on the features above and professional knowledge of image forensics, please provide a detailed forensic analysis.

Please output strictly in JSON format as per the following schema.
"""

    PROCESS_NAME = "Forensic_Traces"

    def __init__(self, config: dict = {}, store=None, image_type="natural", prompt_path: str = None, forensic_llm=None):
        """初始化图像分析器"""
        super().__init__(config, self.PROCESS_NAME, llm=forensic_llm)
        self.store = store if store is not None else {}
        if image_type == "natural":
            self.image_analyst = ForensicTracesPrompt(self.llm, store=self.store, prompt_path=prompt_path)
        else:
            raise ValueError(f"Unsupported image type: {image_type}. Supported types are: natural.")

    def process_file(self, image_path: Path, image_base64: None, image_format, image_features, image_label, *args, **kwargs) -> Any:
        """执行图像分析"""
        if image_path is not None:
            image_path = Path(image_path)
            session_id = hashlib.sha256(image_path.as_posix().encode()).hexdigest()
        else:
            session_id = hashlib.sha256(image_base64.encode()).hexdigest()
        image_features = json.dumps(image_features, ensure_ascii=False, cls=CustomJsonEncoder)
        user_prompt = self.USER_PROMPT.format(IMAGE_FEATURES=image_features, IMAGE_LABEL=image_label)
        human_prompt = self.load_human_msg(image_path, user_prompt, image_base64=image_base64, image_format=image_format)
        analysis_result = self.image_analyst.run(human_prompt, session_id)
        return analysis_result
