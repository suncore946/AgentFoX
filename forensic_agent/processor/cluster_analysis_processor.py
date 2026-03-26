# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Any, Dict

from typing import Dict, Any
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLLM
from .base_prompt import BaseParser
from .base_processor import BaseProcessor


class ClusterAnalysisModel(BaseModel):
    """簇结果分析模型"""

    reasoning_process: str = Field(..., description="Key reasoning process and basis")
    confidence_level: str = Field(..., description="Confidence level assessment of model results")


class ClusterAnalysisPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = Path(__file__).parent.parent / "configs" / "prompts" / "cluster_analysis_prompt.txt"
        super().__init__(llm, prompt_path, ClusterAnalysisModel, store=store)


class ClusterAnalysisProcessor(BaseProcessor):
    USER_PROMPT = """Please conduct a comprehensive analysis based on the cluster profile and clustering results, and provide conclusions and confidence levels for each model.
**Cluster Profile**
{CLUSTER_PROFILE}
**Clustering Results for This Image**
{CLUSTER_RESULTS}
"""
    PROCESS_NAME = "cluster_analysis"

    def __init__(self, config: dict, store=None, image_type="natural", prompt_path: str = None, tools_llm=None):
        """初始化图像分析器"""
        super().__init__(config, self.PROCESS_NAME, llm=tools_llm)
        self.store = store if store is not None else {}
        if image_type == "natural":
            self.image_analyst = ClusterAnalysisPrompt(self.llm, store=self.store, prompt_path=prompt_path)
        else:
            raise ValueError(f"Unsupported image type: {image_type}. Supported types are: natural.")

    def process_file(self, image_path, model_result, model_profile, *args, **kwargs) -> Any:
        """执行图像分析"""
        image_path = Path(image_path)
        user_prompt = self.USER_PROMPT.format(CLUSTER_PROFILE=model_profile, CLUSTER_RESULTS=model_result)
        human_prompt = self.load_human_msg(text_content=user_prompt)
        session_id = hashlib.sha256(image_path.as_posix().encode()).hexdigest()
        analysis_result = self.image_analyst.run(human_prompt, session_id)
        return analysis_result
