# -*- coding: utf-8 -*-
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLLM
from .base_prompt import BaseParser
from .base_processor import BaseProcessor


class ComplexReasoningModel(BaseModel):
    """综合伪造分析结果：基于元信息、空间域特征、频域特征、多专家模型结果的智能深度分析，提供可解释的取证推理文档"""

    # 1. 综合异常评估（对应prompt的第1部分：量化每个维度的异常分数、权重和整体分数）
    comprehensive_anomaly_assessment: Dict[str, Dict[str, float]] = Field(
        description="综合异常评估：列出每个维度（元信息、空间域、频域、多专家模型）的异常分数（0-10）、权重（总和为100%），并计算整体异常分数（加权平均值）。基于技术特征和专家结果量化严重度。"
    )
    overall_anomaly_score: float = Field(description="整体异常分数：所有维度的加权平均值（0-10），表示伪造痕迹的综合严重度。", ge=0, le=10)

    # 2. 证据链和深度推理（对应prompt的第2部分：详细证据描述、整合分析和冲突处理）
    evidence_chain: List[Dict[str, Any]] = Field(
        description="证据链和深度推理：详细描述整合后的证据，包括元信息、空间域、频域、多专家模型结果的关联分析、伪造痕迹推理"
    )
    conflict_and_uncertainty_analysis: Dict[str, str] = Field(
        description="冲突和不确定性分析：处理模型间或特征间分歧的原因解释（如'空间域一致但频域异常，可能源于对抗攻击'），并评估不确定性水平（低/中/高）。"
    )
    # 3. 伪造概率和置信度（对应prompt的第3部分：概率评估、置信度和不确定性说明）
    forgery_probability: float = Field(
        description="伪造概率评估：基于量化分析和多源证据推断的整体伪造概率（0-100%），符合法庭科学证据标准。", ge=0, le=100
    )
    detection_confidence: Dict[str, Any] = Field(
        description="检测置信度：包括量化分数（0-100%）、水平（低/中/高）和不确定性因素说明（如数据盲区或对抗攻击影响）。"
    )
    # 4. 最终结论（对应prompt的第4部分：专业判断、关键依据和建议）
    final_conclusion: str = Field(description="专业取证报告：符合数字取证标准的总结，包括主要发现、技术认定、风险评估和可解释推理链。")


class ComplexReasoningPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = self.__class__.__name__
        super().__init__(llm, prompt_path, ComplexReasoningModel, store=store)


class FinalDeterminationProcessor(BaseProcessor):
    # 支持的图片后缀
    USER_PROMPT = "簇类说明如下: '''{CLUSTER_DATA}''', 专家报告如下: '''{EXPERT_DATA}'''"
    USER_PROMPT_FEATURES_ONLY = "簇分类结果如下: '''{CLUSTER_DATA}'''"
    USER_PROMPT_EXPERT_ONLY = "专家报告如下: '''{EXPERT_DATA}'''"
    USER_PROMPT_NONE = "请基于提供的图像进行分析"
    PROCESS_NAME = "complex_reasoning_analysis"

    def __init__(self, config: dict, store=None, prompt_path: str = None, llm=None):
        """初始化图像分析器"""
        super().__init__(config, self.PROCESS_NAME, llm=llm)
        self.store = store if store is not None else {}
        self.complex_reasoning = ComplexReasoningPrompt(self.llm, store=self.store, prompt_path=prompt_path)

    def process_file(
        self,
        image_path: Path,
        image_base64: None,
        cluster_data: dict | str = None,
        expert_data: dict | str = None,
        *args,
        **kwargs,
    ) -> Any:
        """执行图像分析"""
        image_path = Path(image_path)
        if isinstance(cluster_data, dict):
            cluster_data = json.dumps(cluster_data, ensure_ascii=False)

        # 根据数据是否为None选择不同的提示模板
        if cluster_data is not None and expert_data is not None:
            user_prompt = self.USER_PROMPT.format(CLUSTER_DATA=cluster_data, EXPERT_DATA=expert_data)
        elif cluster_data is not None and expert_data is None:
            user_prompt = self.USER_PROMPT_FEATURES_ONLY.format(CLUSTER_DATA=cluster_data)
        elif cluster_data is None and expert_data is not None:
            user_prompt = self.USER_PROMPT_EXPERT_ONLY.format(EXPERT_DATA=expert_data)
        else:
            user_prompt = self.USER_PROMPT_NONE

        human_prompt = self.load_human_msg(image_path, user_prompt, image_base64=image_base64)
        session_id = hashlib.sha256(image_path.as_posix().encode()).hexdigest()
        analysis_result = self.complex_reasoning.run(human_prompt, session_id)
        return analysis_result
