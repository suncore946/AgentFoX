# -*- coding: utf-8 -*-
import json
import hashlib
from langchain_core.language_models import BaseLLM
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from .base_prompt import BaseParser, ErrorDetail
from .base_processor import BaseProcessor


class ModelProfilesAnalysis(BaseModel):
    observation: str = Field(
        ...,
        description="Describe the overall performance level and distribution characteristics of each model on the calibrated validation set; identify the stratified distribution of performance (tiers such as excellent, good, average); provide preliminary ranking trends and relative advantage relationships based on performance on the validation set after calibration; and offer 3–5 qualitative key insights regarding model rankings and calibration effects.",
    )
    depth_interpretation: str = Field(
        ...,
        description="Analyze the hierarchical ranking relationships among models and the differences in tiered performance characteristics; qualitatively assess the impact and degree of improvement of calibration on the ranking status of different models; qualitatively compare each model’s relative strengths and weaknesses; identify traits of models that stand out on the validation set; and evaluate ranking reliability and potential for broader application.",
    )
    comparative_analysis: str = Field(
        ...,
        description="Describe the ranking landscape based on performance on the validation set after calibration; use trend analysis, relative comparison, and hierarchical segmentation to present the ranking relationships among models; avoid listing specific numbers; focus on analyzing performance characteristics and differences in calibration responses across tiers such as the first tier, second tier, etc.",
    )
    suggestions: str = Field(
        ...,
        description="Based on ranking performance on the validation set, qualitatively indicate model selection preferences for different application scenarios; explain the reference value and applicability of this calibrated-validation-set evaluation for model selection on other datasets; qualitatively summarize the effectiveness of the calibration strategy and its positive impact on model rankings; and provide ranking-driven application recommendations.",
    )


class ModelProfilesPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = self.__class__.__name__
        super().__init__(llm, prompt_path, ModelProfilesAnalysis, store=store)


class ExpertProfilesProcessor(BaseProcessor):
    # 支持的图片后缀

    USER_PROMPT = """
    # Specific Task
    Please provide professional interpretation and analysis of the performance data of the AI image detection models I provide on the calibration validation set, with in-depth analysis focusing on model ranking and calibration effectiveness.

    **Important Note**: The data analyzed comes from model performance evaluation on the validation set after calibration. The summary and conclusions from your analysis are expected to serve as reference standards for model selection on other datasets to some extent.

    # Specific Analysis Requirements
    ## Part One: Calibration Validation Set Performance Overview and Preliminary Ranking
    1. **Overall Performance Observation**: Qualitatively describe the overall performance level and distribution characteristics of each model on the calibration validation set
    2. **Performance Tier Summary**: Identify the hierarchical distribution of model performance, describing the characteristics of different performance tiers such as excellent, good, average, etc.
    3. **Preliminary Ranking Trends**: Based on post-calibration validation set performance, qualitatively analyze model ranking patterns and relative advantage relationships
    4. **Key Findings**: Present 3-5 qualitative core insights regarding model ranking and calibration effectiveness

    ## Part Two: In-depth Ranking Interpretation and Calibration Effect Analysis
    1. **Ranking Pattern Analysis**: Qualitatively describe the hierarchical relationships among models, analyzing the performance characteristic differences between first-tier, second-tier, etc.
    2. **Calibration Effect Assessment**: From a multifaceted, analyze the impact and degree of improvement that calibration has on different models' ranking positions
    3. **Relative Advantage Analysis**: Qualitatively compare the relative strengths and weaknesses of each model, identifying model characteristics that stand out on the validation set
    4. **Ranking Stability Assessment**: Evaluate the reliability and generalization potential of rankings based on current validation set performance

    ## Part Three: Model Selection Guidance and Calibration Value Summary
    1. **Ranking-driven Selection Recommendations**: Based on validation set ranking performance, qualitatively provide model selection preferences for different application scenarios
    2. **Reference Value Assessment**: Elaborate on the reference significance and applicability of this calibration validation set evaluation results for model selection on other datasets
    3. **Calibration Strategy Insights**: Based on validation set performance, qualitatively summarize the effectiveness of calibration strategies and their positive impact on model ranking

    # Output Requirements
    Please conduct qualitative analysis strictly following the above three-part structure, avoiding excessive specific numerical enumeration. Focus on demonstrating model ranking and calibration effects through qualitative approaches such as trend description, relative comparison, and hierarchical analysis, ensuring clear analytical logic, ranking-oriented focus, and calibration-centered approach.

    # Data Input
    {MODEL_DATA}
    """

    PROCESS_NAME = "calibration_profile"

    def __init__(self, config: dict, store=None, prompt_path: str = None, llm=None):
        """初始化图像分析器"""
        self.store = store if store is not None else {}
        if prompt_path is None:
            if "prompt_path" in config.get("llm", {}):
                prompt_path = config["llm"]["prompt_path"]
            elif "prompt_path" in config:
                prompt_path = config["prompt_path"]
        super().__init__(config, self.PROCESS_NAME, llm=llm)
        self.complex_reasoning = ModelProfilesPrompt(self.llm, store=self.store, prompt_path=prompt_path)

    def process_file(self, model_data, *args, **kwargs) -> Dict:
        """执行图像分析"""
        text_content = self.USER_PROMPT.format(MODEL_DATA=json.dumps(model_data, ensure_ascii=False))
        human_prompt = self.load_human_msg(text_content=text_content)
        session_id = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
        analysis_result = self.complex_reasoning.run(human_prompt, session_id)
        return analysis_result
