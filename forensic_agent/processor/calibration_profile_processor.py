# -*- coding: utf-8 -*-
import json
import hashlib
from langchain_core.language_models import BaseLLM
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from .base_prompt import BaseParser, ErrorDetail
from .base_processor import BaseProcessor


class CalibrationProfilesPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = self.__class__.__name__
        super().__init__(llm, prompt_path, None, store=store)


class CalibrationProfilesProcessor(BaseProcessor):
    # 支持的图片后缀

    USER_PROMPT = """
    Please analyze the calibration results of the following image authenticity binary classification model. I have provided various metrics before and after calibration along with their Bootstrap confidence intervals:
    {CALIBRATION_DATA}

**Please conduct the analysis following this structure:**
1. **Calibration Effectiveness Assessment**
- Improvement in various calibration error metrics
- Statistical significance of improvements (based on whether confidence intervals overlap)
- Overall evaluation of calibration quality
2. **Performance Impact Analysis**
- Changes in F1 and ACC and their statistical significance
- Quantitative analysis of performance loss/gain
- Assessment of acceptability of performance changes
3. **Probability Quality Analysis**
- Improvement in Brier Score and Log Loss
- Degree of enhancement in prediction probability reliability
- Changes in uncertainty quantification quality
4. **Statistical Significance Summary**
- Which improvements have statistical significance
- Analysis of confidence interval overlap situations
- Assessment of improvement credibility
5. **Practical Recommendations**
- Whether calibration was successful and if the model is suitable for deployment
- Further optimization directions
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
        self.complex_reasoning = CalibrationProfilesPrompt(self.llm, store=self.store, prompt_path=prompt_path)

    def process_file(self, calibration_data, *args, **kwargs) -> Dict:
        """执行图像分析"""
        text_content = self.USER_PROMPT.format(CALIBRATION_DATA=json.dumps(calibration_data, ensure_ascii=False))
        human_prompt = self.load_human_msg(text_content=text_content)
        session_id = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
        analysis_result = self.complex_reasoning.run(human_prompt, session_id)
        return analysis_result
