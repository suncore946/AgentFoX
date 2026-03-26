# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Any, Dict

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseLLM
from .base_prompt import BaseParser
from .base_processor import BaseProcessor


class ExpertAnalysisPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = Path(self.__file__).parent.parent / "configs" / "prompts" / "expert_analysis_prompt.txt"
        super().__init__(llm, prompt_path, None, store=store)


class ExpertAnalysisProcessor(BaseProcessor):

    USER_PROMPT = """Create a comprehensive forensic analysis report comparing multiple AI forensic model outputs

**Analysis Focus:**
- Analyze discrepancies between AI model predictions/classifications
- Evaluate each model's reasoning process and confidence indicators
- Identify systematic biases, edge cases, and failure modes
- Apply expert judgment to reconcile conflicting outputs
- Consider model architecture, training data, and domain specialization

**Report Structure:**
1. **Executive Summary** - Key findings and recommendations
2. **Individual Model Analysis** - Each model's strengths, weaknesses, and specializations
3. **Conflict Assessment** - Areas of disagreement and underlying causation
4. **Recommendations** - Which approach to use when, and suggestions for improvement

**Input Data:** {MODELS_PROFILE}

**Writing Style:** 
- Maintain a professional and authoritative tone
- Clear, declarative statements
- Reference specific models with justification
- Keep content concise and well-organized

**Final Output Requirement:** Conclude with a single, coherent paragraph that provides a logically clear summary of your comprehensive analysis.

"""
    PROCESS_NAME = "expert_analysis"

    def __init__(self, config: dict = None, store=None, prompt_path: str = None, tools_llm=None):
        """初始化图像分析器"""
        super().__init__(config, self.PROCESS_NAME, llm=tools_llm)
        self.store = store if store is not None else {}
        self.image_analyst = ExpertAnalysisPrompt(self.llm, store=self.store, prompt_path=prompt_path)

    def process_file(self, model_profile, *args, **kwargs) -> Any:
        """执行图像分析"""
        user_prompt = self.USER_PROMPT.format(MODELS_PROFILE=model_profile)
        human_prompt = self.load_human_msg(text_content=user_prompt)
        session_id = hashlib.sha256(user_prompt.encode()).hexdigest()
        analysis_result = self.image_analyst.run(human_prompt, session_id)
        return analysis_result

    def process_file_with_gt(self, model_profile, image_path, image_label, *args, **kwargs) -> Any:
        """执行图像分析"""
        user_prompt = self.USER_PROMPT.format(MODELS_PROFILE=model_profile)
        human_prompt = self.load_human_msg(text_content=user_prompt)
        session_id = hashlib.sha256(user_prompt.encode()).hexdigest()
        self.image_analyst.run(human_prompt, session_id)

        human_prompt_gt = HumanMessage(
            role="user",
            content=f"""
Now I will provide the gt label information for this image. Please conduct a comprehensive re-analysis based on the previous analysis results combined with the true labels, and provide the final conclusion.
Please note that the conclusion should not contain the true label information, but should only use it as reference for analysis. Please directly provide the revised comprehensive analysis conclusion without including any other content, including modal particles.

```
This image is labeled as: {image_label}
```
""",
        )
        analysis_result_gt = self.image_analyst.run(human_prompt_gt, session_id)
        return analysis_result_gt
