# -*- coding: utf-8 -*-
import json
import hashlib
from typing import Dict, Any, List
import numpy as np
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage

from .base_prompt import BaseParser, ErrorDetail
from .base_processor import BaseProcessor
from ..utils.custom_json_encoder import CustomJsonEncoder


class ClusterProfiles(BaseModel):
    """聚类分析的完整配置文件"""

    cluster_profile: Dict[str, str] = Field(
        description="Dictionary mapping cluster_id to cluster profile details.Provide an analysis of the advantages and disadvantages of different expert models within each cluster, and provide detailed explanations for the reasons.",
        required=True,
    )
    comprehensive_analysis: str = Field(description="Overall analysis of model performance across all clusters.", required=True)


class ClusteringProfiles(BaseModel):
    clustering_analysis: Dict[str, str] = Field(
        required=True, description="Dictionary mapping clustering method names to their detailed analysis."
    )


class ClusteringProfilesPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = self.__class__.__name__
        super().__init__(llm, prompt_path, ClusterProfiles, store=store)


class ClusteringAnalysisPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None, prompt_path: str = None):
        if prompt_path is None:
            prompt_path = self.__class__.__name__
        super().__init__(llm, prompt_path, ClusteringProfiles, store=store)


class ClusteringProfilesProcessor(BaseProcessor):
    # 支持的图片后缀

    USER_PROMPT = """
Analyze each individual cluster and identify which models perform well or poorly within that specific cluster.

# PREFERRED ANALYSIS STYLE
1. Prioritize qualitative analysis. Focus primarily on narrative descriptions, architecture/capability comparisons, error modes, and scenario suitability.
2. Avoid quantitative analysis whenever possible (e.g., specific numbers, percentages, tables, or statistical tests). Only include a small amount of highly summarized quantitative information when it is truly necessary and would significantly strengthen the conclusions.
3. If numbers must be provided, present them briefly and without detail (for example, “significantly higher” or “substantially lower”), and do not display original tables or complete data lists.

# CRITICAL OUTPUT FORMAT REQUIREMENTS
1. You MUST return ONLY valid JSON - no markdown, no code blocks, no explanations
2. The response must be a single JSON object with exactly these two top-level fields:
   - "cluster_profile": a dictionary/object (NOT an array)
   - "comprehensive_analysis": a string
3. Each cluster_id must be a KEY in the "cluster_profile" dictionary
4. Each cluster value must contain exactly one string field with the analysis for that cluster, including explanations of the advantages and disadvantages of different forensic expert models under this cluster and detailed explanations of the underlying reasons.
5. DO NOT add any text before or after the JSON
6. Ensure the JSON is properly closed with all brackets and braces

# EXACT JSON STRUCTURE (follow this precisely)
{{
  "cluster_profile": {{
    "cluster_0": {{
      "Provide an analysis of the advantages and disadvantages of different expert models within each cluster, and provide detailed explanations for the reasons."
    }},
    "cluster_1": {{
      "Provide an analysis of the advantages and disadvantages of different expert models within each cluster, and provide detailed explanations for the reasons."
    }}
    ...
  }},
  "comprehensive_analysis": "Overall analysis across all clusters..."
}}

# ANALYSIS REQUIREMENTS
- Analyze EVERY cluster present in the performance data
- Use specific metrics and evidence from the performance data
- Identify models that excel vs. struggle in each specific cluster
- Provide actionable insights based on architectural factors
- Include quantitative comparisons where relevant

# DATA TO ANALYZE:
Clustering Name:
```
{CLUSTERING_NAME}
```

Clustering Desc
```
{CLUSTERING_DESC}
```

Per-Cluster Performance Metrics
```
{CLUSTER_PERFORMANCE}
```

Model Knowledge Base
```
{MODEL_PROFILES}
```
Return the complete, valid JSON object now:

"""

    PROCESS_NAME = "expert_profile"

    def __init__(self, config: dict, store=None, prompt_path: str = None, llm=None):
        """初始化图像分析器"""
        self.store = store if store is not None else {}
        if prompt_path is None:
            if "prompt_path" in config.get("llm", {}):
                prompt_path = config["llm"]["prompt_path"]
            elif "prompt_path" in config:
                prompt_path = config["prompt_path"]
        super().__init__(config, self.PROCESS_NAME, llm=llm)
        self.complex_reasoning = ClusteringProfilesPrompt(self.llm, store=self.store, prompt_path=prompt_path)
        self.clustering_analysis = ClusteringAnalysisPrompt(self.llm, store=self.store, prompt_path=prompt_path)

    def process_file(
        self,
        clustering_name: List[str] | str,
        clustering_description: str,
        performance: dict,
        model_profiles: dict,
        *args,
        **kwargs,
    ) -> Dict:
        """执行图像分析"""
        text_content = self.USER_PROMPT.format(
            CLUSTERING_NAME=clustering_name,
            CLUSTERING_DESC=clustering_description,
            CLUSTER_PERFORMANCE=json.dumps(performance, ensure_ascii=False),
            MODEL_PROFILES=json.dumps(model_profiles, ensure_ascii=False),
        )
        human_prompt = self.load_human_msg(text_content=text_content)
        session_id = hashlib.sha256(str(clustering_name).encode("utf-8")).hexdigest()
        analysis_result = self.complex_reasoning.run(human_prompt, session_id)
        return analysis_result

    def clustering_profile(self, info, *args, **kwargs):
        """执行聚类本身分析分析并返回结构化结果"""
        info = json.dumps(info, ensure_ascii=False)
        msg = f"""
As a data science expert, please conduct a comprehensive professional evaluation and analysis of the following different clustering methods based on the provided clustering quality analysis results.

## Analysis Requirements:
Please conduct in-depth analysis from the following dimensions, ensuring that each dimension's assessment is specific, quantified, and provides practical guidance:

### 1. Clustering Quality Assessment
- **Internal Quality Metrics**: Evaluate cluster cohesion and separation based on Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, etc.
- **Clustering Stability**: Analyze the stability and reproducibility of clustering results
- **Outlier Handling**: Identify and assess the impact of noise points and outliers

### 2. In-depth Clustering Feature Analysis
- **Cluster Profiling**: Provide detailed descriptions of core characteristics, size distribution, and typical sample features of each cluster
- **Inter-cluster Differences**: Quantitatively analyze key distinguishing features and similarities between different clusters
- **Feature Importance Ranking**: Identify key feature dimensions that contribute most to cluster segmentation

### 3. Real Scenario Application Value
- **Practicality Assessment**: Evaluate the usability and effectiveness of clustering results in real scenarios
- **Application Scenario Recommendations**: Propose specific application scenarios and implementation strategies
- **Risk Alerts**: Point out potential application risks and considerations

### Output Requirements:
- Output one paragraph for each clustering method, no less than 200 words, ensuring detailed and in-depth content
- Provide quantifiable evaluation conclusions and actionable recommendations

## Analysis Data:
{info}

Please ensure the analysis results are professional, objective, and have practical guidance value. Output a complete analytical text for each clustering method.
"""
        human_msg = HumanMessage(content=msg)
        session_id = hashlib.sha256(str(info).encode("utf-8")).hexdigest()
        clustering_analysis = self.clustering_analysis.run(human_msg, session_id)
        return clustering_analysis
