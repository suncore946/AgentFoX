from typing import Any, Dict

import numpy as np
import pandas as pd
from ...processor import ExpertAnalysisProcessor

from .expert_results_tool import ExpertResultsTool
from .tools_base import ToolsBase, skip_auto_register
from ...utils import create_chat_llm
from ...manager.datasets_manager import DatasetsManager
from ...manager.profile_manager import ProfileManager


class ExpertAnalysisTool(ToolsBase):
    def __init__(self, config, tools_llm, profile_manager: ProfileManager, datasets_manager: DatasetsManager, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.get("llm_config", None):
            # 使用配置文件中的 LLM 设置
            self.tools_llm = create_chat_llm(config["llm_config"])
        else:
            self.tools_llm = tools_llm  # 使用传入的 LLM 实例
        self.expert_analysis = ExpertAnalysisProcessor(config, prompt_path=config.get("prompt_path", None), tools_llm=self.tools_llm)
        self.model_results_tools = ExpertResultsTool(config, profile_manager, datasets_manager, *args, **kwargs)

        expert_analysis = profile_manager.expert_analysis
        if expert_analysis.empty:
            self.expert_analysis_data = datasets_manager.detail_data.drop_duplicates(subset=["image_path"], keep="first")
        else:
            self.expert_analysis_data = expert_analysis

        self.profiles_manager = profile_manager

        self.open_calibration = config.get("open_calibration", True)

        if self.open_calibration:
            self.target_columns = "expert_analysis"
        else:
            self.target_columns = "expert_analysis_without_calibration"

    @property
    def name(self) -> str:
        return "expert_analysis"

    @property
    def description(self) -> str:
        return """Expert model analysis tool. Can intelligently analyze and integrate detection results of multiple expert models result based on expert model profiles."""

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        params = self.args_schema.model_validate(kwargs)
        image_path = params.get_image_path()

        model_result = self.model_results_tools.execute(**kwargs)
        profiles = self.profiles_manager.get_model_profiles(list(model_result.keys()))

        # 检查目标列是否存在于 detail_data
        if self.target_columns in self.expert_analysis_data.columns:
            # 根据 image_path 获取对应的 expert_analysis 内容（可能为空）
            expert_rows = self.expert_analysis_data.loc[
                self.expert_analysis_data["image_path"] == image_path,
                self.target_columns,
            ]
            if not expert_rows.empty and expert_rows.iloc[0] is not np.nan:
                # 安全获取第一条值
                model_analysis = str(expert_rows.iloc[0])
                if len(model_analysis.strip()) >= 10000:
                    model_analysis = model_analysis[:10000] + "...[truncated]"
                if not model_analysis.startswith("Error during expert analysis"):
                    return {"model_analysis": model_analysis}

        # 无缓存或缓存包含错误，执行实时分析
        # TODO: 需要调整目标LLM, 目前使用与tools_llm相同的LLM
        try:
            self.logger.warning(f"Performing expert analysis for image_path={image_path}")
            info: Dict[str, Any] = {"expert_result": model_result, "expert_profiles": profiles}
            if self.open_calibration:
                info["calibration_profiles"] = self.profiles_manager.calibration_profiles
            model_analysis = self.expert_analysis.run(model_profile=info)
            return {"model_analysis": model_analysis}
        except Exception as e:
            # 记录错误并返回包含错误信息的结构
            self.logger.exception(f"Expert analysis failed for image_path={image_path}")
            return {"error": f"Error during expert analysis: {str(e)}"}
