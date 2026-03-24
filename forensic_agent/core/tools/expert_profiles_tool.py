from typing import Any, Dict
from .expert_results_tool import ExpertResultsTool
from .tools_base import ToolsBase
from ...utils import create_chat_llm
from ...manager.datasets_manager import DatasetsManager
from ...manager.profile_manager import ProfileManager


class ExpertProfilesTool(ToolsBase):
    def __init__(self, config, tools_llm, profile_manager: ProfileManager, datasets_manager: DatasetsManager, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.get("llm_config", None):
            # 使用配置文件中的 LLM 设置
            self.tools_llm = create_chat_llm(config["llm_config"])
        else:
            self.tools_llm = tools_llm  # 使用传入的 LLM 实例
        self.model_results_tools = ExpertResultsTool(config, profile_manager, datasets_manager, *args, **kwargs)
        self.profiles_manager = profile_manager
        self.open_calibration = config.get("open_calibration", True)

    @property
    def name(self) -> str:
        return "expert_profiles"

    @property
    def description(self) -> str:
        return """Retrieve expert profile information for each expert model, including detailed performance metrics and applicability analysis for each expert model."""

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        model_result = self.model_results_tools.execute(**kwargs)
        model_info = self.profiles_manager.get_model_profiles(list(model_result.keys()))
        if self.open_calibration:
            model_info["calibration_profile"] = self.profiles_manager.calibration_profiles
            model_info["calibration_note"] = "Due to differences between the calibration data and real-world scenarios, the results are for reference only. "
        return model_info
