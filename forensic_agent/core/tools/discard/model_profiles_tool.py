from typing import Any
from ..tools_base import ToolsBase
from ....manager.profile_manager import ProfileManager
from ....manager.datasets_manager import DatasetsManager


class ModelProfilesTool(ToolsBase):
    """查询模型画像工具"""

    # args_schema = ModelProfilesSchema

    def __init__(self, config, profile_manager: ProfileManager, datasets_manager: DatasetsManager, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.profiles_manager = profile_manager
        self.target_models = datasets_manager.expert_models

    @property
    def name(self) -> str:
        return "model_profiles"

    @property
    def description(self) -> str:
        return "Retrieve expert model profile information."

    def execute(self, **kwargs: Any):
        """执行模型画像查询（支持结构化kwargs输入）"""
        return self.profiles_manager.get_model_profiles(self.target_models)
