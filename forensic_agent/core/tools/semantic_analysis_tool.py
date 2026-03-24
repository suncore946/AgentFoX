from typing import Any
from ...processor.detection_processor import DetectionProcessor
from ...manager.feature_manager import FeatureManager
from ...manager.image_manager import ImageManager
from ...manager.profile_manager import ProfileManager
from .tools_base import ToolsBase


class SemanticAnalysisTool(ToolsBase):
    """特征提取工具"""

    def __init__(self, config, feature_manager, image_manager, profile_manager: ProfileManager, tools_llm, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.feature_manager: FeatureManager = feature_manager
        self.image_manager: ImageManager = image_manager
        self.semantic_profiles = profile_manager.semantic_profiles
        self.detection_processor = DetectionProcessor(llm=tools_llm, prompt_path=config["prompt_path"])

    @property
    def name(self) -> str:
        return "semantic_analysis"

    @property
    def description(self) -> str:
        return "Analyzes semantic-level clues and provides conclusions about whether an image is AI-generated based on semantic features and anomalies"

    def execute(self, **kwargs: Any):
        """执行特征提取（重命名为execute以匹配接口；原run逻辑不变）"""
        params = self.args_schema.model_validate(kwargs)
        image_path = params.get_image_path()  # 确保image_path存在
        semantic_profile = self.semantic_profiles.get(image_path, None)

        if not semantic_profile:
            self.logger.warning(f"Extracting semantic features for image: {image_path}")
            semantic_profile, _ = self.feature_manager.run(image_path=image_path)

        ret = {
            "semantic_observations": semantic_profile["observations"],
            "semantic_detected_anomalies": semantic_profile["detected_anomalies"],
            "semantic_result": semantic_profile["pred_label"],
        }
        return ret
