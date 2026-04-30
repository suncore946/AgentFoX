"""Semantic analysis tool.

中文说明: 该工具读取/生成单张图片的语义鉴伪描述, 是最小 test 流程的唯一默认工具。
English: This tool loads or generates semantic forensic descriptions for one
image and is the only default tool in the minimal test flow.
"""

from __future__ import annotations

from typing import Any

from ...manager.datasets_manager import DatasetsManager
from ...manager.image_manager import ImageManager
from ...manager.profile_manager import ProfileManager
from ...processor.image_feat_processor import ImageFeatProcessor
from ...utils import create_chat_llm
from .tools_base import ToolsBase


class SemanticAnalysisTool(ToolsBase):
    """Analyze semantic-level AIGC clues.

    中文说明: 如果缓存中已有 semantic profile, 直接复用; 否则调用 VLM 生成并写入缓存。
    English: If a semantic profile already exists in cache, it is reused;
    otherwise the VLM generates one and the result is cached.
    """

    def __init__(
        self,
        config: dict,
        image_manager: ImageManager,
        profile_manager: ProfileManager,
        tools_llm,
        datasets_manager: DatasetsManager | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.tools_llm = create_chat_llm(config["llm_config"]) if config.get("llm_config") else tools_llm
        self.image_manager = image_manager
        self.profile_manager = profile_manager
        self.datasets_manager = datasets_manager
        self.feat_processor = ImageFeatProcessor(config, image_manager, semantic_llm=self.tools_llm)

    @property
    def name(self) -> str:
        return "semantic_analysis"

    @property
    def description(self) -> str:
        return (
            "Analyze semantic-level image clues and return observations, detected anomalies, "
            "and a binary AIGC prediction."
        )

    def _run(self, image_path: str) -> dict[str, Any]:
        """Run semantic analysis for one image.

        中文说明: 该方法是工具核心逻辑, execute 只负责解析 LangGraph 注入参数。
        English: This method contains the tool core; execute only parses
        LangGraph-injected parameters.
        """
        self.profile_manager.ensure_profile_paths(datasets_manager=self.datasets_manager, kind="semantic")
        semantic_profile = self.profile_manager.get_semantic_profile(image_path=image_path)
        if not semantic_profile:
            self.logger.info(f"Extracting semantic profile for image: {image_path}")
            semantic_profile, _ = self.feat_processor.run(image_path=image_path)
            self.profile_manager.save_runtime_semantic_profile(
                image_path=image_path,
                semantic_profile=semantic_profile,
                datasets_manager=self.datasets_manager,
            )

        return {
            "semantic_observations": semantic_profile["observations"],
            "semantic_detected_anomalies": semantic_profile["detected_anomalies"],
            "semantic_result": int(semantic_profile["pred_label"]),
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Tool execution entrypoint.

        中文说明: image_path 从 LangGraph state 注入, 用户无需手动传递。
        English: image_path is injected from LangGraph state, so users do not
        need to provide it manually.
        """
        params = self.args_schema.model_validate(kwargs)
        return self._run(image_path=params.get_image_path())
