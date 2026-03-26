from typing import Any
import pandas as pd

from .tools_base import ToolsBase
from ...processor import ClusterAnalysisProcessor
from ...manager.profile_manager import ProfileManager
from ...manager.datasets_manager import DatasetsManager

from ...utils import create_chat_llm
from loguru import logger


class ClusteringAnalysisTool(ToolsBase):
    def __init__(
        self,
        config,
        tools_llm,
        profile_manager: ProfileManager,
        datasets_manager: DatasetsManager,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        if config.get("llm_config", None):
            # 使用配置文件中的 LLM 设置
            self.tools_llm = create_chat_llm(config["llm_config"])
        else:
            self.tools_llm = tools_llm  # 使用传入的 LLM 实例
        self.cluster_analysis = ClusterAnalysisProcessor(config, prompt_path=config["prompt_path"], tools_llm=self.tools_llm)
        self.profile_manager = profile_manager
        self.clustering_data = datasets_manager.clustering_data

    @property
    def name(self) -> str:
        return "clustering_analysis"

    @property
    def description(self) -> str:
        return "Conduct a qualitative evaluation and provide a descriptive analysis of the effectiveness of the clustering results."

    def get_cluster_result(self, image_path):
        """执行特征提取（重命名为execute以匹配接口；原run逻辑不变）"""
        logger.debug(f"提取图像簇特征: {image_path}")
        # 查找匹配的记录
        matched_rows: pd.DataFrame = self.clustering_data[self.clustering_data["image_path"] == image_path]
        if matched_rows.empty:
            logger.error(f"未找到匹配的聚类簇信息: {image_path}")
            return {"error": "未找到匹配的聚类簇信息"}
        # 返回第一条匹配记录
        result = matched_rows.iloc[0].to_dict()
        # 删除image_path字段，避免冗余
        result.pop("image_path", None)
        # 将数值转为整数
        for key, value in result.items():
            if isinstance(value, (int, float)):
                result[key] = int(value)
        logger.debug(f"成功提取聚类簇信息: {result}")

        return result

    @staticmethod
    def deep_merge(dest: dict, src: dict):
        """将 src 合并入 dest，遇到 dict -> 递归合并；否则用 src 覆盖 dest"""
        for k, v in src.items():
            if k in dest and isinstance(dest[k], dict) and isinstance(v, dict):
                ClusteringAnalysisTool.deep_merge(dest[k], v)
            else:
                dest[k] = v

    def execute(self, **kwargs: Any):
        """执行语义伪造推断"""
        params = self.args_schema.model_validate(kwargs)
        image_path = params.get_image_path()
        clustering_result = self.get_cluster_result(image_path)
        clustering_analysis = self.profile_manager.get_clustering_analysis(clustering_result)
        cluster_profile = self.profile_manager.get_clustering_performance(clustering_result)
        
        # 是否加入各个簇的性能指标
        self.deep_merge(dict(clustering_analysis), dict(cluster_profile))
        return clustering_analysis
