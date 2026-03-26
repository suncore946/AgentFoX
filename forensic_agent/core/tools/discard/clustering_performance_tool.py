from typing import Any
import pandas as pd

from ..tools_base import ToolsBase, skip_auto_register
from ....manager.profile_manager import ProfileManager
from ....manager.datasets_manager import DatasetsManager

from loguru import logger


class ClusteringPerformanceTool(ToolsBase):
    def __init__(
        self,
        config,
        profile_manager: ProfileManager,
        datasets_manager: DatasetsManager,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.profile_manager = profile_manager
        self.clustering_data = datasets_manager.clustering_data

    @property
    def name(self) -> str:
        return "clustering_performance"

    @property
    def description(self) -> str:
        """
        该工具可以查询当前图片在不同图像特征（如不同特征提取方法）下进行聚类后对应的簇结果。以及在不同取证模型在训练集中对应簇结果图片的预测性能的说明与全体训练集的整体预测性能说明。
        """
        return "This tool can query the clustering results of the current image under different image features (such as different feature extraction methods), as well as provide descriptions of the prediction performance of different forensic models on the images in the corresponding clusters within the training set, and an overall description of the prediction performance on the entire training set."

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
                result[key] = f"cluster_{int(value)}"
        logger.debug(f"成功提取聚类簇信息: {result}")
        return result

    def execute(self, **kwargs: Any):
        """执行语义伪造推断"""
        params = self.args_schema.model_validate(kwargs)
        image_path = params.get_image_path()
        clustering_result = self.get_cluster_result(image_path)
        cluster_profile, clustering_desc = self.profile_manager.get_clustering_performance(clustering_result)
        return {
            "clustering_result": clustering_result,
            "clustering_description": clustering_desc,
            "cluster_performance": cluster_profile,
            "overall_performance": self.profile_manager.get_clustering_overall_performance(),
        }
