# if self.target_models:
#     tmp = {}
#     if "overall_performance" in cluster_profiles:
#         for performance in cluster_profiles["overall_performance"]["performance"]:
#             if performance in self.target_models:
#                 tmp[performance] = cluster_profiles["overall_performance"]["performance"][performance]
#         cluster_profiles["overall_performance"]["performance"] = tmp

from collections import defaultdict
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
import pandas as pd

from .base_manager import BaseManager
from ..core.core_exceptions import ConfigurationError


@dataclass
class DecisionRecommendation:
    """决策推荐"""

    primary: str
    secondary: str


@dataclass
class DecisionScenario:
    """决策场景"""

    scenario: str
    environment_keywords: List[str]
    aigc_focus: List[str]
    recommendations: DecisionRecommendation
    justification: str


class ProfileManager(BaseManager):
    """画像管理器"""

    def __init__(self, models_config: dict):
        """
        初始化模型画像管理器

        Args:
            models_config: 模型配置字典
        """
        self.config = models_config
        self.logger = logger
        self.target_models = self.config.get("expert_models", None)

        self._decision_scenarios: List[DecisionScenario] = []

        # 从配置获取画像文件路径
        default_profiles_path = Path(__file__).parent.parent / "configs" / "model_profiles.json"
        self.model_profiles_path = Path(models_config.get("model_profiles", default_profiles_path))
        self.clustering_performance_profiles_path = Path(models_config["clustering_performance_profiles"])
        self.clustering_analysis_profiles_path = Path(models_config["clustering_analysis_profiles"])
        self.semantic_profiles_path = self.config.get("semantic_profiles", None)
        self.expert_analysis_path = self.config.get("expert_analysis", None)

        if self.semantic_profiles_path:
            self.semantic_profiles_path = Path(self.semantic_profiles_path)
        self.calibration_profiles_path = Path(self.config.get("calibration_profiles", ""))

        self._clustering_performance = self._load_clustering_profiles(self.clustering_performance_profiles_path)
        self._clustering_analysis = self._load_clustering_profiles(self.clustering_analysis_profiles_path)
        self._model_profiles = self._load_model_profiles()
        self._semantic_profiles = self._load_semantic_profiles()
        self._calibration_profiles = self._load_calibration_profiles()
        self._expert_analysis = self._load_expert_analysis()

    @property
    def calibration_profiles(self) -> Dict[str, Any]:
        """校准画像数据"""
        target_models = set(self.target_models or [])

        def should_include(model_name: str) -> bool:
            return not target_models or model_name in target_models

        ece = {
            model_name: ece_data.get("ece_bootstrap_ci", {})
            for model_name, ece_data in self._calibration_profiles.get("calibrations_data", {}).items()
            if should_include(model_name)
        }

        analysis = self._calibration_profiles.get("calibrations_analysis", {})

        quality = {
            model_name: quality_data
            for model_name, quality_data in self._calibration_profiles.get("calibrations_quality", {}).items()
            if should_include(model_name)
        }

        return {
            "ece_bootstrap_ci": ece,
            "quality": quality,
            "analysis": analysis,
        }

    @property
    def semantic_profiles(self) -> Dict[str, Any]:
        """语义画像数据"""
        return self._semantic_profiles

    @property
    def expert_analysis(self) -> pd.DataFrame:
        """专家分析数据"""
        return self._expert_analysis

    def get_model_profiles(self, model_names) -> Dict[str, Dict]:
        """所有模型画像"""
        if isinstance(model_names, str):
            model_names = [model_names]

        if "calibration_note" in model_names:
            model_names.remove("calibration_note")

        if self.target_models:
            assert all(
                name in self.target_models for name in model_names
            ), f"请求的模型画像 {model_names} 不在目标模型列表 {self.target_models} 中"
        # 根据model_names列表返回对应的画像, 如果不存在则返回空字典
        # ret = {
        #     "model_profiles": {name: self._model_profiles.get(name, {}) for name in model_names},
        #     "model_evaluation": self._model_profiles.get("model_evaluation", {}),
        # }
        # return ret
        return {name: self._model_profiles.get(name, {}) for name in model_names}

    def get_clustering_overall_performance(self) -> Dict[str, Dict]:
        return self._clustering_performance["overall_performance"]["performance"]

    def get_clustering_analysis(self, cluster_names: Dict[str, List[int]]) -> Dict[str, Dict]:
        """集群画像配置"""
        ret = defaultdict(dict)
        for cluster_name, indices in cluster_names.items():
            if cluster_name.lower() not in self._clustering_analysis:
                self.logger.debug(f"集群 '{cluster_name}' 的画像数据不存在")
                continue
            cluster_data = self._clustering_analysis[cluster_name.lower()]

            # 如果indices不是可迭代对象, 则转为列表
            if not isinstance(indices, (list, tuple)):
                indices = [indices]

            for index in indices:
                # 简化索引名称生成逻辑
                if isinstance(index, int):
                    name = f"cluster_{index}"
                elif isinstance(index, str) and "_" in index:
                    name = f"cluster_{index.split('_')[-1]}"
                else:
                    # 处理其他情况
                    name = f"cluster_{str(index)}"

                profile_data = cluster_data.get("cluster_profile", {}).get(name, None)
                ret[cluster_name][name] = profile_data
                # clustering_analysis_data = cluster_data.get("comprehensive_analysis", None)
                # if clustering_analysis_data is None or profile_data is None:
                #     raise ConfigurationError(f"集群 '{cluster_name}' 中索引 '{name}' 的综合分析数据或画像数据不存在")
                # ret[cluster_name]["clustering_analysis"] = clustering_analysis_data
            ret[cluster_name]["overall_analysis"] = self._clustering_analysis["overall_analysis"]["clustering_analysis"][cluster_name]
        return dict(ret)

    def get_clustering_performance(self, cluster_names: Dict[str, List[int]]) -> Dict[str, Dict]:
        """集群画像配置"""
        ret = defaultdict(dict)
        for cluster_name, indices in cluster_names.items():
            if cluster_name.lower() not in self._clustering_performance:
                self.logger.debug(f"集群 '{cluster_name}' 的画像数据不存在")
                continue
            cluster_data = self._clustering_performance[cluster_name.lower()]
            # 如果indices不是可迭代对象, 则转为列表
            if not isinstance(indices, (list, tuple)):
                indices = [indices]

            for index in indices:
                # 简化索引名称生成逻辑
                if isinstance(index, int):
                    name = f"cluster_{index}"
                elif isinstance(index, str) and "_" in index:
                    name = f"cluster_{index.split('_')[-1]}"
                else:
                    # 处理其他情况
                    name = f"cluster_{str(index)}"

                ranking_info = cluster_data["forensic_expert_cluster_performance"][name]
                if self.target_models:
                    # 基于self.target_models过滤排名信息
                    ranking_info = {model: performance for model, performance in ranking_info.items() if model in self.target_models}
                ret[cluster_name] = {
                    "cluster_name": name,
                    "Cluster_rounding_experts_performance_ranking": ranking_info,
                    # "clustering_quality_metrics": cluster_data.get("clustering_quality_metrics", {}),
                }

        return ret

    def _load_clustering_profiles(self, target_path) -> Dict[str, Dict]:
        """获取集群画像配置"""
        ret = {}
        with open(target_path, "r", encoding="utf-8") as f:
            cluster_profiles = json.load(f)
        # 所有的key转为小写，方便后续查询
        for clustering_name, profiles in cluster_profiles.items():
            # 对profiles进行处理, 将list转换为dict, key为cluster_name
            if isinstance(profiles, list):
                profile_dict = {}
                for profile in profiles:
                    cluster_name = profile.get("cluster_name")
                    if cluster_name:
                        profile_dict[cluster_name] = profile
                ret[clustering_name.lower()] = profile_dict
            elif isinstance(profiles, dict):
                # 已经是dict，直接存储
                ret[clustering_name.lower()] = profiles
            else:
                self.logger.warning(f"未知的profiles类型: {type(profiles)}，跳过 {clustering_name}")
        return ret

    def _load_model_profiles(self) -> Dict[str, Dict]:
        """加载模型画像数据"""
        ret = {}
        with open(self.model_profiles_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
        # 基于self.target_models过滤模型画像
        if self.target_models:
            data["profiles"] = [p for p in data.get("profiles", []) if p.get("model") in self.target_models]
        # 解析模型画像
        for profile_data in data.get("profiles", []):
            model_name = profile_data.pop("model", None)
            ret[model_name] = profile_data
        ret["model_evaluation"] = data.get("model_evaluation", {})
        return ret

    def _load_semantic_profiles(self) -> Dict[str, Any]:
        """加载语义画像数据"""
        if self.semantic_profiles_path and self.semantic_profiles_path.exists():
            with open(self.semantic_profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        self.logger.warning(f"语义画像文件不存在: {self.semantic_profiles_path}, 跳过加载语义画像")
        return {}

    def _load_calibration_profiles(self) -> Dict[str, Any]:
        """加载校准画像数据"""
        if not self.calibration_profiles_path.exists():
            self.logger.warning(f"校准画像文件不存在: {self.calibration_profiles_path}, 跳过加载校准画像")
            return {}
        with open(self.calibration_profiles_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_expert_analysis(self) -> pd.DataFrame:
        """加载专家分析数据"""
        if not self.expert_analysis_path or not Path(self.expert_analysis_path).exists():
            self.logger.warning(f"专家分析文件不存在: {self.expert_analysis_path}, 跳过加载专家分析")
            return pd.DataFrame()
        df = pd.read_csv(self.expert_analysis_path)
        return df

    def _parse_decision_scenario(self, data: Dict[str, Any]) -> DecisionScenario:
        """解析决策场景数据"""
        recommendations_data = data.get("recommendations", {})
        recommendations = DecisionRecommendation(
            primary=recommendations_data.get("primary", ""),
            secondary=recommendations_data.get("secondary", ""),
        )

        return DecisionScenario(
            scenario=data.get("scenario", ""),
            environment_keywords=data.get("environment_keywords", []),
            aigc_focus=data.get("AIGC_focus", []),
            recommendations=recommendations,
            justification=data.get("justification", ""),
        )

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)
