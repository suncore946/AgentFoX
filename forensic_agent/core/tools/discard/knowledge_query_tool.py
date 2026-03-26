"""
知识库查询工具实现
"""

from typing import Dict, Any, List
from loguru import logger
from ..tools_base import ToolsBase


class KnowledgeQueryTool(ToolsBase):
    """知识库查询工具"""

    @property
    def name(self) -> str:
        return "query_knowledge_base"

    @property
    def description(self) -> str:
        return "在案例知识库中搜索相似的历史案例"

    def validate_input(self, *args, **kwargs: Any) -> bool:
        """验证输入参数"""
        top_k = kwargs.get("top_k", 3)

        if not isinstance(top_k, int) or top_k <= 0:
            logger.error("top_k 参数必须是正整数")
            return False

        return True

    def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        """执行知识库查询"""
        top_k = kwargs.get("top_k", 3)
        query_features = kwargs.get("features")  # 可选的特征向量

        try:
            # 模拟历史案例数据 - 实际应用中应该查询真实数据库
            historical_cases = [
                {
                    "case_id": "case_001",
                    "image_type": "social_media",
                    "classification": "FULL_AI_GENERATED",
                    "successful_models": ["CO_SPY"],
                    "confidence": 0.89,
                    "features_similarity": 0.85,
                    "context": "社交媒体压缩图像",
                },
                {
                    "case_id": "case_002",
                    "image_type": "high_quality",
                    "classification": "FULL_AI_GENERATED",
                    "successful_models": ["DRCT"],
                    "confidence": 0.84,
                    "features_similarity": 0.78,
                    "context": "高质量AI生成图像",
                },
                {
                    "case_id": "case_003",
                    "image_type": "mixed_quality",
                    "classification": "PARTIAL_AI_GENERATED",
                    "successful_models": ["PatchShuffle"],
                    "confidence": 0.76,
                    "features_similarity": 0.72,
                    "context": "部分AI编辑图像",
                },
                {
                    "case_id": "case_004",
                    "image_type": "natural",
                    "classification": "REAL",
                    "successful_models": ["CO_SPY", "DRCT"],
                    "confidence": 0.91,
                    "features_similarity": 0.68,
                    "context": "自然拍摄图像",
                },
            ]

            # 按相似度排序并返回前top_k个
            sorted_cases = sorted(historical_cases, key=lambda x: x["features_similarity"], reverse=True)
            similar_cases = sorted_cases[:top_k]

            # 生成推荐
            recommendation = self._generate_recommendation(similar_cases)

            result = {
                "status": "success",
                "similar_cases": similar_cases,
                "recommendation": recommendation,
                "confidence_boost": 0.1,  # 知识库匹配带来的置信度提升
                "total_cases_found": len(historical_cases),
            }

            logger.info(f"找到 {len(similar_cases)} 个相似案例，推荐模型: {recommendation['model']}")
            return result

        except Exception as e:
            raise RuntimeError(f"知识库查询失败: {e}")

    def _generate_recommendation(self, similar_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于相似案例生成推荐"""
        if not similar_cases:
            return {"model": "CO_SPY", "reasoning": "无相似案例，使用默认模型", "confidence": 0.5}

        # 统计最成功的模型
        model_success = {}
        for case in similar_cases:
            for model in case["successful_models"]:
                weight = case["features_similarity"] * case["confidence"]
                model_success[model] = model_success.get(model, 0) + weight

        # 选择权重最高的模型
        best_model = max(model_success.items(), key=lambda x: x[1])

        return {
            "model": best_model[0],
            "reasoning": f"基于 {len(similar_cases)} 个相似案例，该模型成功率最高",
            "confidence": min(0.9, best_model[1] / len(similar_cases)),
            "alternative_models": list(model_success.keys()),
        }
