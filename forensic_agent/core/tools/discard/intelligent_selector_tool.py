"""
基于LLM的智能模型选择工具实现
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from ..tools_base import ToolsBase, skip_auto_register


@skip_auto_register
class IntelligentSelectorTool(ToolsBase):
    """基于LLM的智能模型选择工具"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = kwargs.get("llm_client")  # LLM客户端
        self.model_knowledge_base = self._build_model_knowledge_base()

    @property
    def name(self) -> str:
        return "llm_intelligent_model_selection"

    @property
    def description(self) -> str:
        return "基于LLM推理的智能模型选择工具，通过分析检测输入自动选择最适合的专家模型组合"

    def _execute_impl(self, image_path, **kwargs: Any) -> Dict[str, Any]:
        """执行基于LLM的智能模型选择"""
        try:
            # 解析输入参数
            detection_input = self._parse_detection_input(kwargs)

            # 准备LLM推理的输入
            llm_prompt = self._prepare_llm_prompt(detection_input, image_path)

            # 调用LLM进行模型选择推理
            llm_response = self._query_llm_for_selection(llm_prompt)

            # 解析LLM输出并生成最终选择结果
            selection_result = self._parse_llm_response(llm_response, detection_input)

            # 验证和优化选择结果
            validated_result = self._validate_and_optimize_selection(selection_result)

            logger.info(f"LLM智能选择完成: {validated_result['selected_models']}")
            return validated_result

        except Exception as e:
            # 回退到规则基础选择
            logger.warning(f"LLM选择失败，使用回退策略: {e}")
            return self._fallback_selection(kwargs)

    def _parse_detection_input(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """解析检测输入信息"""
        detection_input_str = kwargs.get("detection_input", "{}")

        if isinstance(detection_input_str, str):
            try:
                detection_input = json.loads(detection_input_str)
            except json.JSONDecodeError:
                detection_input = {}
        else:
            detection_input = detection_input_str or {}

        # 标准化输入格式
        return {
            "initial_score": detection_input.get("initial_score", 0.5),
            "embedding": detection_input.get("embedding", []),
            "clues": detection_input.get("clues", {}),
            "image_properties": detection_input.get("image_properties", {}),
            "suspected_type": detection_input.get("suspected_type", "unknown"),
            "quality_metrics": detection_input.get("quality_metrics", {}),
            "artifacts": detection_input.get("artifacts", []),
        }

    def _prepare_llm_prompt(self, detection_input: Dict[str, Any], image_path: str) -> str:
        """准备LLM推理提示"""

        prompt_template = """
作为AI图像取证专家，请根据以下信息智能选择最适合的检测模型组合：

## 可用模型信息：
{model_knowledge}

## 检测输入分析：
- 初始AI评分: {initial_score:.3f}
- 疑似类型: {suspected_type}
- 检测线索: {clues}
- 图像属性: {image_properties}
- 质量指标: {quality_metrics}
- 发现的伪影: {artifacts}

## 选择要求：
1. 根据初始评分选择合适数量的模型（高分1-2个，低分2-3个）
2. 基于疑似类型和线索匹配专长模型
3. 考虑计算资源和时间成本
4. 提供详细的选择推理

请以JSON格式回答：
{{
    "selected_models": ["model1", "model2"],
    "selection_reasoning": "详细推理过程",
    "confidence_level": "high/medium/low",
    "resource_efficiency": "资源使用评估",
    "expected_performance": "预期性能分析"
}}
"""

        return prompt_template.format(
            model_knowledge=self._format_model_knowledge(),
            initial_score=detection_input["initial_score"],
            suspected_type=detection_input["suspected_type"],
            clues=json.dumps(detection_input["clues"], ensure_ascii=False),
            image_properties=json.dumps(detection_input["image_properties"], ensure_ascii=False),
            quality_metrics=json.dumps(detection_input["quality_metrics"], ensure_ascii=False),
            artifacts=detection_input["artifacts"],
        )

    def _query_llm_for_selection(self, prompt: str) -> str:
        """查询LLM进行模型选择"""
        if not self.llm_client:
            raise RuntimeError("LLM客户端未配置")

        try:
            # 这里需要根据实际的LLM客户端接口调整
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,  # 较低的温度确保一致性
                system_message="你是一个专业的AI图像取证模型选择专家。",
            )
            return response

        except Exception as e:
            logger.error(f"LLM查询失败: {e}")
            raise

    def _parse_llm_response(self, llm_response: str, detection_input: Dict[str, Any]) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试从响应中提取JSON
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                llm_result = json.loads(json_str)
            else:
                raise ValueError("无法找到有效JSON")

            # 构建标准化结果
            result = {
                "status": "success",
                "selected_models": llm_result.get("selected_models", ["CO_SPY"]),
                "reasoning": llm_result.get("selection_reasoning", "LLM推理选择"),
                "confidence_level": llm_result.get("confidence_level", "medium"),
                "resource_efficiency": llm_result.get("resource_efficiency", "balanced"),
                "expected_performance": llm_result.get("expected_performance", "standard"),
                "selection_method": "llm_reasoning",
                "model_count": len(llm_result.get("selected_models", [])),
                "input_analysis": {
                    "initial_score": detection_input["initial_score"],
                    "suspected_type": detection_input["suspected_type"],
                    "clues_available": len(detection_input["clues"]) > 0,
                    "has_embedding": len(detection_input["embedding"]) > 0,
                },
            }

            return result

        except Exception as e:
            logger.error(f"LLM响应解析失败: {e}")
            # 生成默认结果
            return self._generate_default_llm_result(detection_input)

    def _validate_and_optimize_selection(self, selection_result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和优化选择结果"""
        selected_models = selection_result.get("selected_models", [])
        available_models = ["CO_SPY", "DRCT", "PatchShuffle", "FreqNet", "CLIPC2P"]

        # 验证模型名称
        validated_models = []
        for model in selected_models:
            if model in available_models:
                validated_models.append(model)
            else:
                # 尝试模糊匹配
                matched_model = self._fuzzy_match_model(model, available_models)
                if matched_model:
                    validated_models.append(matched_model)

        # 确保至少有一个模型
        if not validated_models:
            validated_models = ["CO_SPY"]  # 默认模型

        # 限制模型数量（最多3个）
        if len(validated_models) > 3:
            validated_models = validated_models[:3]

        # 更新结果
        selection_result["selected_models"] = validated_models
        selection_result["model_count"] = len(validated_models)

        # 添加资源使用评估
        selection_result["resource_optimization"] = self._calculate_resource_efficiency(validated_models)

        return selection_result

    def _fuzzy_match_model(self, model_name: str, available_models: List[str]) -> Optional[str]:
        """模糊匹配模型名称"""
        model_lower = model_name.lower()

        # 直接匹配
        for available in available_models:
            if model_lower in available.lower() or available.lower() in model_lower:
                return available

        # 关键词匹配
        keyword_mapping = {
            "co": "CO_SPY",
            "spy": "CO_SPY",
            "drct": "DRCT",
            "patch": "PatchShuffle",
            "shuffle": "PatchShuffle",
            "freq": "FreqNet",
            "clip": "CLIPC2P",
        }

        for keyword, model in keyword_mapping.items():
            if keyword in model_lower:
                return model

        return None

    def _calculate_resource_efficiency(self, selected_models: List[str]) -> Dict[str, Any]:
        """计算资源使用效率"""
        # 模型资源消耗估算（相对值）
        model_costs = {
            "CO_SPY": 1.0,  # 最快
            "DRCT": 1.5,  # 中等
            "PatchShuffle": 2.0,  # 较慢
            "FreqNet": 1.8,  # 较慢
            "CLIPC2P": 2.2,  # 最慢
        }

        total_cost = sum(model_costs.get(model, 1.0) for model in selected_models)
        max_possible_cost = sum(model_costs.values())

        efficiency = 1.0 - (total_cost / max_possible_cost)

        return {
            "total_cost": total_cost,
            "efficiency_ratio": efficiency,
            "estimated_speedup": max_possible_cost / total_cost,
            "resource_saved": f"{(1 - total_cost/max_possible_cost)*100:.1f}%",
        }

    def _build_model_knowledge_base(self) -> Dict[str, Any]:
        """构建模型知识库"""
        return {
            "CO_SPY": {
                "type": "CNN-based",
                "strengths": ["快速检测", "JPEG压缩鲁棒", "社交媒体图像"],
                "best_for": ["高置信度验证", "实时检测", "压缩图像"],
                "performance": "高速度，中等精度",
                "cost": "低",
            },
            "DRCT": {
                "type": "Transformer-based",
                "strengths": ["扩散模型检测", "艺术风格图像", "高质量图像"],
                "best_for": ["AI艺术检测", "扩散生成检测", "风格化图像"],
                "performance": "中等速度，高精度",
                "cost": "中等",
            },
            "PatchShuffle": {
                "type": "Patch-based",
                "strengths": ["局部篡改检测", "空间不一致", "精确定位"],
                "best_for": ["局部伪造", "低置信度案例", "空间伪影"],
                "performance": "低速度，高精度",
                "cost": "高",
            },
            "FreqNet": {
                "type": "Frequency-domain",
                "strengths": ["频域特征", "GAN检测", "微小伪影"],
                "best_for": ["GAN生成检测", "频域伪影", "微妙篡改"],
                "performance": "中等速度，高精度",
                "cost": "中等",
            },
            "CLIPC2P": {
                "type": "Vision-Language",
                "strengths": ["语义理解", "新型AIGC", "跨模态"],
                "best_for": ["新颖AI内容", "语义一致性", "多模态检测"],
                "performance": "低速度，高泛化",
                "cost": "高",
            },
        }

    def _format_model_knowledge(self) -> str:
        """格式化模型知识为文本"""
        formatted = []
        for model, info in self.model_knowledge_base.items():
            formatted.append(f"**{model}** ({info['type']})")
            formatted.append(f"  - 优势: {', '.join(info['strengths'])}")
            formatted.append(f"  - 适用: {', '.join(info['best_for'])}")
            formatted.append(f"  - 性能: {info['performance']}")
            formatted.append(f"  - 成本: {info['cost']}")
            formatted.append("")
        return "\n".join(formatted)

    def _generate_default_llm_result(self, detection_input: Dict[str, Any]) -> Dict[str, Any]:
        """生成默认LLM结果（当LLM失败时）"""
        initial_score = detection_input["initial_score"]
        suspected_type = detection_input["suspected_type"]

        # 基于规则的回退选择
        if initial_score >= 0.8:
            selected_models = ["CO_SPY"]
            reasoning = "高置信度，使用快速验证模型"
        elif initial_score >= 0.6:
            selected_models = ["CO_SPY", "DRCT"]
            reasoning = "中等置信度，使用平衡模型组合"
        else:
            selected_models = ["PatchShuffle", "DRCT", "CO_SPY"]
            reasoning = "低置信度，使用全面分析模型组合"

        # 根据疑似类型调整
        if suspected_type == "diffusion":
            selected_models = ["DRCT", "CO_SPY"]
        elif suspected_type == "gan":
            selected_models = ["FreqNet", "PatchShuffle"]

        return {
            "status": "success",
            "selected_models": selected_models,
            "reasoning": reasoning + " (规则回退)",
            "confidence_level": "medium",
            "resource_efficiency": "balanced",
            "expected_performance": "standard",
            "selection_method": "rule_based_fallback",
            "model_count": len(selected_models),
            "input_analysis": {
                "initial_score": initial_score,
                "suspected_type": suspected_type,
                "clues_available": len(detection_input["clues"]) > 0,
                "has_embedding": len(detection_input["embedding"]) > 0,
            },
        }

    def _fallback_selection(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """完全回退选择策略"""
        try:
            detection_input = self._parse_detection_input(kwargs)
            return self._generate_default_llm_result(detection_input)
        except Exception as e:
            logger.error(f"回退选择也失败: {e}")
            return {
                "status": "error",
                "selected_models": ["CO_SPY"],  # 最基本的回退
                "reasoning": f"所有选择策略失败，使用默认模型: {e}",
                "confidence_level": "low",
                "resource_efficiency": "minimal",
                "expected_performance": "basic",
                "selection_method": "emergency_fallback",
                "model_count": 1,
                "error": str(e),
            }

    def supports_batch(self) -> bool:
        """是否支持批处理"""
        return True

    def get_resource_requirements(self) -> Dict[str, Any]:
        """获取资源需求"""
        return {
            "memory": "low",  # LLM推理内存需求低
            "compute": "medium",  # 需要LLM推理
            "storage": "minimal",
            "network": "required",  # 可能需要访问LLM服务
        }
