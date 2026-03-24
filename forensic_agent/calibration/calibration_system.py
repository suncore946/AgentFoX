"""
模型校准系统
整合置信度校准优化器和系统主入口，提供完整的校准流程
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from loguru import logger
from .calibration_methods import CalibrationMethods
from .calibration_evaluator import CalibrationEvaluator
from ..configs.config_dataclass import CalibrationConfig
from ..processor.expert_profiles_processor import ExpertProfilesProcessor


class CalibrationSystem:
    """
    模型校准系统主类

    核心问题: 模型输出的预测概率与实际准确率不匹配
    目标: 通过校准方法修正模型的过自信问题，使预测概率更接近实际准确率

    功能:
    - 协调所有子模块
    - 提供完整的校准流程
    - 自动选择最佳校准方法
    - 批量处理多个模型的校准
    - 多模型集成预测校准
    - 应用校准参数到新数据
    - 生成模型画像
    - 提供便捷的API接口
    """

    # 常量定义
    DEFAULT_REQUIRED_COLUMNS = ["pred_label", "gt_label"]
    MODEL_REQUIRED_COLUMNS = ["model_name", "pred_label", "gt_label"]

    def __init__(self, calibration_df, test_df, val_df, config: CalibrationConfig, group_columns: str | List[str] | None = "model_name"):
        """
        初始化校准系统

        Args:
            calibration_df: 校准集数据
            test_df: 测试集数据
            val_df: 验证集数据
            config: 校准配置，如果为None则使用默认配置
            group_columns: 分组列名，用于多模型校准
        """
        self.config = config

        # 初始化数据
        self.calibration_df = calibration_df
        self.test_df = test_df
        self.val_df = val_df

        # 支持 group_columns 为 None（全局性能）
        if group_columns is None:
            self.group_columns = None
        else:
            self.group_columns = [group_columns] if isinstance(group_columns, str) else group_columns

        # 初始化子模块
        self.calibration_methods = CalibrationMethods(self.config)
        self.calibration_evaluator = CalibrationEvaluator(self.config)
        self.expert_profiles_processor = ExpertProfilesProcessor(self.config.llm, prompt_path=self.config.model_profile_prompt_path)

        logger.info("模型校准系统初始化完成")

    def select_best_calibration(self, model_df: pd.DataFrame, target_column="pred_label") -> Dict[str, Any]:
        """自动选择最佳校准方法"""
        logger.info("开始校准方法选择...")

        # 处理全局校准的情况
        if self.group_columns is None:
            logger.info("执行全局校准")
            pred_probs = model_df[target_column].values
            gt_labels = model_df["gt_label"].values
            original_metrics = self.calibration_methods.calculate_metrics(gt_labels, pred_probs)

            # 尝试所有校准方法
            calibration_details = []
            method_ece_scores = []

            for method_name in self.config.calibration_methods:
                result = self.calibration_methods.execute_calibration(method_name, pred_probs, gt_labels)
                if result and result.get("success", False):
                    result["original_metrics"] = original_metrics
                    calibration_details.append(result)
                    method_ece_scores.append(result["calibration_metrics"].get("ece", np.inf))

            if not calibration_details:
                raise ValueError("没有成功执行任何校准方法")

            best_result = min(
                calibration_details,
                key=lambda x: (
                    x["calibration_metrics"].get("ece", np.inf),
                    x["calibration_metrics"].get("brier_score", np.inf),
                    x["calibration_metrics"].get("f1_score", np.inf),
                ),
            )
            print("best_result:", best_result)
            print("选择的校准方法:", best_result.get("method_name", "未知"))
            # 计算统计指标
            comparison = {
                "selected_ece": round(float(best_result["calibration_metrics"].get("ece", np.inf)), 5),
                "avg_method_ece": round(float(np.mean(method_ece_scores)), 5),
                "best_method_ece": round(float(min(method_ece_scores)), 5),
                "worst_method_ece": round(float(max(method_ece_scores)), 5),
            }
            logger.info(f"全局校准方法对比: {comparison}")
            return {"global": {"best_result": best_result, "calibration_details": calibration_details}}

        # 分组进行校验
        final_result = {}
        for name, group in model_df.groupby(self.group_columns):
            pred_probs = group[target_column].values
            gt_labels = group["gt_label"].values
            original_metrics = self.calibration_methods.calculate_metrics(gt_labels, pred_probs)

            # 尝试所有校准方法
            calibration_details = []
            method_ece_scores = []

            for method_name in self.config.calibration_methods:
                result = self.calibration_methods.execute_calibration(method_name, pred_probs, gt_labels)
                if result and result.get("success", False):
                    result["original_metrics"] = original_metrics
                    calibration_details.append(result)
                    method_ece_scores.append(result["calibration_metrics"].get("ece", np.inf))
            if not calibration_details:
                raise ValueError("没有成功执行任何校准方法")

            best_result = min(
                calibration_details,
                key=lambda x: (
                    x["calibration_metrics"].get("ece", np.inf),
                    x["calibration_metrics"].get("brier_score", np.inf),
                    x["calibration_metrics"].get("f1_score", np.inf),
                ),
            )

            # 计算统计指标
            comparison = {
                "selected_ece": round(float(best_result["calibration_metrics"].get("ece", np.inf)), 5),
                "avg_method_ece": round(float(np.mean(method_ece_scores)), 5),
                "best_method_ece": round(float(min(method_ece_scores)), 5),
                "worst_method_ece": round(float(max(method_ece_scores)), 5),
            }
            final_result[name] = {"best_result": best_result, "calibration_details": calibration_details}
            logger.info(f"分组 [{name}]校准方法对比: {comparison}")
        return final_result

    def models_calibration(self, required_columns: List[str] = None, target_column="pred_prob") -> Dict[str, Any]:
        """
        模型预测结果置信度校准， 在校准集上学习校准参数，在验证集上评估校准效果

        Args:
            calibration_df: 校准集DataFrame，用于学习校准参数
            test_df: 校准集DataFrame，用于评估校准效果
            required_columns: 必需的列名

        Returns:
            Dict: 校准结果，包含效果验证和模型排名
        """
        # 验证输入数据
        if required_columns is None:
            required_columns = self.MODEL_REQUIRED_COLUMNS

        logger.info("阶段1: 在校准集上学习校准参数")
        logger.info(f"校准集记录数: {len(self.calibration_df)}")
        if self.group_columns:
            logger.info(f"按分组列 {self.group_columns} 进行校准参数学习")
        else:
            logger.info("进行全局校准参数学习")

        # 第二阶段：在测试集上应用校准参数并评估
        logger.info("阶段2: 在测试集上应用校准参数并评估")

        calibration = self.select_best_calibration(self.calibration_df, target_column)
        calibrated_group, _ = self.calibrate(calibration, self.test_df, "test")
        self.test_df = calibrated_group

        if self.val_df.empty:
            logger.warning("验证集为空，跳过验证集评估")
            val_report = {}
        else:
            logger.info("在验证集上评估校准效果")
            val_report = {}
            # val_report = calibrate(val_df, "val")
        return val_report

    def calibrate(self, calibration, target_df: pd.DataFrame, calibrate_type="test", target_column="pred_prob"):
        """
        应用校准参数并评估性能
        Args:
            target_df: 目标数据集
            calibrate_type: 校准类型 ("test" 或 "val")
        Returns:
            final_calibrated_df: 合并后的完整校准数据集
            assessment_report: 校准评估报告
        """
        assessment_report = {}
        calibrated_dfs_list = []  # 1. 初始化列表用于存储每个分组的df
        # 注意：这里假设 self.group_columns, self.calibration, self.target_column 等都在类属性中
        # 如果 target_column 是参数传入的，请确保函数签名中有它

        for group_key, group_data in target_df.groupby(self.group_columns):
            group_name = "*".join(map(str, group_key)) if isinstance(group_key, tuple) else str(group_key)

            # 假设 calibration 是类属性 self.calibration 或者在外部定义的
            calibration_params = calibration[group_key]["best_result"]
            # 应用校准参数 (得到当前分组的结果)
            # 注意：这里使用一个临时变量 current_calibrated_group
            current_calibrated_group = self.calibration_methods.apply_calibration_params(
                group_data, calibration_params, target_column=target_column
            )

            # 2. 将当前分组结果添加到列表中
            calibrated_dfs_list.append(current_calibrated_group)
            if calibrate_type == "val":
                # 评估校准前后的性能 (使用当前分组的数据 current_calibrated_group)
                original_metrics = self.calibration_evaluator.evaluate_groups_performance(
                    current_calibrated_group, target_column=target_column, group_columns="model_name"
                )
                calibrated_metrics = self.calibration_evaluator.evaluate_groups_performance(
                    current_calibrated_group, target_column="calibration_prob", group_columns="model_name"
                )
                assessment_report[group_name] = self.calibration_evaluator.evaluate_calibration(
                    group_name,
                    original_result=original_metrics[group_name],
                    calibrated_result=calibrated_metrics[group_name],
                    model_df=group_data,
                )
            else:
                assessment_report[group_name] = "测试集不评估校准结果"
        # 3. 循环结束后，将列表合并为一个完整的 DataFrame
        if calibrated_dfs_list:
            final_calibrated_df = pd.concat(calibrated_dfs_list, axis=0, ignore_index=True)
        else:
            # 防止传入空df导致报错
            final_calibrated_df = pd.DataFrame()
        return final_calibrated_df, assessment_report

    def calibration_profile(self, val_metrics: Dict) -> Dict[str, Any]:
        """
        生成模型画像和排名分析

        Args:
            val_df: 包含 model_name, gt_label, pred_prob, calibration_prob 的数据框
            val_metrics: 包含各模型详细指标的字典

        Returns:
            包含排名分析和模型画像的字典
        """
        logger.info("生成模型画像和排名...")

        # 提取模型名称
        model_names = list(val_metrics.keys())

        if not model_names:
            logger.warning("没有找到模型数据")
            return {"error": "No models found"}

        # ==================== 1. 排名 ====================
        original_models_data = []
        calibrated_models_data = []
        for model_name, metrics in val_metrics.items():
            # 收集原始和校准后的整体指标
            original_models_data.append(metrics["original_result"]["metrics"] | {"model_name": model_name})
            calibrated_models_data.append(metrics["calibration_result"]["metrics"] | {"model_name": model_name})

        original_df = pd.DataFrame(original_models_data)
        calibrated_df = pd.DataFrame(calibrated_models_data)

        # ==================== 2. 显著性排名 ====================
        original_significance = self.calibration_evaluator.evaluate_significance_rank(original_df)
        calibrated_significance = self.calibration_evaluator.evaluate_significance_rank(calibrated_df)

        # ==================== 3. 生成整体洞察 ====================
        insights = {
            "original_significance": original_significance,
            "calibrated_significance": calibrated_significance,
        }
        analysis = self.expert_profiles_processor.run(model_data=insights)
        insights["model_analysis"] = analysis
        return insights

    def run(self, target_column="pred_prob", out_report=False) -> Dict[str, Any]:
        """
        运行完整的校准流程

        Args:
            target_column: 目标预测概率列名

        Returns:
            包含所有结果的字典
        """
        logger.info("开始运行完整的校准流程...")

        try:
            # 步骤: 校准参数计算
            logger.info("=== 校准参数计算 ===")
            calibrations_report = self.models_calibration(
                required_columns=["model_name", "image_path", target_column],
                target_column=target_column,
            )

            # 步骤2: 对模型进行排名
            if out_report:
                # 模型排名计算
                logger.info("=== 模型排名计算 ===")
                profile_info = self.calibration_profile(calibrations_report)
            else:
                print("没有校准结果可用于生成模型画像")
                return {}

            logger.info("完整校准流程执行完成")
            return {
                "calibrations_quality": calibrations_report,
                "calibrations_profile": profile_info,
            }

        except Exception as e:
            logger.error(f"校准流程执行失败: {e}")
            raise e
