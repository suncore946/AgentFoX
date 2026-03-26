"""
置信度校准系统
实现温度缩放(Temperature Scaling)和ECE计算
基于方案第4-5节的要求
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import pandas as pd

from ..calibration.calibration_methods import CalibrationMethods
from .base_manager import BaseManager


@dataclass
class CalibrationResult:
    """校准结果"""

    temperature: float
    calibrated_probabilities: np.ndarray
    original_ece: float
    calibrated_ece: float
    success: bool
    error: Optional[str] = None


class CalibrationManager(BaseManager):
    """多模型校准管理器"""

    def __init__(self, config: dict = None):
        self.logger = logger
        self.config = config if config else {}

        self.calibrator = CalibrationMethods(self.config.get("ece_bins", 20))
        self.calibration_file = self.config.get(
            "calibration_file_path", Path(__file__).parent.parent / "configs" / "calibration_result.json"
        )
        self.model_calibrations: Dict[str, float] = {}

        self.load_calibration_data()

    def run(self, model_name, pred_values, *args, **kwargs):
        method, params = self.model_calibrations[model_name]
        return self.calibrator.apply_calibration_to_array(
            pred_values=pred_values,
            method=method,
            params=params,
            input_type="probs",
        )

    def batch_run(self, model_df: pd.DataFrame, output_column="calibration_prob", *args, **kwargs):
        """批量处理入口方法，直接在原DataFrame上更新校准概率"""
        if output_column not in model_df.columns:
            model_df[output_column] = np.nan  # 先初始化一列
        grouped = model_df.groupby("model_name")
        for model_name, group in grouped:
            if model_name not in self.model_calibrations:
                self.logger.warning(f"模型 {model_name} 没有校准参数，跳过处理")
                continue
            model_df.loc[group.index, output_column] = self.calibrator.apply_calibration(
                model_df=group,
                calibration=self.model_calibrations[model_name],
                output_column=output_column,
            )
            self.logger.info(f"模型 {model_name} 批量校准完成，处理了 {len(group)} 条数据")
        return model_df

    def load_calibration_data(self) -> None:
        """
        加载预计算的校准数据

        Args:
            calibration_data: 校准数据字典
        """
        self.logger.info(f"使用的校准文件路径: {self.calibration_file}")
        with open(self.calibration_file, "r") as f:
            calibration_data = json.load(f)

        for model_name, params in calibration_data.items():
            self.model_calibrations[model_name] = params["calibration_func"]
            try:
                self.calibrator.validate_calibration_params(**params["calibration_func"])
            except ValueError as e:
                self.logger.error(f"模型 {model_name} 校准参数无效: {e}")
                continue
        self.logger.info(f"加载了{len(self.model_calibrations)}个模型的校准参数")
