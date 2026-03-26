from typing import Dict, Any
import pandas as pd
from .tools_base import ToolsBase
from .schema.model_profiles_schema import ModelProfilesSchema
from ...manager.datasets_manager import DatasetsManager
from ...manager.profile_manager import ProfileManager


class ExpertResultsTool(ToolsBase):
    # args_schema = ModelProfilesSchema

    def __init__(self, config, profile_manager: ProfileManager, datasets_manager: DatasetsManager, *args, **kwargs):
        self.profiles_manager = profile_manager
        self.target_columns = ["pred_prob"]
        self.open_calibration = config.get("open_calibration", True)
        if self.open_calibration:
            self.target_columns.append("calibration_prob")
        self.detail_data = datasets_manager.detail_data
        self.target_models = profile_manager.target_models
        super().__init__(config, *args, **kwargs)

    @property
    def name(self) -> str:
        return "expert_results"

    @property
    def description(self) -> str:
        if self.open_calibration:
            return "Obtain the forensic expert’s prediction probability and calibrated probability results on whether the image is AI-generated."
        else:
            return "Obtain the forensic expert’s probability result on whether the image is AI-generated."

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """执行模型结果查询"""
        try:
            params = self.args_schema.model_validate(kwargs)
            target_models = getattr(params, "model_names", [])
            if len(target_models) <= 1 or target_models[0].lower() == "all":
                target_models = self.target_models

            invalid_models = sorted(set(target_models) - set(self.target_models))
            if invalid_models:
                return {"error": f"Invalid model names: {invalid_models}. Available models: {self.target_models}"}

            target_models = [m for m in target_models if m in self.target_models]
            if not target_models:
                target_models = self.target_models
        except Exception:
            target_models = self.target_models

        image_path = params.get_image_path()
        matched = self.detail_data[self.detail_data["image_path"] == image_path]
        # 根据 image_path 查询结果，若无匹配返回空字典
        if matched.empty:
            raise ValueError(f"No model prediction results found for image_path: {image_path}")

        model_results: pd.DataFrame = matched.set_index("model_name").copy()

        # 只选择存在的目标列，避免 KeyError
        available_cols = [c for c in self.target_columns if c in model_results.columns]
        if not available_cols:
            # 没有可用列，返回仅含索引的空结构
            base = {idx: {} for idx in model_results.index}
            return base if target_models is None else {m: base.get(m, {}) for m in target_models}

        # 强制数值列为 numeric，非数值变为 NaN，然后四舍五入
        model_results.loc[:, available_cols] = model_results.loc[:, available_cols].apply(
            lambda s: pd.to_numeric(s, errors="coerce").round(4)
        )

        # 仅返回存在的模型结果，忽略未找到的模型名
        # 重命名柱子以匹配输出要求
        model_info = (
            model_results[available_cols]
            .rename(columns={"pred_prob": "prediction probability", "calibration_prob": "calibrated probability"})
            .to_dict(orient="index")
        )

        if not target_models or len(target_models) == 1 and target_models[0].lower() == "all":
            ret = model_info
        else:
            ret = {m: model_info[m] for m in target_models if m in model_info}
        if ret == {}:
            raise ValueError(
                f"No model prediction results found for image_path: {image_path} with specified models. target_models: {target_models}"
            )

        if self.open_calibration:
            # 校准是在训练集上进行的, 与真实图像存在偏差。
            ret["calibration_note"] = (
                """When the calibration results do not match the predicted results, you should carefully consider the calibration results. This might be due to noise interference. Calibration are fitted on the training set and evaluated on the validation set; they may not match real-world conditions. Calibrated probabilities are provided for reference only and should not be used as sole evidence. """
            )
        return ret
