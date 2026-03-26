# 绘图 - 将所有模型的分布图放到同一个HTML文件中
from model_calibration.metrics.performance_visualizer import PerformanceVisualizer
from pathlib import Path
import json
import numpy as np
import pandas as pd
from model_calibration import CalibrationConfig
from model_calibration.metrics import PerformanceVisualizer


def main():
    # 创建默认配置
    config = CalibrationConfig(
        save_path=Path(r"E:\桌面\Agent\outputs"),
        metrics_res=None,
        model_res=Path("resources/result/eval"),
        # model_res=Path("resources/eval"),
        log_level="INFO",
        # 启用高级功能
        ace_adaptive=True,
        enable_bma=True,
        bma_method="bic",
        ensemble_size=5,
        dropout_samples=10,
    )
    performance_visualizer = PerformanceVisualizer(config)
    # 生成包含所有模型的统一HTML文件
    with open(config.save_path / "calibration_result.json", "r") as f:
        calibration_result = json.load(f)

    for value in calibration_result:
        performance_visualizer.plot_reliability_diagram(
            y_true=value["true_labels"],
            y_prob_original=value["original_probs"],
            y_prob_calibrated=value["calibrated_probs"],
            model_name=value["model_name"],
        )
        break

    # performance_visualizer.plot_reliability_diagram(models_data=calibration_result, save_name="所有模型概率分布对比.html")


if __name__ == "__main__":
    main()
    print("绘图完成，结果保存在 outputs 目录下的 '所有模型概率分布对比.html' 文件中。")
