from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from forensic_agent.configs.config_dataclass import CalibrationConfig
from forensic_agent.calibration import CalibrationSystem
from forensic_agent.data_operation.database_manager import DatabaseManager
from forensic_agent.data_operation.dataset_loader import load_project_data
from forensic_agent.utils.custom_json_encoder import CustomJsonEncoder
from cfg import CONFIG
import json


def load_data(val_name, test_name, train_name="GenImage"):
    """加载数据并进行基本验证"""
    try:
        train_data_path = CONFIG.get("train_dataset", None)
        val_data_path = CONFIG.get("val_dataset", None)
        if train_data_path and val_data_path:
            calibration_data = pd.read_csv(train_data_path)
            # 将model_name 中的 Patch_Shuffle 替换为 PatchShuffle
            val_data = pd.read_csv(val_data_path)
            val_data["model_name"] = val_data["model_name"].str.replace("Patch_Shuffle", "PatchShuffle")
        elif train_data_path:
            print("正在从CSV文件加载训练数据...")
            data = pd.read_csv(train_data_path)
            calibration_data, val_data = train_test_split(data.copy(), test_size=0.5, random_state=42)
        else:
            print("正在从数据库加载数据...")
            data, _ = load_project_data(**CONFIG["dataset"])
            print(f"✅ 数据加载完成，记录数: {len(data)}")

            if data is None or data.empty:
                raise ValueError("加载的数据为空")

            if "image_path" not in data.columns:
                raise ValueError("数据中缺少 image_path 列")

            if val_name == "GenImage":
                # 随机切分数据，50%作为校准集，50%作为验证集
                calibration_data, val_data = train_test_split(data.copy(), test_size=0.5, random_state=42)
            else:
                # 切分data, 将dataset_name为'Genimage++'的数据作为验证集, 其余作为校准集
                calibration_data = data[data["dataset_name"] == train_name].copy()
                if val_name:
                    val_data = data[data["dataset_name"] == val_name].copy()
                    print(f"✅ 验证集记录数: {len(val_data)}")
                else:
                    val_data = pd.DataFrame()

        # 要求找到dataset_name为test_name的测试集
        test_data_path = CONFIG.get("test_dataset", None)
        if test_data_path:
            test_data = pd.read_csv(test_data_path)
        else:
            test_names = test_name if isinstance(test_name, list) else [test_name]
            test_data = data[data["dataset_name"].isin(test_names)].copy()

        # image_path和dataset_name合并去重
        test_data = test_data.drop_duplicates(subset=["image_path", "model_name"])

        calibration_data["model_name"] = calibration_data["model_name"].str.replace("Patch_Shuffle", "PatchShuffle")
        val_data["model_name"] = val_data["model_name"].str.replace("Patch_Shuffle", "PatchShuffle")
        test_data["model_name"] = test_data["model_name"].str.replace("Patch_Shuffle", "PatchShuffle")

        print(f"✅ 校准集记录数: {len(calibration_data)}")
        print(f"✅ 验证集记录数: {len(val_data)}")
        print(f"✅ 测试集记录数: {len(test_data)}")

        return calibration_data, val_data, test_data

    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        raise


def main(val_name="GenImage", test_name="x-fuse", train_name="GenImage-train"):
    """主函数：运行模型校准系统"""

    print("=== AIGC模型校准系统 ===")
    print("基于后处理校准（Post-hoc Calibration）修正模型过自信问题")
    start_time = time.time()

    calibration_df, val_df, test_df = load_data(val_name, test_name, train_name)

    # 创建默认配置
    calibration_config = CalibrationConfig(**CONFIG["calibration"])
    print(f"📁 输出目录: {calibration_config.save_dir}")
    print(f"🔧 校准方法: {', '.join(calibration_config.calibration_methods)}")
    print(f"📊 ECE阈值: {calibration_config.ece_threshold}")
    print(f"🎯 自适应ACE: {'✅' if calibration_config.ace_adaptive else '❌'}")
    print(f"🔗 BMA集成: {'✅' if calibration_config.enable_bma else '❌'}")

    # 初始化校准系统
    calibration_system = CalibrationSystem(
        calibration_df=calibration_df,
        val_df=val_df,
        test_df=test_df,
        config=calibration_config,
    )

    # 运行完整校准流程
    print("开始执行完整校准流程...")
    print("这可能需要几分钟时间，请耐心等待...\n")
    calibration_results = calibration_system.run()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n✅ 校准流程完成，耗时: {execution_time:.2f} 秒")

    if calibration_results:
        # 保存完整结果为json文件
        save_path = calibration_config.save_dir / f"calibration_profile_{test_name}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(calibration_results, f, indent=4, ensure_ascii=False, cls=CustomJsonEncoder)
        print(f"\n✅ 模型校准画像已保存: {save_path}")

    # print("将结果保存到数据库和JSON文件中...")
    # saver = DatabaseManager(db_dir=CONFIG["dataset"]["db_dir"])
    # saver.insert_calibration_results(val_df) if not val_df.empty else None
    # saver.insert_calibration_results(test_df) if not test_df.empty else None
    # 保存校准后的结果到csv文件
    test_csv_path = calibration_config.save_dir / f"calibrated_test_results_{test_name}.csv"
    calibration_system.test_df.to_csv(test_csv_path, index=False)
    print(f"✅ 测试集校准结果已保存到: {test_csv_path}")

    if CONFIG.get("val_dataset", None):
        calibration_system.val_csv_path = calibration_config.save_dir / f"calibrated_val_results_{val_name}.csv"
        if not val_df.empty:
            val_df.to_csv(calibration_system.val_csv_path, index=False)
            print(f"✅ 验证集校准结果已保存到: {calibration_system.val_csv_path}")
        train_csv_path = calibration_config.save_dir / f"calibrated_train_results_{train_name}.csv"
        calibration_system.calibration_df.to_csv(train_csv_path, index=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        print("请检查数据目录是否存在，或查看日志获取更多信息")
        import traceback

        traceback.print_exc()
