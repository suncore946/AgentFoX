"""
1. model: ["CLIPC2P", "DeeCLIP", "DRCT", "Patch_Shuffle", "SPAI"]
2. 训练数据集: ["GenImage"]
3. 测试数据集: ["AIGIBench", "Chameleon", "CO-SPYBench", "WildRF", "WIRE", "synthbuster", "Community-Forensics-eval"]

流程:
1. 读取数据集
2. 根据检测结果进行单热编码
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml
from forensic_agent.application_builder import ApplicationBuilder
from forensic_agent.core.forensic_agent import ForensicAgent
from forensic_agent.core.forensic_llm import ForensicLLM
from forensic_agent.core.forensic_tools import ForensicTools
from forensic_agent.core.tools.expert_analysis_tool import ExpertAnalysisTool
from forensic_agent.core.tools.expert_results_tool import ExpertResultsTool
from forensic_agent.manager.config_manager import ConfigManager
from forensic_agent.manager.datasets_manager import DatasetsManager
from forensic_agent.manager.image_manager import ImageManager
from forensic_agent.manager.logger_manager import LoggerManager
from forensic_agent.manager.profile_manager import ProfileManager
from forensic_agent.manager.semantic_manager import SemanticAnalysisManager
from forensic_agent.processor.expert_analysis_processor import ExpertAnalysisProcessor


class DataLabelingPipeline:
    """AFA应用程序主类"""

    def __init__(self, target_path, config_path: str | Path = None, is_debug=True, max_workers=32, *args, **kwargs) -> None:
        """
        初始化AFA应用程序
        Args:
            config_path: 配置文件路径
        """
        self._container = None
        self._orchestrator = None
        self._config = None
        self._initialized = False

        self.target_path: Path = Path(target_path)

        self.config_path = Path(config_path)
        assert self.config_path.exists(), "配置文件不存在，请提供有效的配置路径"

        print(f"配置文件路径: {self.config_path}")
        # 读取配置文件
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.image_manager = ImageManager(self.config.get("image_config", {}))
        self.profiles_manager = ProfileManager(self.config.get("profiles", {}))
        self.datasets_manager = DatasetsManager(self.config.get("labeling_datasets", {}))

        # 如果是csv文件，则读取第一列作为图像路径
        if self.target_path.suffix == ".csv":
            df = pd.read_csv(self.target_path)
            image_paths = df["image_path"].drop_duplicates().tolist()
        else:
            with open(self.target_path, "r", encoding="utf-8") as f:
                image_paths = list(json.load(f).keys())

        self.datasets_manager.filtration(image_paths)

        self.model_results_tools = ExpertResultsTool(self.config, self.profiles_manager, self.datasets_manager, *args, **kwargs)

        self.save_dir = Path(self.config.get("save_dir", "./outputs/data_labeling"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        feature_config = self.config.get("feature_extraction", {})
        llm_config = self.config.get("labeling_config", {})
        self.forensic_llm = ForensicLLM(llm_config)
        self.semantic_manager = SemanticAnalysisManager(
            feature_config,
            self.image_manager,
            semantic_llm=self.forensic_llm.llm,
            prompt_path=llm_config["prompt_path"],
        )

        self.open_calibration = self.config["agent"].get("open_calibration", False)
        self.expert_analysis = ExpertAnalysisProcessor(
            config=self.config,
            tools_llm=self.forensic_llm.llm,
            prompt_path=self.config["tools"]["ExpertAnalysis"]["prompt_path"],
        )

        self.is_debug = is_debug
        self.max_workers = max_workers

    def semantic_label(self):
        dataset_name = Path(target_path).parent.name
        save_file = self.save_dir / f"{dataset_name}_semantic_labeling_results.json"
        image_paths = self.datasets_manager.get_image_and_label()

        # 如果save_file已存在，则跳值不为None的项
        if save_file.exists():
            with open(save_file, "r", encoding="utf-8") as f:
                ret: dict = json.load(f)
            # 已完成标注（非 None）的图像集合
            processed = {p for p, d in ret.items() if d}
            if processed:
                print(f"已存在标注结果，已完成 {len(processed)} 张图像，跳过这些图像")
            # 待处理的图像字典（排除已完成）
            image_paths = {k: v for k, v in image_paths.items() if k not in processed}
            if not image_paths:
                print("没有需要标注的图像，已全部完成。")
                return
        else:
            ret = {}
            print(f"总共读取到 {len(image_paths)} 张图像需要标注")

        # 多线程并发
        if self.is_debug:
            with tqdm(total=len(image_paths), desc="标注进度", unit="image") as pbar:
                for image_path, image_data in image_paths.items():
                    try:
                        semantic_result, _ = self.semantic_manager.run(
                            image_path=image_path,
                            image_label=image_data["gt_label"],
                        )
                        ret[image_path] = {"gt_label": image_data["gt_label"]} | semantic_result
                    except Exception as e:
                        print(f"处理图像 {image_path} 时出错: {e}")
                        ret[image_path] = None
                    pbar.update(1)
        else:
            # 线程池并发
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_image = {
                    executor.submit(
                        self.semantic_manager.run,
                        image_path=image_path,
                        image_label=image_data["gt_label"],
                    ): (image_path, image_data)
                    for image_path, image_data in image_paths.items()
                }

                with tqdm(total=len(image_paths), desc="标注进度", unit="image") as pbar:
                    for future in as_completed(future_to_image):
                        image_path, image_data = future_to_image[future]
                        try:
                            semantic_result, _ = future.result()
                            ret[image_path] = {"gt_labels": image_data["gt_label"]} | semantic_result
                        except Exception as e:
                            print(f"处理图像 {image_path} 时出错: {e}")
                            ret[image_path] = None
                        pbar.update(1)

        # 保存结果
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(ret, f, indent=4, ensure_ascii=False)
        print(f"标注结果已保存到 {save_file}")

    def expert_label(self, target_path: str):
        with open(target_path, "r", encoding="utf-8") as f:
            image_info = json.load(f)

        dataset_name = Path(target_path).parent.name
        save_file = self.save_dir / f"{dataset_name}_expert_labeling_results.json"
        # 如果save_file已存在，则跳值不为None的项
        if save_file.exists():
            with open(save_file, "r", encoding="utf-8") as f:
                ret: dict = json.load(f)
            target_image_paths = []
            for image_path, image_data in ret.items():
                if image_data is None:
                    target_image_paths.append(image_path)
            print(f"已存在标注结果，跳过已有结果的图像，剩余 {len(target_image_paths)} 张图像待标注")
            image_info = {k: v for k, v in image_info.items() if k in target_image_paths}
        else:
            ret = {}
            print(f"总共读取到 {len(image_info)} 张图像需要标注")

        # 多线程并发
        if self.is_debug:
            with tqdm(total=len(image_info), desc="标注进度", unit="image") as pbar:

                for image_path, image_data in image_info.items():
                    try:
                        model_result = self.model_results_tools.execute(state={"image_path": image_path})
                        profiles = self.profiles_manager.get_model_profiles(list(model_result.keys()))
                        info = {
                            "model_result": model_result,
                            "model_profiles": profiles,
                            "semantic_anomalies": image_data["detected_anomalies"],
                        }
                        if self.open_calibration:
                            info["calibration_profile"] = self.profiles_manager.calibration_profiles
                        expert_result = self.expert_analysis.process_file_with_gt(
                            model_profile=info,
                            image_label=image_data["gt_label"],
                        )

                        ret[image_path] = {
                            "gt_label": image_data["gt_label"],
                            "expert_analysis": expert_result,
                        }
                    except Exception as e:
                        print(f"处理图像 {image_path} 时出错: {e}")
                        ret[image_path] = None
                    pbar.update(1)
        else:
            # 线程池并发
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_image = {
                    executor.submit(
                        self.semantic_manager.process_file_with_gt,
                        image_path=image_path,
                        image_label=image_data["gt_label"],
                    ): (image_path, image_data)
                    for image_path, image_data in image_info.items()
                }

                with tqdm(total=len(image_info), desc="标注进度", unit="image") as pbar:
                    for future in as_completed(future_to_image):
                        image_path, image_data = future_to_image[future]
                        try:
                            expert_result, _ = future.result()
                            ret[image_path] = {"gt_label": image_data["gt_label"]} | expert_result
                        except Exception as e:
                            print(f"处理图像 {image_path} 时出错: {e}")
                            ret[image_path] = None
                        pbar.update(1)

        # 保存结果
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(ret, f, indent=4, ensure_ascii=False)
        print(f"标注结果已保存到 {save_file}")


if __name__ == "__main__":
    config_path = "/data2/yuyangxin/Agent/forensic_agent/configs/config_qwen3_32b_genimage.yaml"
    # target_path = "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage/semantic_errors.json"
    # target_path = "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage/samples_by_label.csv"
    target_path = "/data2/yuyangxin/Agent/outputs/data_labeling/GenImage_semantic_labeling_results.json"
    labeling = DataLabelingPipeline(target_path, config_path=config_path, is_debug=True, max_workers=32)
    # labeling.semantic_label()
    labeling.expert_label(target_path=target_path)
