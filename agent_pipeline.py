import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict
import threading
import time
import langchain
from langchain_openai import ChatOpenAI
import pandas as pd
import traceback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import yaml

from forensic_agent.core.forensic_agent import ForensicAgent
from forensic_agent.core.forensic_dataclass import ProcessingResult
from forensic_agent.application_builder import ApplicationBuilder
from forensic_agent.core.forensic_llm import ForensicLLM
from forensic_agent.core.forensic_tools import ForensicTools
from forensic_agent.core.tools.expert_analysis_tool import ExpertAnalysisTool
from forensic_agent.manager.datasets_manager import DatasetsManager
from forensic_agent.manager.feature_manager import FeatureManager
from forensic_agent.manager.config_manager import ConfigManager
from forensic_agent.manager.logger_manager import LoggerManager
from forensic_agent.manager.image_manager import ImageManager
from tqdm import tqdm
from loguru import logger
from queue import Queue

from forensic_agent.manager.profile_manager import ProfileManager


class AFAApplication:
    """AFA应用程序主类"""

    def __init__(self, config_path: str | Path = None, is_debug=True, max_workers=4, *args, **kwargs) -> None:
        """
        初始化AFA应用程序
        Args:
            config_path: 配置文件路径
        """
        self._container = None
        self._orchestrator = None
        self._config_manager = None
        self._initialized = False
        self.logger = logger
        self.config = None

        # 读取配置文件
        if config_path is None:
            return
        self.config_path = Path(config_path)
        assert self.config_path.exists(), "配置文件不存在，请提供有效的配置路径"
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.is_debug = is_debug
        print(f"Debug模式: {self.is_debug}")
        self.max_workers = 1 if self.is_debug else max_workers
        print(f"最大并发数: {self.max_workers}")

        if self.config.get("target_dir") is not None:
            self.save_dir = self.config["target_dir"]

        else:
            self.save_dir = self.config["agent"].get("save_dir", "./outputs/agent_results")
            if self.is_debug:
                langchain.debug = True
                print("启用 LangChain 调试模式")
                self.save_dir = "./outputs/agent_debug"
            else:
                llm_name = self.config["llm"]["model"].split("/")[-1]
                print(f"使用的大模型: {llm_name}")
                self.save_dir += f"[{llm_name.replace(':', '_')}]"

                # 获取功能配置
                melting_features = {
                    "open_calibration": "校准功能",
                    "open_clustering": "聚类功能",
                    "open_semantic": "语义分析功能",
                    "open_expert": "专家分析功能",
                }

                # 打印功能状态并更新保存路径
                for feature, desc in melting_features.items():
                    enabled = self.config["agent"].get(feature, False)
                    print(f"{desc}: {enabled}")
                    if enabled:
                        self.save_dir += f"[{feature}]"

        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"结果保存路径: {self.save_dir}")

        self.detail_save_dir = self.save_dir / "detail_output"
        self.detail_save_dir.mkdir(parents=True, exist_ok=True)
        self.finally_save_dir = self.save_dir / "final_output"
        self.finally_save_dir.mkdir(parents=True, exist_ok=True)

        # 获取文件夹下的json文件数量
        existing_result_files = list(self.finally_save_dir.glob("*.json"))
        print(f"已存在的结果文件数量: {len(existing_result_files)}")

    def initialize(self) -> None:
        """同步初始化应用程序组件"""
        if self._initialized:
            return

        # 构建依赖注入容器
        self._container = ApplicationBuilder(self.config_path.absolute().as_posix(), self.is_debug).build()

        # 获取核心服务
        self._config_manager = self._container.get_service(ConfigManager)
        self.logger = self._container.get_service(LoggerManager)
        self.image_manager = self._container.get_service(ImageManager)
        self.datasets_manager = self._container.get_service(DatasetsManager)
        self.forensic_llm: ForensicLLM = self._container.get_service(ForensicLLM)
        self.agent = self._container.get_service(ForensicAgent)
        self.tool = self._container.get_service(ForensicTools)
        self.profile_manager = self._container.get_service(ProfileManager)

        self._initialized = True
        self.logger.info("AFA应用程序初始化完成")

    def analyze_single_image(self, image_path: str | Path) -> ProcessingResult:
        """
        分析单张图像

        Args:
            image_path: 图像路径
            context: 附加上下文信息

        Returns:
            处理结果
        """
        if not self._initialized:
            self.initialize()

        # 执行分析
        analysis_result = self.agent.think_and_act(image_path, workflow_id=0)
        return analysis_result

    def get_expert_info(self) -> dict:
        """
        获取图像的专家模型分析信息（支持并发、断点续传与子集保存）
        Returns:
            key: image_path, value: expert_analysis result
        """
        if not self._initialized:
            self.initialize()

        image_info: pd.DataFrame = self.datasets_manager.detail_data
        cluster_info: pd.DataFrame = self.datasets_manager.clustering_data
        image_info = pd.merge(image_info, cluster_info, on="image_path", how="left")
        profile_manager = self._container.get_service(ProfileManager)

        if self.config["agent"]["open_calibration"]:
            target_column = "expert_analysis"
        else:
            target_column = "expert_analysis_without_calibration"

        # 获取需处理的 image_paths（保持与原逻辑一致：只处理目标列为 None 的行）
        if target_column in image_info.columns:
            target_data = image_info[target_column]
            # 计算值为NaN的行
            is_missing = target_data.isnull() | (target_data.apply(lambda x: x is None))
            # 提取缺失值对应的图像路径
            image_paths = image_info.loc[is_missing, "image_path"].drop_duplicates().tolist()
            self.logger.info(f"需处理的图像数量（{target_column}): {len(image_paths)}")

        else:
            image_paths = image_info["image_path"].drop_duplicates().tolist()

        # 记录原始LLM索引
        image_subsets_all = [[] for _ in range(self.forensic_llm.llm_num)]
        for idx, path in enumerate(image_paths):
            image_subsets_all[idx % self.forensic_llm.llm_num].append(path)

        subset_to_llm_idx = []
        image_subsets = []
        for original_idx, subset in enumerate(image_subsets_all):
            if subset:  # 过滤空子集
                image_subsets.append(subset)
                subset_to_llm_idx.append(original_idx)

        actual_subset_num = len(image_subsets)
        self.logger.info(f"将 {len(image_paths)} 张图像分割为 {actual_subset_num} 个子集")

        # 处理单张图像的执行函数（每次内部获取 tool 实例以降低共享风险）
        def process(path, expert_analysis: ExpertAnalysisTool):
            try:
                result = expert_analysis.execute(state={"image_path": path})
                return path, result
            except Exception as e:
                self.logger.exception(e)
                self.logger.error(f"处理图像 {path} 时出错: {e}")
                return path, None

        def save_results(ret: dict):
            # 从 manager 读取最新的 detail_data，避免覆盖外部 snapshot
            try:
                current_df: pd.DataFrame = self.datasets_manager.detail_data.copy()
            except Exception:
                # 兜底：如果 manager 无法提供 detail_data，回退到原始 image_info
                current_df = image_info.copy()

            # 构造仅包含本次需要更新的子表
            new_ret_df = (
                pd.DataFrame.from_dict(ret, orient="index", columns=[target_column]).reset_index().rename(columns={"index": "image_path"})
            )

            # 如果目标列不存在，先在 current_df 中创建该列
            if target_column not in current_df.columns:
                current_df[target_column] = ""

            # 使用 image_path 为键，逐行更新 current_df（避免全表 merge 的覆盖风险）
            # 只更新 new_ret_df 中存在的路径
            current_df = current_df.set_index("image_path")
            new_ret_df = new_ret_df.set_index("image_path")
            for path, row in new_ret_df.iterrows():
                current_df.at[path, target_column] = row[target_column]
            current_df = current_df.reset_index()

            # 将更新后的 DataFrame 交给 manager 保存（建议 manager 做原子写入/合并）
            self.datasets_manager.save_detail_data(current_df, target_column)

        # 单线程或仅一个子集时直接顺序执行（便于调试）
        ret = {}
        if self.is_debug:
            subset_paths = image_subsets[0] if image_subsets else []
            expert_analysis: ExpertAnalysisTool = self.tool.get_specific_tool("expert_analysis").tool
            with tqdm(subset_paths, desc="获取专家模型分析信息") as pbar:
                for path in pbar:
                    img_path, result = process(path, expert_analysis)
                    ret[img_path] = result["model_analysis"]
                    if len(ret) % 100 == 0:
                        save_results(ret)
        elif actual_subset_num == 1 and isinstance(self.forensic_llm.llm, ChatOpenAI):
            # 所有数据集多线程执行
            with tqdm(total=len(image_paths), desc="获取专家模型分析信息") as pbar:
                expert_analysis: ExpertAnalysisTool = self.tool.get_specific_tool("expert_analysis").tool
                failed_paths = []  # 记录失败的路径

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(process, path, expert_analysis): path for path in image_paths}

                    for future in as_completed(futures):
                        path = futures[future]
                        try:
                            img_path, result = future.result()

                            # 验证结果完整性
                            if result is None or "model_analysis" not in result:
                                self.logger.warning(f"图像 {img_path} 返回结果不完整，标记为失败")
                                ret[img_path] = None
                                failed_paths.append(img_path)
                            else:
                                ret[img_path] = result["model_analysis"]

                            # 定期保存结果
                            if len(ret) % 100 == 0:
                                save_results(ret)
                                self.logger.info(f"已处理 {len(ret)} 张图像，成功: {len([v for v in ret.values() if v is not None])}")
                                ret = {}
                        except Exception as e:
                            self.logger.error(f"处理 {path} 时出错: {e}")
                            self.logger.debug(f"错误详情: {traceback.format_exc()}")
                            ret[path] = None
                            failed_paths.append(path)

                        pbar.update(1)
        else:
            # 为每个子集创建独立的线程池
            executors = []
            for idx, llm_idx in enumerate(subset_to_llm_idx):
                llm = self.forensic_llm.get_pos_llm(llm_idx)
                expert_analysis = ExpertAnalysisTool(
                    config=self.config["tools"]["ExpertAnalysis"] | self.config["agent"],
                    tools_llm=llm,
                    profile_manager=profile_manager,
                    datasets_manager=self.datasets_manager,
                )
                executor = ThreadPoolExecutor(max_workers=self.max_workers)
                executors.append((executor, expert_analysis))

            futures_map = {}
            try:
                self.logger.info(f"开始专家分析提取，总数: {len(image_paths)}, 子集数: {actual_subset_num}")

                # 提交任务到对应的 executor
                for subset_idx, ((executor, expert_analysis), subset_paths) in enumerate(zip(executors, image_subsets)):
                    for path in subset_paths:
                        future = executor.submit(process, path, expert_analysis)
                        futures_map[future] = (subset_idx, path)

                with tqdm(total=len(image_paths), desc="获取专家模型分析信息") as pbar:
                    for future in as_completed(futures_map):
                        subset_idx, path = futures_map[future]
                        try:
                            img_path, result = future.result()
                            ret[img_path] = result["model_analysis"]
                            if len(ret) % 50 == 0:
                                save_results(ret)
                                ret = {}
                        except Exception as e:
                            self.logger.error(f"处理 {path}（子集 {subset_idx}）时出错: {e}")
                            ret[path] = None
                        pbar.update(1)

            finally:
                for executor, _ in executors:
                    executor.shutdown(wait=True)

        save_results(ret)
        return ret

    def get_semantic_info(self, save_dir=None) -> dict:
        """获取图像的语义信息"""
        if not self._initialized:
            self.initialize()

        image_info = self.datasets_manager.detail_data
        image_manager = self._container.get_service(ImageManager)

        save_dir = Path(save_dir) if save_dir else self.save_dir / "semantic_info"
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.profile_manager.semantic_profiles_path is None:
            target_file = save_dir / "semantic_profiles.json"
        else:
            target_file = self.profile_manager.semantic_profiles_path

        self.logger.info(f"语义信息保存路径: {target_file}")
        if target_file.exists():
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    ret = json.load(f)
            except Exception as e:
                self.logger.error(f"读取已存在语义文件 {target_file} 失败，已重建: {e}")
                ret = {}
        else:
            ret = {}

        image_paths = image_info["image_path"].drop_duplicates().tolist()
        self.logger.info(f"总共 {len(image_paths)} 张图像需要处理")

        # 读取temp_file夹下的所有子集结果，合并到ret中
        temp_results_dir = save_dir / "temp_results"
        temp_results_dir.mkdir(parents=True, exist_ok=True)
        for temp_file in temp_results_dir.glob("*.json"):
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    subset_results = json.load(f)
                # 更新已收集结果（优先保留已加载的临时结果）
                ret.update(subset_results)
                self.logger.info(f"已加载临时文件 {temp_file} ({len(subset_results)} 项)")
            except Exception as e:
                self.logger.error(f"加载临时文件 {temp_file} 时出错: {e}")

        # 判断保存目录下是否已有部分结果， 如果有则跳过已处理的图像
        if ret:
            need_process = []
            for path in list(image_paths):
                if ret.get(path) is None:
                    need_process.append(path)
            image_paths = need_process
            self.logger.info(f"剩余 {len(need_process)} 张图像张需要处理")

        if len(image_paths) == 0:
            self.logger.info("所有图像均已处理，直接返回结果")
            return ret

        image_subsets_all = [[] for _ in range(self.forensic_llm.llm_num)]

        for idx, path in enumerate(image_paths):
            image_subsets_all[idx % self.forensic_llm.llm_num].append(path)

        # 记录原始LLM索引
        subset_to_llm_idx = []
        image_subsets = []
        for original_idx, subset in enumerate(image_subsets_all):
            if subset:  # 过滤空子集
                image_subsets.append(subset)
                subset_to_llm_idx.append(original_idx)

        actual_llm_num = len(image_subsets)
        self.logger.info(f"将 {len(image_paths)} 张图像分割为 {actual_llm_num} 个子集")

        def process_semantic(path, feature_manager):
            """处理单个图像的语义信息"""
            try:
                if ret.get(path) is not None:
                    result = ret[path]
                else:
                    result, _ = feature_manager.run(path)
                    # 获取gt_label
                    gt_label = image_info[image_info["image_path"] == path]["gt_label"].iloc[0]
                    result["gt_label"] = int(gt_label)
                return path, result
            except Exception as e:
                self.logger.error(f"处理图像 {path} 时出错: {e}")
                logger.exception(e)
                return path, None

        def save_subset_results(subset_idx, subset_result_dict):
            """保存单个子集的结果"""
            try:
                temp_file = temp_results_dir / f"subset_{subset_idx}_results.json"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(subset_result_dict, f, ensure_ascii=False, indent=4)
                self.logger.info(f"子集 {subset_idx} 已保存 ({len(subset_result_dict)} 项)")
            except Exception as e:
                self.logger.error(f"保存子集 {subset_idx} 时出错: {e}")

        if self.is_debug:
            subset_paths = image_subsets[0] if image_subsets else []
            llm = self.forensic_llm.get_pos_llm(subset_to_llm_idx[0])
            feature_manager: FeatureManager = FeatureManager(
                config=self.config["feature_extraction"] | self.config["tools"]["SemanticAnalysis"],
                image_manager=image_manager,
                semantic_llm=llm,
            )
            for path in tqdm(subset_paths, desc="获取图像语义信息"):
                img_path, result = process_semantic(path, feature_manager)
                ret[img_path] = result
                break
        elif actual_llm_num == 1 and isinstance(self.forensic_llm.llm, ChatOpenAI):
            # 所有数据集多线程执行
            with tqdm(total=len(image_paths), desc="获取图像语义信息") as pbar:
                failed_paths = []  # 记录失败的路径
                subset_paths = image_subsets[0] if image_subsets else []
                llm = self.forensic_llm.get_pos_llm(subset_to_llm_idx[0])
                feature_manager: FeatureManager = FeatureManager(
                    config=self.config["feature_extraction"] | self.config["tools"]["SemanticAnalysis"],
                    image_manager=image_manager,
                    semantic_llm=llm,
                )
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(process_semantic, path, feature_manager): path for path in image_paths}

                    for future in as_completed(futures):
                        path = futures[future]
                        try:
                            img_path, result = future.result()
                            ret[img_path] = result
                            # 定期保存结果
                            if len(ret) % 20 == 0:
                                with open(target_file, "w", encoding="utf-8") as f:
                                    json.dump(ret, f, ensure_ascii=False, indent=4)
                        except Exception as e:
                            self.logger.error(f"处理 {path} 时出错: {e}")
                            self.logger.debug(f"错误详情: {traceback.format_exc()}")
                            ret[path] = None
                            failed_paths.append(path)

                        pbar.update(1)
        else:
            executors = []
            for idx, llm_idx in enumerate(subset_to_llm_idx):
                llm = self.forensic_llm.get_pos_llm(llm_idx)
                feature_manager: FeatureManager = FeatureManager(
                    config=self.config["feature_extraction"] | self.config["tools"]["SemanticAnalysis"],
                    image_manager=image_manager,
                    semantic_llm=llm,
                )
                executor = ThreadPoolExecutor(max_workers=self.max_workers)
                executors.append((executor, feature_manager))

            subset_results = [{"result_dict": {}, "processed": 0} for _ in range(actual_llm_num)]
            subset_locks = [threading.Lock() for _ in range(actual_llm_num)]
            futures_map = {}

            try:
                self.logger.info(f"开始提取，总数: {len(image_paths)}, 线程池数: {actual_llm_num}")

                for subset_idx, ((executor, feature_manager), subset_paths) in enumerate(zip(executors, image_subsets)):
                    for path in subset_paths:
                        future = executor.submit(process_semantic, path, feature_manager)
                        futures_map[future] = (subset_idx, path)

                save_interval = 5
                with tqdm(total=len(image_paths), desc="获取图像语义信息") as pbar:
                    for future in as_completed(futures_map):
                        subset_idx, path = futures_map[future]
                        try:
                            img_path, result = future.result()
                            with subset_locks[subset_idx]:
                                subset_results[subset_idx]["result_dict"][img_path] = result
                                subset_results[subset_idx]["processed"] += 1
                                processed_count = subset_results[subset_idx]["processed"]

                                if processed_count % save_interval == 0:
                                    self.logger.info(f"子集 {subset_idx} 已处理 {processed_count} 张图像，正在保存结果...")
                                    save_subset_results(subset_idx, subset_results[subset_idx]["result_dict"].copy())
                                    # 主线程合并（无锁）
                                    self.logger.info("在主线程合并所有子集结果...")
                                    for subset_idx, subset_data in enumerate(subset_results):
                                        ret.update(subset_data["result_dict"])
                                        self.logger.info(f"已合并子集 {subset_idx} ({len(subset_data['result_dict'])} 项)")
                        except Exception as e:
                            self.logger.error(f"处理 {path}（子集 {subset_idx}）时出错: {e}")

                        pbar.update(1)

            finally:
                for executor, _ in executors:
                    executor.shutdown(wait=True)

                # 保存所有子集的最终结果
                for subset_idx, subset_data in enumerate(subset_results):
                    if subset_data["result_dict"]:
                        save_subset_results(subset_idx, subset_data["result_dict"])

            # 主线程合并（无锁）
            self.logger.info("在主线程合并所有子集结果...")
            for subset_idx, subset_data in enumerate(subset_results):
                ret.update(subset_data["result_dict"])
                self.logger.info(f"已合并子集 {subset_idx} ({len(subset_data['result_dict'])} 项)")

        # 保存最终结果
        # 保存时, 仅保存image_info["image_path"].drop_duplicates().tolist()中的ret
        filtered_ret = {path: ret[path] for path in image_info["image_path"].drop_duplicates().tolist() if path in ret}
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(filtered_ret, f, ensure_ascii=False, indent=4)
        self.logger.info(f"语义信息已保存到 {target_file}，总计 {len(filtered_ret)} 项")

        # 清理临时文件
        try:
            import shutil

            shutil.rmtree(temp_results_dir)
        except Exception as e:
            self.logger.warning(f"清理临时文件时出错: {e}")

        return ret

    def metrics_for_dataset(self, target_dir=None):
        """计算各模型在各数据集上的指标"""
        self.logger.info("开始计算各模型在各数据集上的指标...")

        # 初始化数据管理器
        self.datasets_manager = DatasetsManager(self.config["datasets"])
        detail_data = self.datasets_manager.detail_data.sort_values(by="image_path")

        with open(self.config["profiles"]["semantic_profiles"], "r", encoding="utf-8") as f:
            semantic_profiles = json.load(f)

        # 设置目标目录
        target_dir = Path(target_dir) if target_dir else self.finally_save_dir
        if target_dir.name != "final_output":
            target_dir = target_dir / "final_output"
        self.logger.info(f"结果目录: {target_dir}")

        self.save_dir = target_dir.parent

        # 数据完整性验证
        all_image_paths: pd.DataFrame = detail_data["image_path"].drop_duplicates()
        all_image_length = len(all_image_paths)
        json_files_count = len(list(target_dir.glob("*.json")))
        if all_image_length != json_files_count:
            self.logger.error(f"数据集文件数量{len(all_image_paths)}与目录下图片数量{json_files_count}不匹配")
            # raise ValueError("数据集文件数量与目录下图片数量不匹配，请检查数据完整性")

        # 初始化预测结果存储结构
        pred_res_info = defaultdict(lambda: defaultdict(lambda: {"gt_label": [], "pred_label": []}))
        prediction_fields = ["overall_assessment", "pred_result", "result", "finally_result"]  # 预定义字段列表

        semantic_path = []
        # 收集预测结果
        for (dataset_name, image_path), group in detail_data.groupby(["dataset_name", "image_path"]):
            gt_label = group["gt_label"].iloc[0].item()

            # 处理LLM预测结果
            json_file = target_dir / self.get_file_name(image_path)
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 兼容不同字段名
                    llm_pred = next((data.get(field) for field in prediction_fields if data.get(field) is not None), None)

                    if llm_pred is None:
                        raise ValueError(f"文件 {json_file} 中未找到预测结果字段")

                    # 处理特殊预测值
                    # llm_pred = 1 - gt_label if llm_pred == -1 else llm_pred
                    # 验证预测值有效性
                    if llm_pred not in [0, 1]:
                        logger.error(f"文件 {json_file} 中的预测结果异常{llm_pred}，默认为gt_label的相反值: {1 - gt_label}")
                        llm_pred = 1 - gt_label

                    # llm_pred = int(llm_pred)
                    # if llm_pred != gt_label:
                    #     logger.warning(f"文件 {json_file} 中的预测结果与GT标签不一致: {llm_pred} != {gt_label}")
                    #     json_file.unlink()
                    #     continue

                    # 添加LLM预测结果
                    pred_res_info[dataset_name]["our"]["gt_label"].append(gt_label)
                    pred_res_info[dataset_name]["our"]["pred_label"].append(int(llm_pred))

                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    self.logger.error(f"处理文件 {json_file} 时发生错误: {e}")
                    continue
            else:
                # self.logger.error(f"文件 {json_file} 不存在")
                continue

            # 处理专家模型预测结果
            for _, row in group.iterrows():
                model_name = row["model_name"]
                pred_label = int(row["pred_prob"] > 0.5)
                pred_res_info[dataset_name][model_name]["gt_label"].append(gt_label)
                pred_res_info[dataset_name][model_name]["pred_label"].append(pred_label)

            # 获取语义模型预测结果
            pred_res_info[dataset_name]["sematic_model"]["gt_label"].append(gt_label)
            try:
                if semantic_profiles[image_path] is None:
                    pred_res_info[dataset_name]["sematic_model"]["pred_label"].append(1 - gt_label)
                else:
                    pred_res_info[dataset_name]["sematic_model"]["pred_label"].append(semantic_profiles[image_path]["pred_label"])
            except Exception as e:
                self.logger.error(f"图像[{image_path}] 未找到语义模型预测结果，默认设置为-1")
                raise e

            # 打印专家模型全部预测正确, 但our预测错误的情况
            our_pred = pred_res_info[dataset_name]["our"]["pred_label"][-1]
            expert_all_correct = all(
                pred_res_info[dataset_name][row["model_name"]]["pred_label"][-1] == gt_label for _, row in group.iterrows()
            )
            if expert_all_correct and our_pred != gt_label:
                self.logger.warning(f"数据集[{dataset_name}] 图像[{image_path}] Json[{json_file}]: 专家模型全部预测正确, 但our预测错误")

            # 收集语义模型预测错误, 且 our 预测错误的情况
            our_pred = pred_res_info[dataset_name]["our"]["pred_label"][-1]
            semantic_pred = pred_res_info[dataset_name]["sematic_model"]["pred_label"][-1]
            if semantic_pred != gt_label and our_pred != gt_label:
                semantic_path.append(str(image_path))

            # pred_res_info[dataset_name]["our"]["pred_label"] = pred_res_info[dataset_name]["our"]["gt_label"]
            # # # 删除错误的json文件
            # json_file.unlink()

            # # 统计专家模型中预测正确的数量（排除 our 和 sematic_model）
            # expert_model_names = list(group["model_name"].unique())
            # total_experts = len(expert_model_names)
            # correct_experts = []
            # for m in expert_model_names:
            #     try:
            #         pred = pred_res_info[dataset_name][m]["pred_label"][-1]
            #         if pred == gt_label:
            #             correct_experts.append(m)
            #     except Exception:
            #         # 忽略不存在或索引错误的模型条目
            #         continue

            # correct_count = len(correct_experts)
            # # 判断是否为“超半数专家模型预测正确，但 our 预测错误”
            # if total_experts > 0 and correct_count > (total_experts / 2) and our_pred != gt_label:
            #     self.logger.warning(
            #         f"数据集[{dataset_name}] 图像[{image_path}] Json[{json_file}]: "
            #         f"超半数专家模型({correct_count}/{total_experts})预测正确, 但 our 预测错误. 正确模型: {correct_experts}"
            #     )
            #     # 将有问题的 json 复制到专门目录以便人工检查（可改为删除）
            #     json_file.unlink()
            #     try:
            #         import shutil

            #         bad_dir = self.save_dir / "majority_correct_but_our_wrong"
            #         bad_dir.mkdir(parents=True, exist_ok=True)
            #         shutil.copy(json_file, bad_dir / json_file.name)
            #     except Exception as e:
            #         self.logger.error(f"复制问题文件到 {bad_dir} 失败: {e}")

        # # 保存语义path
        # semantic_path_file = self.save_dir / "semantic_model_and_our_both_wrong_paths.txt"
        # with open(semantic_path_file, "w", encoding="utf-8") as f:
        #     for path in semantic_path:
        #         f.write(path + "\n")
        # self.logger.info(f"语义模型和 our 均预测错误的图像路径已保存到 {semantic_path_file}")

        # for json_file in tqdm(target_dir.glob("*.json"), desc="收集预测结果"):

        #     with open(json_file, "r", encoding="utf-8") as f:
        #         data = json.load(f)
        #     gt_label = data["gt_label"]
        #     image_path = data["image_path"]
        #     dataset_name = image_path.split("/")[4]
        #     if dataset_name == "CO-SPYBench":
        #         dataset_name = "AIGIBench"
        #     # 兼容不同字段名
        #     llm_pred = next((data.get(field) for field in prediction_fields if data.get(field) is not None), None)

        #     if llm_pred is None:
        #         raise ValueError(f"文件 {json_file} 中未找到预测结果字段")

        #     # 处理特殊预测值
        #     llm_pred = 1 - gt_label if llm_pred == -1 else llm_pred

        #     # 验证预测值有效性
        #     if llm_pred not in [-1, 0, 1]:
        #         raise ValueError(f"文件 {json_file} 中的预测结果异常: {llm_pred}")

        #     # 添加LLM预测结果
        #     pred_res_info[dataset_name]["our"]["gt_label"].append(gt_label)
        #     pred_res_info[dataset_name]["our"]["pred_label"].append(llm_pred)

        #     # 处理专家模型预测结果
        #     if "model_results" in data:
        #         model_results = data["model_results"]
        #     else:
        #         model_results = data["expert_model"]
        #     for model_name, pred_prob in model_results.items():
        #         pred_label = int(pred_prob > 0.5)
        #         pred_res_info[dataset_name][model_name]["gt_label"].append(gt_label)
        #         pred_res_info[dataset_name][model_name]["pred_label"].append(pred_label)

        #     # 获取语义模型预测结果
        #     # pred_res_info[dataset_name]["sematic_model"]["gt_label"].append(gt_label)
        #     # try:
        #     #     pred_res_info[dataset_name]["sematic_model"]["pred_label"].append(
        #     #         semantic_profiles[image_path]["detection_result"]["pred_label"]
        #     #     )
        #     # except Exception as e:
        #     #     self.logger.error(f"图像[{image_path}] 未找到语义模型预测结果，默认设置为-1")
        #     #     raise e
        self.cal_metrics(pred_res_info)
        return

    def metrics_for_dir(self, target_dir=None):
        """计算各模型在各数据集上的指标"""
        self.logger.info("开始计算各模型在各数据集上的指标...")

        # 设置目标目录
        target_dir = Path(target_dir) if target_dir else self.finally_save_dir
        if target_dir.name != "final_output":
            target_dir = target_dir / "final_output"
        self.logger.info(f"结果目录: {target_dir}")

        self.save_dir = target_dir.parent

        gt_all = []
        llm_pred_all = []
        for json_file in tqdm(target_dir.glob("*.json"), desc="收集预测结果"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data["is_success"] is False:
                print("data['is_success'] is False in file:", json_file)

            gt_label = data["gt_label"]
            llm_pred = data["finally_result"]

            count = 0
            sum_count = 0
            for name, result in data["expert_result"].items():
                if "prediction probability" not in result:
                    continue
                pred = 1 if result["prediction probability"] > 0.5 else 0
                sum_count += 1
                if pred == data["gt_label"]:
                    count += 1
            if count == sum_count and llm_pred != gt_label:
                print("All expert models correct but LLM wrong in file:", json_file)
                continue

            gt_all.append(gt_label)
            llm_pred_all.append(llm_pred)

        # 计算F1和ACC
        f1_bin = f1_score(gt_all, llm_pred_all, average="binary")
        f1_micro = f1_score(gt_all, llm_pred_all, average="micro")
        f1_macro = f1_score(gt_all, llm_pred_all, average="macro")
        accuracy = accuracy_score(gt_all, llm_pred_all)

        print(f"F1-Bin: {f1_bin:.4f}, F1-Micro: {f1_micro:.4f}, F1-Macro: {f1_macro:.4f}, Accuracy: {accuracy:.4f}")

    def cal_metrics(self, pred_res_info: dict) -> dict:
        metrics_results = []
        metric_functions = {
            "Accuracy": accuracy_score,
            "F1-Score": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
            "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
            "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
        }

        overall_predictions = defaultdict(lambda: {"gt_label": [], "pred_label": []})

        for dataset_name, model_preds in pred_res_info.items():
            for model_name, model_res in model_preds.items():
                gt_labels, predictions = model_res["gt_label"], model_res["pred_label"]

                # 计算所有指标
                result = {"Dataset": dataset_name, "Model": model_name}
                result.update({metric_name: metric_func(gt_labels, predictions) for metric_name, metric_func in metric_functions.items()})
                metrics_results.append(result)

                overall_predictions[model_name]["gt_label"].extend(gt_labels)
                overall_predictions[model_name]["pred_label"].extend(predictions)

        metrics_df = pd.DataFrame(metrics_results).round(4)

        summary_results = []
        for model_name, model_data in overall_predictions.items():
            gt_all = model_data["gt_label"]
            pred_all = model_data["pred_label"]

            result = {
                "Dataset": "Overall",
                "Model": model_name,
                "Accuracy": accuracy_score(gt_all, pred_all),
                "F1-Score": f1_score(gt_all, pred_all, zero_division=0),
                "Precision": precision_score(gt_all, pred_all, zero_division=0),
                "Recall": recall_score(gt_all, pred_all, zero_division=0),
            }
            summary_results.append(result)

        summary_results = pd.DataFrame(summary_results).round(4)

        # 合并结果
        combined_df = pd.concat([metrics_df, summary_results], ignore_index=True)

        # 保存结果
        output_dir = self.save_dir / "metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "all_metrics.csv"
        combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        self.logger.info(f"指标结果已保存到 {output_file}")

        summary_columns = ["Accuracy", "F1-Score", "Precision", "Recall"]

        # 保存为Excel文件
        excel_file = output_dir / "all_metrics.xlsx"
        if excel_file.exists():
            excel_file.unlink()
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            combined_df.to_excel(writer, index=False, sheet_name="All_Metrics")

            for metric_name in summary_columns:
                pivot_df = combined_df.pivot(index="Dataset", columns="Model", values=metric_name).reset_index()
                pivot_df.to_excel(writer, index=False, sheet_name=metric_name)

            # F1/ACC sheet: 行=Model，列=Dataset
            acc_pivot = combined_df.pivot_table(index="Model", columns="Dataset", values="Accuracy", aggfunc="first")
            f1_pivot = combined_df.pivot_table(index="Model", columns="Dataset", values="F1-Score", aggfunc="first")

            acc_f1_values = []
            for model in acc_pivot.index:
                new_row = {"Model": model}
                for dataset in acc_pivot.columns:
                    acc = acc_pivot.loc[model, dataset]
                    f1 = f1_pivot.loc[model, dataset]

                    if pd.notna(acc) and pd.notna(f1):
                        new_row[dataset] = f"{f1:.4f}/{acc:.4f}"
                    else:
                        new_row[dataset] = "N/A"
                acc_f1_values.append(new_row)

            acc_f1_final_df = pd.DataFrame(acc_f1_values)
            acc_f1_final_df.to_excel(writer, index=False, sheet_name="F1_ACC")

        self.logger.info(f"指标结果已保存到 {excel_file}")
        return combined_df

    def get_file_name(self, image_path: str | Path) -> str:
        name = hashlib.md5(str(image_path).encode()).hexdigest()
        return f"{name}.json"

    def test(self) -> ProcessingResult:
        if not self._initialized:
            self.initialize()

        sorted_data = self.datasets_manager.detail_data.sort_values(by="image_path")
        grouped_image_info = sorted_data.groupby("image_path")

        if self.is_debug:
            # Debug模式: 单线程直接处理
            with tqdm(total=len(grouped_image_info), desc="总体分析进度") as pbar:
                for image_path, group_data in grouped_image_info:
                    # 计算当前bar的数量
                    if pbar.n >= 200:
                        break

                    unique_acc_count = group_data["acc_count"].nunique()
                    if unique_acc_count <= 1:
                        pbar.update(1)
                        continue

                    if (self.finally_save_dir / self.get_file_name(image_path)).exists():
                        pbar.update(1)
                        continue
                    try:
                        analysis_result = self.agent.think_and_act(image_path, 0)
                        self._save_result(image_path, group_data, analysis_result)
                    except Exception as e:
                        self.logger.exception(e)
                        self.logger.error(f"处理图像时出错: {e}")
                    pbar.update(1)
        else:
            self.test_concurrency(grouped_image_info)

    def test_concurrency(self, grouped_image_info):
        # 启动 Agent 的工作线程
        self.agent.start_workers()
        self.logger.info(f"并发数量:{self.agent.num_workflows}")
        # 创建结果队列
        result_queue = Queue()
        error_queue = Queue()
        try:
            total_items = len(grouped_image_info)
            # 使用两个进度条：一个用于收集/提交任务（position=0），一个用于运行/完成任务（position=1）
            with tqdm(total=total_items, desc="收集任务进度", position=0) as pbar_collect, tqdm(
                total=0, desc="运行进度", position=1
            ) as pbar_run:
                task_count = 0
                pending_tasks = {}  # {image_path: group_data}

                # 提交所有任务（更新收集进度条）
                for image_path, group_data in grouped_image_info:
                    if (self.finally_save_dir / self.get_file_name(image_path)).exists():
                        pbar_collect.update(1)
                        continue

                    # 轮询分配到不同的 workflow
                    workflow_id = task_count % self.agent.num_workflows
                    self.agent.submit_task(image_path, result_queue, error_queue, workflow_id)
                    pending_tasks[image_path] = group_data
                    task_count += 1
                    pbar_collect.update(1)

                # 设置运行进度条的总数为实际提交的任务数
                self.logger.info(f"已提交 {task_count} 个任务，开始收集结果...")
                self.logger.info(f"待处理任务列表: {len(pending_tasks.keys())}")
                pbar_run.total = task_count
                pbar_run.refresh()

                completed = 0
                while completed < task_count:
                    # 如果pending_tasks为空，跳出循环
                    if not pending_tasks:
                        self.logger.info("所有任务均已处理完毕")
                        break

                    # 检查结果队列
                    if not result_queue.empty():
                        image_path, analysis_result = result_queue.get()
                        group_data = pending_tasks.pop(image_path, None)
                        if image_path is None or group_data is None:
                            self.logger.error(f"未找到任务 {image_path} 的信息，跳过...")
                        else:
                            self._save_result(image_path, group_data, analysis_result)
                        completed += 1
                        pbar_run.update(1)
                    # 检查错误队列
                    elif not error_queue.empty():
                        image_path, error = error_queue.get()
                        pending_tasks.pop(image_path, None)
                        if image_path is None:
                            self.logger.error(f"未找到任务 {image_path} 的信息，跳过...")
                        else:
                            self.logger.error(f"处理图像 {image_path} 时出错: {error}")
                        completed += 1
                        pbar_run.update(1)
                    else:
                        time.sleep(1)  # 避免忙等待
        except Exception as e:
            self.logger.exception(f"收集结果时出错: {e}")
            raise e
        finally:
            # 确保关闭工作线程
            self.logger.info("正在关闭工作线程...")
            self.agent.shutdown_workers(wait=True)
            self.logger.info("所有任务已完成")
            # 检查是否有未完成的任务
            if pending_tasks:
                self.logger.warning(f"仍有 {len(pending_tasks)} 个任务未完成: {list(pending_tasks.keys())}")

    def _save_result(self, image_path, group_data, analysis_result):
        """保存结果的辅助方法"""
        gt_label = group_data["gt_label"].tolist()[0]
        final_response = analysis_result.get("final_response", None)
        if final_response is None:
            self.logger.error(f"图像 {image_path} 的分析结果中缺少 final_response，跳过保存")
            return

        if final_response.get("is_success", False) is False:
            self.logger.error(f"图像 {image_path} 的分析结果 final_response 标记为失败，跳过保存。 失败原因: {final_response}")
            return

        final_response["image_path"] = image_path
        final_response["gt_label"] = gt_label
        # model_res = group_data[["model_name", "pred_prob", "calibration_prob"]]
        # final_response["expert_model_pred"] = dict(zip(model_res["model_name"], model_res["pred_prob"]))
        # final_response["expert_model_calibrated_pred"] = dict(zip(model_res["model_name"], model_res["calibration_prob"]))

        # 保存消息
        save_file_name = self.get_file_name(image_path)
        with open(self.detail_save_dir / save_file_name, "w", encoding="utf-8") as f:
            msgs = [m.model_dump() for m in analysis_result["messages"] if m.content]
            json.dump(msgs, f, ensure_ascii=False, indent=4)

        # 保存最终结果
        with open(self.finally_save_dir / save_file_name, "w", encoding="utf-8") as f:
            json.dump(final_response, f, ensure_ascii=False, indent=4)


def main(args):
    app = AFAApplication(**vars(args))

    # 根据mode参数执行不同的功能
    if args.mode == "test":
        app.test()
    elif args.mode == "metrics":
        app.metrics_for_dataset(args.metrics_dir)
    elif args.mode == "only_metrics":
        app.metrics_for_dir(args.metrics_dir)
    elif args.mode == "analyze":
        if not args.image_path:
            raise ValueError("analyze模式需要提供--image_path参数")
        result = app.analyze_single_image(args.image_path)
        print(f"分析完成: {result}")
    elif args.mode == "semantic":
        app.get_semantic_info(args.semantic_save_dir)
    elif args.mode == "expert":
        app.get_expert_info()
    else:
        print(f"未知的模式: {args.mode}")
        print("可用模式: test, metrics, analyze, semantic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFA应用程序 - 图像鉴伪分析工具")
    parser.add_argument("--config_path", type=str, help="配置文件路径")
    parser.add_argument("--is_debug", action="store_true", default=False, help="启用调试模式")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "metrics", "only_metrics", "analyze", "semantic", "expert"],
        default="test",
        help="运行模式: test(批量测试), metrics(计算指标), analyze(分析单张图片), semantic(提取语义信息)",
    )
    parser.add_argument("--metrics_dir", type=str, default=None, help="metrics模式下的结果目录")
    parser.add_argument("--image_path", type=str, default=None, help="analyze模式下的图像路径")
    parser.add_argument("--semantic_save_dir", type=str, default=None, help="semantic模式下的保存路径")
    parser.add_argument("--max_workers", type=int, default=4, help="最大并发工作线程数")

    # 强制清除所有代理环境变量，确保不走代理
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(var, None)

    # 检查config_path存不存在
    main(parser.parse_args())

    # 使用示例:
    # unset http_proxy
    # unset https_proxy
    # unset HTTP_PROXY
    # unset HTTPS_PROXY
    # 批量测试: python agent_pipeline.py --mode test --config_path forensic_agent/configs/config_4b.yaml
    # 计算指标: python agent_pipeline.py --mode metrics --config_path forensic_agent/configs/config_4b.yaml
    # 分析单图: python agent_pipeline.py --mode analyze --image_path /path/to/image.jpg
    # 提取语义: python agent_pipeline.py --mode semantic --semantic_save_dir ./outputs/semantic
