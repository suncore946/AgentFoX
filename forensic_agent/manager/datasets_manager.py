from pathlib import Path
from typing import List
from loguru import logger
import pandas as pd
from ..data_operation.dataset_loader import load_project_data


class DatasetsManager:
    def __init__(self, config: dict):
        """
        初始化 ForensicDatasets 类

        Args:
            config (dict): 配置字典，包含 test_paths 或 dataset_names/db_dir
        """
        assert isinstance(config, dict), "配置必须是一个字典"
        # 要么包含 test_paths，要么同时包含 dataset_names 和 db_dir
        assert "test_paths" in config or (
            "dataset_names" in config and "db_dir" in config
        ), "配置中必须包含 'test_paths' 或同时包含 'dataset_names' 和 'db_dir'"
        self.config = config
        self.expert_models: List = config.get("expert_models", ["DRCT", "SPAI", "RINE", "PatchShuffle"])
        self.test_paths: str = config.get("test_paths", None)

        # 扩展实现数据库的相关内容
        self._detail_data, self._clustering_data = self.load_data(
            test_paths=self.test_paths,
            model_names=self.expert_models,
            dataset_names=config.get("dataset_names", None),
            db_dir=config.get("db_dir", None),
        )

    @property
    def detail_data(self) -> pd.DataFrame:
        # 将_detail_data根据image_path和model_name进行去重
        info: pd.DataFrame = self._detail_data.drop_duplicates(subset=["image_path", "model_name"])
        # 如果model_name列下有名为vllm的, 去掉vllm
        return info[info["model_name"] != "vllm"]

    @property
    def clustering_data(self) -> pd.DataFrame:
        info: pd.DataFrame = self._clustering_data.drop_duplicates(subset=["image_path"])
        return info

    @staticmethod
    def load_data(test_paths, model_names, dataset_names=None, db_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        统一的数据加载方法

        Args:
            config: 配置字典，可包含 test_paths 或 dataset_names/db_dir

        Returns:
            tuple: (detail_data, clustering_data)
        """
        if test_paths:
            # 从测试文件加载数据
            test_data: pd.DataFrame = pd.read_csv(test_paths)
            if test_data.empty:
                logger.warning(f"测试数据文件为空: {test_paths}")
                return pd.DataFrame(), pd.DataFrame()

            if model_names:
                test_data = test_data[test_data["model_name"].isin(model_names)]
            if test_data.empty:
                raise ValueError(f"测试数据中不包含指定的模型: {model_names}")

            # 确保image_path列存在
            if "image_path" not in test_data.columns:
                logger.error("测试数据缺少必需的image_path列")
                raise ValueError("Missing required column: image_path")

            # 聚类数据
            ignore_cols = ["cluster_LaplacianVar&EdgeDensity&HF_LF_Ratio"]
            cluster_cols = [col for col in test_data.columns if col.startswith("cluster_") and col not in ignore_cols]

            # 详细数据
            detail_cols = [col for col in test_data.columns if not col.startswith("cluster_")]

            return test_data[detail_cols], test_data[["image_path"] + cluster_cols]
        else:
            # 从项目数据库加载数据
            detail_data, db_loader = load_project_data(
                dataset_names=dataset_names,
                db_dir=db_dir,
                model_names=model_names,
            )
            clustering_data = db_loader.db_manager.load_clustering_results(dataset_names=dataset_names)
            return detail_data, clustering_data

    def save_detail_data(self, data: pd.DataFrame, save_name):
        if self.test_paths:
            target_path = Path(self.test_paths).parent / f"{save_name}.csv"
            # 如果目标文件已存在，则先备份
            if target_path.exists():
                backup_path = target_path.with_suffix(target_path.suffix + ".backup")
                target_path.replace(backup_path)
                logger.info(f"已备份旧文件到 {backup_path}")
            data.to_csv(target_path, index=False)
            logger.info(f"详细数据已保存到 {target_path}")
            self._detail_data = data
        else:
            raise NotImplementedError("不是在待测试数据模式下，暂不支持保存详细数据。")

    def filtration(self, image_paths: List[str]) -> pd.DataFrame:
        """
        根据给定的图像路径列表过滤详细数据

        Args:
            image_paths (List[str]): 图像路径列表

        Returns:
            pd.DataFrame: 过滤后的详细数据
        """
        logger.info(f"正在根据提供的图像路径过滤数据，共 {len(image_paths)} 张图像。")
        self._detail_data = self.detail_data[self.detail_data["image_path"].isin(image_paths)]
        self._clustering_data = self.clustering_data[self.clustering_data["image_path"].isin(image_paths)]
        logger.info(f"过滤后剩余 {len(self._detail_data)} 条详细数据记录。")

    def get_image_and_label(self) -> List[str]:
        """
        获取当前详细数据中的所有图像路径

        Returns:
            List[str]: 图像路径列表
        """
        # 要image_path列和gt_label列, 转为下属dict
        # {"image_path": {"gt_label":0/1}}
        ret: pd.DataFrame = self.detail_data[["image_path", "gt_label"]].drop_duplicates().set_index("image_path")
        # 转为字典形式返回, key是image_path, value是{"gt_label":0/1}
        ret = ret["gt_label"].to_dict()
        for k, v in ret.items():
            ret[k] = {"gt_label": v}
        return ret
