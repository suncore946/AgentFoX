from enum import Enum
from pathlib import Path
import random
import pandas as pd

# 用法示例（需根据实际情况传入config和get_data函数）：
# sampler = DatasetSampler(CONFIG, get_data)
# sampler.sample_by_label()
# sampler.sample_by_dataset_name()


class SamplingMethod(Enum):
    ALL = "all"
    LABEL = "label"
    DATASET = "dataset"


class DatasetSampler:
    def __init__(self, config: dict, data, model_results, save_dir, random_state=42):
        """
        Args:
            config: 配置字典，需包含 'dataset'->'model_names'
            get_data_func: 获取数据的函数，返回 (data, model_results)
        """
        self.config = config
        self.data = data
        self.model_results = model_results
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def stratified_sample(self, group: pd.DataFrame, n_datasets, per_sample=None, total_samples=None):
        """对单个组进行分层采样，只返回image_path列表"""
        # 去重
        group = group.drop_duplicates(subset=["image_path"])
        rnd = random.Random(self.random_state)
        if total_samples is None:
            total_samples = per_sample * n_datasets

        if len(group) <= total_samples:
            print(f"组内样本量 {len(group)} 小于等于采样数 {total_samples}，全部保留。")
            return group["image_path"].tolist()

        samples_per_dataset = total_samples // n_datasets
        remaining_samples = total_samples % n_datasets

        sampled_image_paths = []
        dataset_pools = {}
        datasets = group["dataset_name"].unique()
        for dataset in datasets:
            dataset_group = group[group["dataset_name"] == dataset]
            samples = dataset_group["image_path"].tolist()
            rnd.shuffle(samples)  # 固定种子打乱，确保随机且可复现
            dataset_pools[dataset] = {
                "samples": samples,
                "target": samples_per_dataset,
                "allocated": 0,
            }

        shortage = 0
        for i, dataset in enumerate(datasets):
            pool = dataset_pools[dataset]
            if i < remaining_samples:
                pool["target"] += 1
            available = len(pool["samples"])
            can_allocate = min(pool["target"], available)
            if can_allocate > 0:
                allocated_samples = pool["samples"][:can_allocate]
                sampled_image_paths.extend(allocated_samples)
                pool["samples"] = pool["samples"][can_allocate:]
                pool["allocated"] = can_allocate
            if can_allocate < pool["target"]:
                shortage += pool["target"] - can_allocate

        if shortage > 0:
            surplus_datasets = [d for d in datasets if len(dataset_pools[d]["samples"]) > 0]
            round_count = 0
            while shortage > 0 and surplus_datasets:
                round_count += 1
                current_round_allocated = 0
                n_surplus = len(surplus_datasets)
                base_allocation = shortage // n_surplus
                extra_allocation = shortage % n_surplus
                datasets_to_remove = []
                for i, dataset in enumerate(surplus_datasets):
                    pool = dataset_pools[dataset]
                    allocation_this_round = base_allocation + (1 if i < extra_allocation else 0)
                    can_allocate = min(allocation_this_round, len(pool["samples"]))
                    if can_allocate > 0:
                        allocated_samples = pool["samples"][:can_allocate]
                        sampled_image_paths.extend(allocated_samples)
                        pool["samples"] = pool["samples"][can_allocate:]
                        pool["allocated"] += can_allocate
                        shortage -= can_allocate
                        current_round_allocated += can_allocate
                        print(f"第{round_count}轮: 数据集 '{dataset}' 补充了 {can_allocate} 个样本，剩余可用: {len(pool['samples'])}")
                    if len(pool["samples"]) == 0:
                        datasets_to_remove.append(dataset)
                for dataset in datasets_to_remove:
                    surplus_datasets.remove(dataset)
                if current_round_allocated == 0:
                    break

        if shortage > 0:
            print(f"仍有 {shortage} 个样本未分配，所有数据集样本均已用尽。")

        print(f"采样完成: 目标 {total_samples}, 实际 {len(sampled_image_paths)}")
        for dataset in datasets:
            pool = dataset_pools[dataset]
            print(f"  数据集 '{dataset}': 分配 {pool['allocated']} 个样本")

        return sampled_image_paths

    def sample_by_dataset(self, samples_per_combination=50, file_name="samples_by_dataset.csv", force_reload=True):
        stats_save_path = self.save_dir / file_name
        if stats_save_path.exists() and not force_reload:
            print(f"检测到已有采样结果文件 '{file_name}'，直接加载。")
            stats_df = pd.read_csv(stats_save_path)
            model_results = stats_df.pivot_table(index="image_path", columns="model_name", values="acc_count", fill_value=False).astype(
                bool
            )
            return stats_df, model_results

        merged = pd.merge(self.data, self.model_results, on="image_path", how="inner")
        merged = merged.drop_duplicates(subset=["image_path"])
        print(f"合并后数据量: {len(merged)}")

        group_cols = self.config["model_names"] + ["gt_label", "dataset_name"]
        sampled_groups = []
        total_groups = 0
        groups_with_insufficient_samples = 0

        for _, group in merged.groupby(group_cols):
            total_groups += 1
            if len(group) <= samples_per_combination:
                print(f"组内样本量 {len(group)} 小于等于采样数 {samples_per_combination}，全部保留。")
                sampled_groups.extend(group["image_path"].tolist())
                groups_with_insufficient_samples += 1
            else:
                sampled = group.sample(n=samples_per_combination, random_state=42)
                sampled_groups.extend(sampled["image_path"].tolist())

        print(f"总共分为 {total_groups} 组")
        print(f"其中 {groups_with_insufficient_samples} 组样本量不足，采用全部保留策略")
        print(f"剩余 {total_groups - groups_with_insufficient_samples} 组进行了随机采样")

        final_sampled = self.data[self.data["image_path"].isin(sampled_groups)].copy()
        final_sampled.to_csv(stats_save_path, index=False)
        print(f"最终采样数据量: {len(final_sampled)}")
        print(f"采样结果已保存为 '{stats_save_path}'")

        model_results = final_sampled.pivot_table(index="image_path", columns="model_name", values="acc_count", fill_value=False).astype(
            bool
        )
        return final_sampled, model_results

    def sample_by_label(self, per_sample=20, exist_content=None, file_name="samples_by_label.csv", force_reload=True):
        stats_save_path = self.save_dir / file_name

        if stats_save_path.exists() and not force_reload:
            print(f"检测到已有采样结果文件 '{file_name}'，直接加载。")
            stats_df = pd.read_csv(stats_save_path)
            model_results = stats_df.pivot_table(index="image_path", columns="model_name", values="acc_count", fill_value=False).astype(
                bool
            )
            return stats_df, model_results

        merged = pd.merge(self.data, self.model_results, on="image_path", how="inner")
        merged = merged.drop_duplicates(subset=["image_path"])
        print(f"合并后数据量: {len(merged)}")

        group_cols = self.config["model_names"] + ["gt_label"]
        if "Patch_Shuffle" in group_cols:
            group_cols.remove("Patch_Shuffle")  # 移除PatchShuffle以避免过度分组
            group_cols.append("PatchShuffle")  # 加入dataset_name以保持一定的分组粒度

        # 采样如果存在特定内容，则无视这部分的数据之后在进行采样
        if exist_content is not None:
            # exist_content为一个列表，包含需要剔除的image_path
            merged = merged[~merged["image_path"].isin(exist_content)]
            print(f"剔除包含 '{exist_content}' 的样本后，剩余数据量: {len(merged)}")

        sampled_groups = []
        n_datasets = merged["dataset_name"].nunique()
        for _, group in merged.groupby(group_cols):
            sampled_groups.extend(self.stratified_sample(group.copy(), n_datasets=n_datasets, per_sample=per_sample))

        final_sampled = self.data[self.data["image_path"].isin(sampled_groups)].copy()
        final_sampled.to_csv(stats_save_path, index=False)
        print(f"最终采样数据量: {len(final_sampled)}")
        print(f"采样结果已保存为 '{stats_save_path}'")
        # 根据image_path去重, 然后统计每个dataset_name下real和fake的样本数量
        model_results = final_sampled.pivot_table(index="image_path", columns="model_name", values="acc_count", fill_value=False).astype(
            bool
        )
        return final_sampled, model_results
