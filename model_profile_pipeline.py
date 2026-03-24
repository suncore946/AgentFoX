import json
from pathlib import Path
import statistics
from typing import Optional
from cfg import CONFIG
from forensic_agent.data_operation.dataset_loader import load_project_data

# 计算ACC和F1-Score
from sklearn.metrics import accuracy_score, f1_score


class ModelProfilePipeline:
    def __init__(self):
        dataset = CONFIG.get("dataset", {})

        self.db_dir = dataset["db_dir"]
        self.model_names = dataset["model_names"]
        self.dataset_names = "WIRA"
        self.dataset_root = dataset.get("datasets_root")
        self.data, self.dataset_processor = self._load_and_process_data(self.dataset_root)
        self.model_profile_path = Path("/data2/yuyangxin/Agent/forensic_agent/configs/profiles/model_profiles.json")
        assert self.model_profile_path.exists(), f"Model profile path {self.model_profile_path} does not exist."

    @staticmethod
    def compute_metrics(gt_labels, pred_probs, threshold=0.5):
        pred_labels = [1 if p >= threshold else 0 for p in pred_probs]
        acc = accuracy_score(gt_labels, pred_labels)
        f1 = f1_score(gt_labels, pred_labels)
        return round(float(acc), 4), round(float(f1), 4)

    def _load_and_process_data(self, dataset_root: str, max_num: Optional[int] = None) -> tuple:
        """加载和处理数据"""
        # 使用统一数据加载器
        data, processor = load_project_data(
            datasets_root=dataset_root,
            db_dir=self.db_dir,
            max_num=max_num,
            dataset_names=self.dataset_names,
            model_names=self.model_names,
        )
        return data, processor

    def metrics_evaluation(self, data):
        """评估指标"""
        # 计算各模型的准确率
        pred_res = {}
        cal_res = {}
        for model_name, group in data.groupby("model_name"):
            # 根据image_path去重，计算准确率和F1分数
            unique_images = group.drop_duplicates(subset=["image_path"])
            gt_labels = unique_images["gt_label"].to_list()
            pred_probs = unique_images["pred_prob"].to_list()
            # 保证按列名检查
            calibration_probs = unique_images["calibration_prob"].to_list() if "calibration_prob" in unique_images.columns else None

            acc, f1 = self.compute_metrics(gt_labels, pred_probs)
            pred_res[model_name] = {"accuracy": acc, "f1_score": f1}

            if calibration_probs is not None:
                cal_acc, cal_f1 = self.compute_metrics(gt_labels, calibration_probs)
            else:
                cal_acc, cal_f1 = None, None
            cal_res[model_name] = {"accuracy": cal_acc, "f1_score": cal_f1}

        # 根据准确率排序并只保留 model 名（处理 None 值）
        def sort_keys_by_metric(res_dict, metric):
            items = [(name, v) for name, v in res_dict.items() if v.get(metric) is not None]
            ranked = [name for name, _ in sorted(items, key=lambda x: x[1][metric], reverse=True)]
            # 将未提供 metric 的模型放到后面（保持原始顺序）
            missing = [name for name, v in res_dict.items() if v.get(metric) is None]
            return ranked + missing

        # 计算统计量（均值、方差、标准差），结果保留4位小数
        def compute_stats_for(res_dict, metric, ranked_list):
            vals = [v[metric] for v in res_dict.values() if v.get(metric) is not None]
            if not vals:
                return {"rank": ranked_list, "variance": None, "mean": None, "std_dev": None}
            mean_v = statistics.mean(vals)
            var_v = statistics.pvariance(vals)  # population variance
            std_v = statistics.pstdev(vals)  # population std dev
            return {
                "rank": ranked_list,
                "variance": round(float(var_v), 4),
                "mean": round(float(mean_v), 4),
                "std": round(float(std_v), 4),
            }

        acc_ranked_pred = sort_keys_by_metric(pred_res, "accuracy")
        acc_ranked_cal = sort_keys_by_metric(cal_res, "accuracy")
        f1_ranked_pred = sort_keys_by_metric(pred_res, "f1_score")
        f1_ranked_cal = sort_keys_by_metric(cal_res, "f1_score")

        # 读取现有 profile（若不存在则初始化），然后写回更新后的内容
        with open(self.model_profile_path, "r", encoding="utf-8") as f:
            model_profile = json.load(f)

        model_profile["model_evaluation"] = {
            "description": "Model Performance Evaluation in GenImage AIGC Detection Dataset(In-domain)",
            "prediction": {
                "description": "Model performance on original prediction probabilities.",
                "data": pred_res,
                "accuracy_rank": compute_stats_for(pred_res, "accuracy", acc_ranked_pred),
                "f1_score_rank": compute_stats_for(pred_res, "f1_score", f1_ranked_pred),
            },
            "calibration": {
                "description": "Model performance on calibrated probabilities. The calibration set is GenImage itself.",
                "data": cal_res,
                "accuracy_rank": compute_stats_for(cal_res, "accuracy", acc_ranked_cal),
                "f1_score_rank": compute_stats_for(cal_res, "f1_score", f1_ranked_cal),
            },
        }

        # 保存回文件
        with open(self.model_profile_path, "w", encoding="utf-8") as f:
            json.dump(model_profile, f, ensure_ascii=False, indent=2)

        return model_profile


if __name__ == "__main__":
    pipeline = ModelProfilePipeline()
    eval_results = pipeline.metrics_evaluation(pipeline.data)
    print("Evaluation Results:", eval_results)
