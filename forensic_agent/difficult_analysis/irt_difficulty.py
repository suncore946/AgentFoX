"""
IRT（心理测量学）难度模块
实现基于2PL模型的可解释难度度量 - GPU强制版本
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any

from ..config.settings import DifficultyConfig
from ..utils.difficult_utils import filter_extreme_samples, compute_correlation_metrics, safe_divide
from ..utils.logger import get_logger

import torch
import torch.nn.functional as F


class IRTDifficultyEstimator:
    """IRT难度估计器 - GPU强制版本"""

    def __init__(self, config: Optional[DifficultyConfig] = None):
        """
        初始化IRT难度估计器

        Args:
            config: 配置对象
        """
        self.config = config or DifficultyConfig()
        self.logger = get_logger(__name__)
        self.is_fitted = False
        self.item_params = None
        self.person_params = None
        self.fit_info = None

        # 强制GPU设置
        self._setup_device()

    def _setup_device(self):
        """设置GPU设备（强制GPU）"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU不可用，此版本需要GPU支持")

        self.device = torch.device("cuda")
        self.use_gpu = True
        self.use_amp = True  # 默认启用混合精度

        self.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        self.logger.info(f"GPU内存: {gpu_memory:.1f} GB")

    def _to_tensor(self, array: np.ndarray, dtype=None) -> torch.Tensor:
        """将numpy数组转换为GPU tensor"""
        if dtype is None:
            dtype = torch.float32
        return torch.from_numpy(array).to(self.device, dtype=dtype)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将GPU tensor转换为numpy数组"""
        return tensor.detach().cpu().numpy()

    def get_device_info(self) -> Dict[str, Any]:
        """
        获取GPU设备信息

        Returns:
            设备信息字典
        """
        info = {
            "torch_version": torch.__version__,
            "device": str(self.device),
            "use_gpu": True,
            "use_amp": self.use_amp,
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
        }

        gpu_props = torch.cuda.get_device_properties(0)
        info["gpu_memory_total_gb"] = gpu_props.total_memory / (1024**3)
        info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        info["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / (1024**2)

        return info

    def fit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        filter_extreme: bool = True,
        max_samples: Optional[int] = None,
        use_fast_init: bool = True,
        use_amp: bool = True,  # 是否使用混合精度
    ) -> Dict[str, Any]:
        """
        拟合2PL IRT模型（GPU版本）

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)
            labels: 真实标签 (n_samples,)
            filter_extreme: 是否过滤极端样本（全对或全错）
            max_samples: 最大样本数量限制（用于大数据集）
            use_fast_init: 是否使用快速初始化
            use_amp: 是否使用混合精度

        Returns:
            拟合结果字典
        """
        self.use_amp = use_amp

        # 数据预处理优化：大数据集采样
        if max_samples and len(predictions) > max_samples:
            self.logger.info(f"数据集过大({len(predictions)}样本)，采样到{max_samples}样本以提高速度")
            sample_indices = np.random.choice(len(predictions), size=max_samples, replace=False)
            predictions = predictions[sample_indices]
            labels = labels[sample_indices]

        # 构造正确性矩阵
        response_matrix = self._construct_response_matrix(predictions, labels)

        # 过滤极端样本（可选）
        if filter_extreme:
            response_matrix, labels_filtered, valid_indices = filter_extreme_samples(
                response_matrix, labels, min_variance_threshold=self.config.irt_min_variance_threshold
            )
        else:
            labels_filtered = labels
            valid_indices = np.arange(len(labels))

        n_items, n_persons = response_matrix.shape

        if n_items == 0 or n_persons < 2:
            raise ValueError(f"数据不足以拟合IRT模型：{n_items} 样本，{n_persons} 模型")

        # GPU拟合2PL模型
        self.logger.info("使用GPU进行IRT模型拟合")
        item_params, person_params, fit_info = self._fit_2pl_model_gpu(response_matrix, use_fast_init)

        # 存储结果
        self.item_params = item_params
        self.person_params = person_params
        self.fit_info = fit_info
        self.is_fitted = True

        # 准备完整结果（包含被过滤的样本）
        full_item_params = self._expand_to_full_samples(item_params, valid_indices, len(labels))

        # 计算拟合质量指标
        quality_metrics = self._calculate_fit_quality_gpu(response_matrix, item_params, person_params)

        # 诊断分析
        diagnostics = self._perform_diagnostics(item_params, person_params, response_matrix)

        return {
            "item_difficulties": full_item_params["difficulties"],
            "item_discriminations": full_item_params["discriminations"],
            "model_abilities": person_params["abilities"],
            "valid_indices": valid_indices,
            "filtered_samples": len(labels) - len(valid_indices),
            "fit_info": fit_info,
            "quality_metrics": quality_metrics,
            "diagnostics": diagnostics,
        }

    def _construct_response_matrix(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        构造正确性矩阵 R[i,m]

        Args:
            predictions: 预测概率矩阵 (n_samples, n_models)
            labels: 真实标签 (n_samples,)

        Returns:
            正确性矩阵 (n_samples, n_models)
        """
        # 转换为二值预测
        binary_predictions = (predictions > 0.5).astype(int)

        # 构造正确性矩阵：预测正确为1，错误为0
        response_matrix = (binary_predictions == labels.reshape(-1, 1)).astype(int)

        return response_matrix

    def _smart_initialization(self, response_matrix: np.ndarray) -> np.ndarray:
        """
        智能初始化：基于数据特征的更好初始值

        Args:
            response_matrix: 响应矩阵 (n_items, n_persons)

        Returns:
            初始参数向量
        """
        n_items, n_persons = response_matrix.shape
        n_params = 2 * n_items + n_persons
        initial_params = np.zeros(n_params)

        # 1. 辨别度初始化：基于项目方差
        item_variances = np.var(response_matrix, axis=1)
        discriminations_init = 0.5 + 2.0 * item_variances
        discriminations_init = np.clip(discriminations_init, 0.2, 3.0)
        initial_params[:n_items] = discriminations_init

        # 2. 难度初始化：改进的logit转换
        pass_rates = np.mean(response_matrix, axis=1)
        pass_rates = np.clip(pass_rates, 0.05, 0.95)
        difficulties_init = -np.log(pass_rates / (1 - pass_rates))
        difficulties_init = difficulties_init * 0.8
        initial_params[n_items : 2 * n_items] = difficulties_init

        # 3. 能力初始化：基于人员正确率
        person_correct_rates = np.mean(response_matrix, axis=0)
        person_correct_rates = np.clip(person_correct_rates, 0.05, 0.95)
        from scipy.stats import norm

        abilities_init = norm.ppf(person_correct_rates) * 0.5
        abilities_init = abilities_init - np.mean(abilities_init)
        initial_params[2 * n_items :] = abilities_init

        return initial_params

    def _fit_2pl_model_gpu(self, response_matrix: np.ndarray, use_fast_init: bool = True) -> Tuple[Dict, Dict, Dict]:
        """
        使用GPU拟合2PL模型：P(R=1) = sigmoid(a_i (θ_m - b_i))

        Args:
            response_matrix: 正确性矩阵 (n_items, n_persons)
            use_fast_init: 是否使用快速初始化

        Returns:
            (项目参数, 人员参数, 拟合信息)
        """
        n_items, n_persons = response_matrix.shape
        self.logger.info(f"开始GPU优化，问题规模: {n_items} items × {n_persons} persons")

        # 转换到GPU tensor
        response_tensor = self._to_tensor(response_matrix.astype(np.float32))

        # 初始化参数（在GPU上）
        if use_fast_init:
            initial_params_np = self._smart_initialization(response_matrix)
        else:
            n_params = 2 * n_items + n_persons
            initial_params_np = np.zeros(n_params)
            initial_params_np[:n_items] = 1.0
            pass_rates = np.mean(response_matrix, axis=1)
            pass_rates = np.clip(pass_rates, 0.01, 0.99)
            initial_params_np[n_items : 2 * n_items] = -np.log(pass_rates / (1 - pass_rates))

        # 转换参数到GPU
        params_tensor = self._to_tensor(initial_params_np, dtype=torch.float32)
        params_tensor.requires_grad_(True)

        # 设置优化器
        optimizer = torch.optim.AdamW([params_tensor], lr=0.01, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        # 设置混合精度
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        best_loss = float("inf")
        best_params = None
        patience_counter = 0
        max_patience = 50

        # 优化循环
        for epoch in range(self.config.irt_max_iter):
            # Adam/AdamW优化器
            optimizer.zero_grad()

            if self.use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = self._negative_log_likelihood_gpu(params_tensor, response_tensor, n_items, n_persons)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self._negative_log_likelihood_gpu(params_tensor, response_tensor, n_items, n_persons)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(loss)

            # 应用参数约束
            with torch.no_grad():
                # 约束辨别度 > 0.1
                params_tensor[:n_items] = torch.clamp(params_tensor[:n_items], min=0.1, max=5.0)
                # 约束难度和能力在合理范围
                params_tensor[n_items:] = torch.clamp(params_tensor[n_items:], min=-5.0, max=5.0)

                # 标准化能力参数（均值为0）
                abilities = params_tensor[2 * n_items :]
                abilities_mean = torch.mean(abilities)
                params_tensor[2 * n_items :] = abilities - abilities_mean

            current_loss = loss.item()

            # 早停检查
            if current_loss < best_loss - self.config.irt_tolerance:
                best_loss = current_loss
                best_params = params_tensor.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                self.logger.info(f"GPU优化早停于第 {epoch+1} 轮，损失: {best_loss:.6f}")
                break

            if (epoch + 1) % 100 == 0:
                lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else "N/A"
                self.logger.info(f"GPU优化第 {epoch+1} 轮，损失: {current_loss:.6f}, 学习率: {lr}")

        # 使用最佳参数
        if best_params is not None:
            final_params = self._to_numpy(best_params)
        else:
            final_params = self._to_numpy(params_tensor.detach())

        # 提取和后处理参数
        discriminations = final_params[:n_items]
        difficulties = final_params[n_items : 2 * n_items]
        abilities = final_params[2 * n_items :]

        # 最终标准化
        abilities = (abilities - np.mean(abilities)) / (np.std(abilities) + 1e-8)

        item_params = {"discriminations": discriminations, "difficulties": difficulties}
        person_params = {"abilities": abilities}

        fit_info = {
            "converged": True,
            "final_likelihood": -best_loss,
            "n_iterations": epoch + 1,
            "optimization_message": "GPU optimization completed",
            "n_items": n_items,
            "n_persons": n_persons,
            "device": str(self.device),
            "use_amp": self.use_amp,
        }

        return item_params, person_params, fit_info

    def _negative_log_likelihood_gpu(
        self, params_tensor: torch.Tensor, response_tensor: torch.Tensor, n_items: int, n_persons: int
    ) -> torch.Tensor:
        """
        GPU版本的负对数似然计算

        Args:
            params_tensor: 参数张量 [a_1...a_I, b_1...b_I, θ_1...θ_P]
            response_tensor: 响应张量 (n_items, n_persons)
            n_items: 项目数量
            n_persons: 人员数量

        Returns:
            负对数似然值（张量）
        """
        # 提取参数
        discriminations = params_tensor[:n_items].unsqueeze(1)  # (n_items, 1)
        difficulties = params_tensor[n_items : 2 * n_items].unsqueeze(1)  # (n_items, 1)
        abilities = params_tensor[2 * n_items :].unsqueeze(0)  # (1, n_persons)

        # 向量化计算logit: a_i * (θ_j - b_i)
        logits = discriminations * (abilities - difficulties)  # (n_items, n_persons)

        # 使用PyTorch的数值稳定sigmoid
        probs = torch.sigmoid(logits)

        # 数值稳定性
        eps = 1e-10
        probs = torch.clamp(probs, eps, 1 - eps)

        # 计算负对数似然
        log_likelihood = torch.sum(response_tensor * torch.log(probs) + (1 - response_tensor) * torch.log(1 - probs))

        return -log_likelihood

    def _expand_to_full_samples(self, item_params: Dict, valid_indices: np.ndarray, total_samples: int) -> Dict:
        """
        将项目参数扩展到全部样本（为被过滤的样本填充默认值）

        Args:
            item_params: 有效样本的项目参数
            valid_indices: 有效样本索引
            total_samples: 总样本数

        Returns:
            扩展后的项目参数
        """
        full_difficulties = np.full(total_samples, np.nan)
        full_discriminations = np.full(total_samples, np.nan)

        full_difficulties[valid_indices] = item_params["difficulties"]
        full_discriminations[valid_indices] = item_params["discriminations"]

        # 为被过滤的样本填充默认值（极难或极易）
        filtered_mask = np.isnan(full_difficulties)
        if np.any(filtered_mask):
            full_difficulties[filtered_mask] = 5.0  # 极难
            full_discriminations[filtered_mask] = 0.1  # 低辨别度

        return {"difficulties": full_difficulties, "discriminations": full_discriminations}

    def _calculate_fit_quality_gpu(self, response_matrix: np.ndarray, item_params: Dict, person_params: Dict) -> Dict[str, float]:
        """
        GPU版本的拟合质量计算

        Args:
            response_matrix: 响应矩阵
            item_params: 项目参数
            person_params: 人员参数

        Returns:
            拟合质量指标字典
        """
        n_items, n_persons = response_matrix.shape

        # 转换到GPU
        response_tensor = self._to_tensor(response_matrix.astype(np.float32))
        discriminations = self._to_tensor(item_params["discriminations"].astype(np.float32))
        difficulties = self._to_tensor(item_params["difficulties"].astype(np.float32))
        abilities = self._to_tensor(person_params["abilities"].astype(np.float32))

        with torch.no_grad():
            # 使用广播计算logit矩阵
            abilities_matrix = abilities.unsqueeze(0)  # (1, n_persons)
            discriminations_matrix = discriminations.unsqueeze(1)  # (n_items, 1)
            difficulties_matrix = difficulties.unsqueeze(1)  # (n_items, 1)

            logits = discriminations_matrix * (abilities_matrix - difficulties_matrix)

            # 计算预测概率
            predicted_probs = torch.sigmoid(logits)

            # 转换为展平张量进行统计计算
            observed = response_tensor.flatten()
            predicted = predicted_probs.flatten()

            # 均方根误差
            rmse = torch.sqrt(torch.mean((observed - predicted) ** 2))

            # 相关系数
            observed_mean = torch.mean(observed)
            predicted_mean = torch.mean(predicted)
            numerator = torch.sum((observed - observed_mean) * (predicted - predicted_mean))
            denominator = torch.sqrt(torch.sum((observed - observed_mean) ** 2) * torch.sum((predicted - predicted_mean) ** 2))
            correlation = numerator / (denominator + 1e-8)

            # 对数似然
            eps = 1e-10
            predicted_clipped = torch.clamp(predicted, eps, 1 - eps)
            log_likelihood = torch.sum(observed * torch.log(predicted_clipped) + (1 - observed) * torch.log(1 - predicted_clipped))

        # 转换回CPU numpy
        rmse_val = self._to_numpy(rmse).item()
        correlation_val = self._to_numpy(correlation).item()
        log_likelihood_val = self._to_numpy(log_likelihood).item()

        # AIC和BIC
        n_params = 2 * n_items + n_persons
        n_observations = n_items * n_persons
        aic = -2 * log_likelihood_val + 2 * n_params
        bic = -2 * log_likelihood_val + np.log(n_observations) * n_params

        return {
            "rmse": rmse_val,
            "correlation": correlation_val if not np.isnan(correlation_val) else 0.0,
            "log_likelihood": log_likelihood_val,
            "aic": aic,
            "bic": bic,
            "n_parameters": n_params,
            "n_observations": n_observations,
            "device": str(self.device),
        }

    def _perform_diagnostics(self, item_params: Dict, person_params: Dict, response_matrix: np.ndarray) -> Dict[str, Any]:
        """
        执行诊断分析

        Args:
            item_params: 项目参数
            person_params: 人员参数
            response_matrix: 响应矩阵

        Returns:
            诊断结果字典
        """
        difficulties = item_params["difficulties"]
        discriminations = item_params["discriminations"]
        abilities = person_params["abilities"]

        # 项目参数诊断
        item_diagnostics = {
            "difficulty_mean": np.mean(difficulties),
            "difficulty_std": np.std(difficulties),
            "difficulty_range": [np.min(difficulties), np.max(difficulties)],
            "discrimination_mean": np.mean(discriminations),
            "discrimination_std": np.std(discriminations),
            "discrimination_range": [np.min(discriminations), np.max(discriminations)],
            "low_discrimination_items": np.sum(discriminations < 0.5),
            "high_discrimination_items": np.sum(discriminations > 2.0),
        }

        # 人员参数诊断
        person_diagnostics = {
            "ability_mean": np.mean(abilities),
            "ability_std": np.std(abilities),
            "ability_range": [np.min(abilities), np.max(abilities)],
        }

        # 异常项目检测
        unusual_items = []
        for i in range(len(difficulties)):
            if discriminations[i] < 0.2:
                unusual_items.append(
                    {
                        "item_index": i,
                        "issue": "very_low_discrimination",
                        "discrimination": discriminations[i],
                        "difficulty": difficulties[i],
                    }
                )

        return {
            "item_diagnostics": item_diagnostics,
            "person_diagnostics": person_diagnostics,
            "unusual_items": unusual_items,
            "n_unusual_items": len(unusual_items),
        }

    def compare_with_other_difficulty(
        self, irt_results: Dict[str, Any], other_difficulty: np.ndarray, other_name: str = "other_difficulty"
    ) -> Dict[str, Any]:
        """
        与其他难度度量进行一致性检验

        Args:
            irt_results: IRT拟合结果
            other_difficulty: 其他难度度量
            other_name: 其他度量的名称

        Returns:
            一致性检验结果
        """
        irt_difficulties = irt_results["item_difficulties"]

        # 只比较有效样本
        valid_mask = ~np.isnan(irt_difficulties)

        if np.sum(valid_mask) < 10:
            raise ValueError("有效样本太少，无法进行可靠的一致性检验")

        irt_valid = irt_difficulties[valid_mask]
        other_valid = other_difficulty[valid_mask]

        # 计算相关性指标
        correlation_metrics = compute_correlation_metrics(irt_valid, other_valid)

        # 一致性分析
        agreement_analysis = {}
        for k in [10, 50, 100]:
            if len(irt_valid) >= k:
                irt_top_k = np.argsort(irt_valid)[-k:]
                other_top_k = np.argsort(other_valid)[-k:]
                overlap = len(set(irt_top_k) & set(other_top_k))
                agreement_analysis[f"top_{k}_overlap"] = overlap / k

        # 分位数一致性
        for q in [0.9, 0.95, 0.99]:
            irt_threshold = np.quantile(irt_valid, q)
            other_threshold = np.quantile(other_valid, q)

            irt_high = irt_valid > irt_threshold
            other_high = other_valid > other_threshold

            agreement = np.sum(irt_high == other_high) / len(irt_valid)
            agreement_analysis[f"quantile_{int(q*100)}_agreement"] = agreement

        return {
            "comparison_name": other_name,
            "n_valid_samples": np.sum(valid_mask),
            "correlation_metrics": correlation_metrics,
            "agreement_analysis": agreement_analysis,
            "message": "Comparison completed successfully",
        }

    def create_irt_dataframe(self, irt_results: Dict[str, Any], image_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        将IRT结果转换为DataFrame格式

        Args:
            irt_results: IRT拟合结果
            image_ids: 图像ID数组

        Returns:
            包含IRT参数的DataFrame
        """
        n_samples = len(irt_results["item_difficulties"])

        # 创建基础DataFrame
        df_data = {
            "irt_difficulty": irt_results["item_difficulties"],
            "irt_discrimination": irt_results["item_discriminations"],
            "is_valid_irt": ~np.isnan(irt_results["item_difficulties"]),
        }

        # 添加图像ID
        if image_ids is not None:
            if len(image_ids) != n_samples:
                raise ValueError(f"image_ids长度({len(image_ids)})与样本数量({n_samples})不匹配")
            df_data["image_id"] = image_ids
        else:
            df_data["image_id"] = np.arange(n_samples)

        df = pd.DataFrame(df_data)

        # 重新排列列顺序
        columns_order = ["image_id", "irt_difficulty", "irt_discrimination", "is_valid_irt"]
        df = df[columns_order]

        return df

    def print_irt_summary(self, irt_results: Dict[str, Any]):
        """
        打印IRT结果摘要

        Args:
            irt_results: IRT拟合结果
        """
        self.logger.info("=== IRT难度度量摘要 (GPU版本) ===")

        fit_info = irt_results["fit_info"]
        print(f"拟合状态: {'成功' if fit_info['converged'] else '失败'}")
        print(f"样本数量: {fit_info['n_items']}")
        print(f"模型数量: {fit_info['n_persons']}")
        print(f"迭代次数: {fit_info['n_iterations']}")
        print(f"过滤样本: {irt_results['filtered_samples']}")
        print(f"计算设备: {fit_info['device']}")
        print(f"混合精度: {'启用' if fit_info['use_amp'] else '禁用'}")

        if "quality_metrics" in irt_results:
            quality = irt_results["quality_metrics"]
            print(f"\n拟合质量:")
            print(f"  RMSE: {quality['rmse']:.4f}")
            print(f"  相关系数: {quality['correlation']:.4f}")
            print(f"  对数似然: {quality['log_likelihood']:.2f}")
            print(f"  AIC: {quality['aic']:.2f}")
            print(f"  BIC: {quality['bic']:.2f}")
            print(f"  计算设备: {quality['device']}")

        if "diagnostics" in irt_results:
            diag = irt_results["diagnostics"]
            print(f"\n项目参数统计:")
            item_diag = diag["item_diagnostics"]
            print(f"  难度: {item_diag['difficulty_mean']:.3f}±{item_diag['difficulty_std']:.3f}")
            print(f"  辨别度: {item_diag['discrimination_mean']:.3f}±{item_diag['discrimination_std']:.3f}")
            print(f"  低辨别度项目: {item_diag['low_discrimination_items']}")
            print(f"  高辨别度项目: {item_diag['high_discrimination_items']}")

            if diag["n_unusual_items"] > 0:
                print(f"\n⚠️ 发现 {diag['n_unusual_items']} 个异常项目")
