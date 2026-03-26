from typing import Dict, List, Tuple, Union
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from netcal.metrics import ECE, MCE, ACE

# 使用 sklearn 的 log_loss 函数
from sklearn.metrics import log_loss as sklearn_log_loss

from ..utils import safe_division


class CalibrationMetrics:
    """
    专门用于计算校准相关指标的工具类。
    包含 ECE、MCE、Brier Score、ACE、BIC 等校准指标的计算。
    """

    # 类常量
    DEFAULT_BINS = 20
    DEFAULT_THRESHOLD = 0.5
    EPSILON = 1e-15
    BOOTSTRAP_ROUNDS: int = 1000
    CONFIDENCE_LEVEL: float = 0.95

    @staticmethod
    def _format_metric(x):
        """将数值格式化为保留4位小数；None 保持 None。"""
        if x is None:
            return None
        try:
            return round(float(x), 4)
        except Exception:
            return x

    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = DEFAULT_BINS) -> float:
        """
        计算期望校准误差 (Expected Calibration Error)
        作用：衡量模型预测概率的整体校准程度
        将模型的预测概率空间划分为M个区间（bin），计算每个区间内平均预测概率与真实标签发生频率之间的绝对差值，然后对这些差值进行加权平均。

        流程:
        - 将预测概率空间分成若干区间（bins）
        - 计算每个区间内预测概率与实际发生频率的绝对差值
        - 对所有区间的差值进行加权平均
        - 值越小表示校准越好

        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测概率
            n_bins: 分箱数量

        Returns:
            ECE值
        """
        if len(y_true) == 0 or len(y_prob) == 0:
            return 0.0

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")

        # 使用 netcal 的 ECE
        ece_metric = ECE(bins=n_bins)
        ece_value = ece_metric.measure(y_prob, y_true)
        return float(ece_value)

    @staticmethod
    def calculate_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = DEFAULT_BINS) -> float:
        """
        计算最大校准误差 (Maximum Calibration Error): 衡量模型预测概率的最坏校准情况

        -找出所有区间中校准误差最大的那个
        -反映模型在某些概率区间的极端错误校准
        -值越小表示校准越好

        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测概率
            n_bins: 分箱数量

        Returns:
            MCE值
        """
        if len(y_true) == 0 or len(y_prob) == 0:
            return 0.0

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")

        # 使用 netcal 的 MCE
        mce_metric = MCE(bins=n_bins)
        mce_value = mce_metric.measure(y_prob, y_true)
        return float(mce_value)

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        计算Brier Score:计算预测概率与真实标签之间的平方误差

        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测概率

        Returns:
            Brier Score值
        """
        if len(y_true) == 0 or len(y_prob) == 0:
            return 0.0

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")

        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def calculate_ace(y_true: np.ndarray, y_prob: np.ndarray, adaptive_bins: bool = True) -> float:
        """
        计算自适应校准误差 (Adaptive Calibration Error): 使用自适应分箱策略计算校准误差

        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测概率
            adaptive_bins: 是否使用自适应分箱

        Returns:
            ACE值
        """
        if len(y_true) == 0 or len(y_prob) == 0:
            return 0.0

        if not adaptive_bins:
            return CalibrationMetrics.calculate_ece(y_true, y_prob, CalibrationMetrics.DEFAULT_BINS)

        # 使用 netcal 的 ACE
        ace_metric = ACE()
        ace_value = ace_metric.measure(y_prob, y_true)
        return float(ace_value)

    @staticmethod
    def calculate_bic(y_true: np.ndarray, y_prob: np.ndarray, n_params: int) -> float:
        """
        计算贝叶斯信息准则 (Bayesian Information Criterion): 平衡模型复杂度和拟合效果

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            n_params: 模型参数数量

        Returns:
            BIC值
        """
        if len(y_true) == 0 or len(y_prob) == 0:
            return np.inf

        n_samples = len(y_true)

        # 计算对数似然
        y_prob_clipped = np.clip(y_prob, CalibrationMetrics.EPSILON, 1 - CalibrationMetrics.EPSILON)

        log_likelihood = np.sum(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))

        # BIC = -2 * log_likelihood + k * log(n)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        return float(bic)

    @staticmethod
    def calculate_model_weights_bma(metrics_list: List[Dict[str, float]], calibration_list: List[Dict[str, float]]) -> np.ndarray:
        """
        基于BMA计算模型权重
        作用：为多个模型分配权重，用于模型集成
            - 基于每个模型的BIC分数计算权重
            - 性能好的模型获得更高权重

        Args:
            metrics_list: 模型性能指标列表
            calibration_list: 模型校准指标列表

        Returns:
            标准化的模型权重
        """
        n_models = len(metrics_list)
        if n_models == 0:
            return np.array([])

        weights = np.zeros(n_models)

        for i, (metrics, calibration) in enumerate(zip(metrics_list, calibration_list)):
            try:
                # 计算综合性能分数
                f1_score = metrics.get("f1", 0.5)
                ece = calibration.get("ece", 0.1)

                # 估算参数数量（简化版本）
                n_params = 3
                n_samples = 1000  # 假设样本数量
                log_likelihood = f1_score * n_samples

                # 计算BIC分数（越小越好）
                bic = -2 * log_likelihood + n_params * np.log(n_samples)

                # 转换为权重（BIC越小权重越大）
                weights[i] = np.exp(-bic / 2)

            except Exception as e:
                logger.warning(f"计算模型 {i} 权重失败: {e}")
                weights[i] = 0.0

        # 标准化权重
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            weights = np.ones(n_models) / n_models

        return weights

    @staticmethod
    def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, Union[int, float]]:
        """计算混淆矩阵相关指标及基础分类指标"""
        # 如果y_pred是概率值，需要转换为预测标签
        if y_pred.dtype == float and np.all((y_pred >= 0) & (y_pred <= 1)):
            y_pred_binary = (y_pred >= threshold).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)

        # 使用sklearn计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred_binary)

        # 处理混淆矩阵的不同情况
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # 处理只有一个类别的情况
            unique_labels = np.unique(y_true)
            if len(unique_labels) == 1:
                if unique_labels[0] == 0:  # 只有real样本
                    tn = np.sum(y_pred_binary == 0)
                    fp = np.sum(y_pred_binary == 1)
                    fn = tp = 0
                else:  # 只有fake样本
                    fn = np.sum(y_pred_binary == 0)
                    tp = np.sum(y_pred_binary == 1)
                    tn = fp = 0
            else:
                tn = fp = fn = tp = 0

        # 使用sklearn库函数计算基础分类指标
        accuracy = accuracy_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0, average="macro")
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)

        # 计算其他衍生指标
        false_positive_rate = safe_division(fp, fp + tn)
        false_negative_rate = safe_division(fn, fn + tp)

        # 计算正负预测值
        positive_predictive_value = precision  # PPV == precision
        negative_predictive_value = safe_division(tn, tn + fn)

        return {
            # 基础分类指标
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            # 衍生指标
            "false_positive_rate": float(false_positive_rate),
            "false_negative_rate": float(false_negative_rate),
            "positive_predictive_value": float(positive_predictive_value),
            "negative_predictive_value": float(negative_predictive_value),
        }

    @staticmethod
    def calculate_class_specific_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_label: int = None,
        ece_bins: int = DEFAULT_BINS,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, float]:
        """计算特定类别的指标。如果未指定class_label，则对所有数据进行测试。"""
        if class_label is None:
            class_mask = np.ones_like(y_true, dtype=bool)
            adjusted_prob = y_prob
            pos_label = 1
        else:
            class_mask = y_true == class_label
            adjusted_prob = y_prob[class_mask]
            pos_label = class_label

        class_samples = np.sum(class_mask).item()
        class_y_true = y_true[class_mask]
        class_y_pred = (adjusted_prob > threshold).astype(int)  # 用adjusted_prob更清晰

        accuracy = accuracy_score(class_y_true, class_y_pred)
        f1 = f1_score(class_y_true, class_y_pred, pos_label=pos_label, zero_division=0)
        precision = precision_score(class_y_true, class_y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(class_y_true, class_y_pred, pos_label=pos_label, zero_division=0)

        ece = CalibrationMetrics.calculate_ece(class_y_true, adjusted_prob, n_bins=ece_bins)
        mce = CalibrationMetrics.calculate_mce(class_y_true, adjusted_prob, n_bins=ece_bins)
        ace = CalibrationMetrics.calculate_ace(class_y_true, adjusted_prob)
        brier_score = CalibrationMetrics.brier_score(class_y_true, adjusted_prob)
        log_loss = CalibrationMetrics.calculate_log_loss(class_y_true, adjusted_prob)

        return {
            "samples_num": round(class_samples, 4),
            "accuracy": CalibrationMetrics._format_metric(accuracy),
            "f1_score": CalibrationMetrics._format_metric(f1),
            "precision": CalibrationMetrics._format_metric(precision),
            "recall": CalibrationMetrics._format_metric(recall),
            "ece": CalibrationMetrics._format_metric(ece),
            "mce": CalibrationMetrics._format_metric(mce),
            "ace": CalibrationMetrics._format_metric(ace),
            "brier_score": CalibrationMetrics._format_metric(brier_score),
            "log_loss": CalibrationMetrics._format_metric(log_loss),
        }

    @staticmethod
    def calculate_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        计算对数损失 (Log Loss): 衡量预测概率与真实标签之间的差异
        标签只有 0, 1; 0表示真实样本，1表示伪造样本
        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测概率

        Returns:
            Log Loss值
        """
        if len(y_true) == 0 or len(y_prob) == 0:
            return None

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")

        if np.all(y_true == 0) or np.all(y_true == 1):
            return None

        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
        return float(sklearn_log_loss(y_true, y_prob))

    @staticmethod
    def calculate_ece_bootstrap_ci(
        y_true: np.ndarray,
        y_prob_original: np.ndarray,
        y_prob_calibrated: np.ndarray,
        n_bins: int = None,
        bootstrap_rounds: int = None,
        confidence_level: float = None,
    ) -> Tuple[float, float]:
        """
        非参数Bootstrap置信区间：对原始数据重采样，计算ECE改进的置信区间
        Args:
            y_true: 真实标签
            y_prob_original: 校准前预测概率
            y_prob_calibrated: 校准后预测概率
            n_bins: ECE分箱数量
            bootstrap_rounds: 重采样次数
            confidence_level: 置信度
        Returns:
            (ci_lower, ci_upper): ECE改进的置信区间
        """
        n = len(y_true)
        n_bins = n_bins or CalibrationMetrics.DEFAULT_BINS
        bootstrap_rounds = bootstrap_rounds or CalibrationMetrics.BOOTSTRAP_ROUNDS
        confidence_level = confidence_level or CalibrationMetrics.CONFIDENCE_LEVEL

        improvements = []
        for _ in range(bootstrap_rounds):
            idx = np.random.choice(n, n, replace=True)
            y_true_bs = y_true[idx]
            y_prob_ori_bs = y_prob_original[idx]
            y_prob_cal_bs = y_prob_calibrated[idx]
            ece_ori = CalibrationMetrics.calculate_ece(y_true_bs, y_prob_ori_bs, n_bins)
            ece_cal = CalibrationMetrics.calculate_ece(y_true_bs, y_prob_cal_bs, n_bins)
            improvements.append(ece_ori - ece_cal)
        alpha = 1 - confidence_level
        ci_lower = np.percentile(improvements, 100 * alpha / 2)
        ci_upper = np.percentile(improvements, 100 * (1 - alpha / 2))
        return (float(ci_lower), float(ci_upper))
