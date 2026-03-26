"""
工具函数模块
提供难度度量系统中常用的数学计算函数
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import warnings


def clip_probabilities(probs: Union[np.ndarray, pd.Series], 
                      eps: float = 1e-6) -> Union[np.ndarray, pd.Series]:
    """
    将概率裁剪到[eps, 1-eps]区间，确保数值稳定性
    
    Args:
        probs: 预测概率
        eps: 裁剪阈值，建议1e-6到1e-4之间
        
    Returns:
        裁剪后的概率
    """
    return np.clip(probs, eps, 1 - eps)


def calculate_nll(y_true: Union[np.ndarray, pd.Series], 
                  y_pred: Union[np.ndarray, pd.Series],
                  eps: float = 1e-6) -> Union[np.ndarray, pd.Series]:
    """
    计算二分类负对数似然损失 (NLL)
    
    NLL = -[y * log(p) + (1-y) * log(1-p)]
    
    Args:
        y_true: 真实标签 (0 或 1)
        y_pred: 预测概率 [0, 1]
        eps: 数值稳定性参数
        
    Returns:
        NLL损失值
    """
    # 确保概率在有效范围内
    y_pred_clipped = clip_probabilities(y_pred, eps)
    
    # 计算NLL
    nll = -(y_true * np.log(y_pred_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_clipped))
    
    return nll


def calculate_entropy(probs: Union[np.ndarray, pd.Series],
                     eps: float = 1e-6) -> Union[np.ndarray, pd.Series]:
    """
    计算二分类熵
    
    H(p) = -p * log(p) - (1-p) * log(1-p)
    
    Args:
        probs: 预测概率 [0, 1]
        eps: 数值稳定性参数
        
    Returns:
        熵值
    """
    # 确保概率在有效范围内
    p_clipped = clip_probabilities(probs, eps)
    
    # 计算熵
    entropy = -(p_clipped * np.log(p_clipped) + 
                (1 - p_clipped) * np.log(1 - p_clipped))
    
    return entropy


def calculate_disagreement_rate(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    计算模型间分歧度（不一致率）
    
    Args:
        predictions: 模型预测概率矩阵，形状为 (n_samples, n_models)
        threshold: 二值化阈值
        
    Returns:
        每个样本的分歧度 [0, 1]
    """
    # 转换为二值预测
    binary_preds = (predictions > threshold).astype(int)
    
    # 计算每个样本的分歧度（不一致的模型数量 / 总模型数量）
    n_models = predictions.shape[1]
    disagreement = np.abs(binary_preds - binary_preds.mean(axis=1, keepdims=True))
    disagreement_rate = disagreement.sum(axis=1) / n_models
    
    return disagreement_rate


def robust_standardize(values: Union[np.ndarray, pd.Series],
                      method: str = 'quantile',
                      lower_q: float = 0.01,
                      upper_q: float = 0.99) -> Union[np.ndarray, pd.Series]:
    """
    稳健标准化：分位裁剪后进行标准化
    
    Args:
        values: 待标准化的值
        method: 标准化方法 ('quantile', 'z_score', 'min_max')
        lower_q: 下分位数
        upper_q: 上分位数
        
    Returns:
        标准化后的值
    """
    if method == 'quantile':
        # 分位裁剪后min-max归一化
        lower_bound = np.quantile(values, lower_q)
        upper_bound = np.quantile(values, upper_q)
        
        # 裁剪
        clipped_values = np.clip(values, lower_bound, upper_bound)
        
        # Min-max归一化到[0, 1]
        if upper_bound > lower_bound:
            normalized = (clipped_values - lower_bound) / (upper_bound - lower_bound)
        else:
            normalized = np.zeros_like(clipped_values)
            
    elif method == 'z_score':
        # 分位裁剪后z-score标准化
        lower_bound = np.quantile(values, lower_q)
        upper_bound = np.quantile(values, upper_q)
        clipped_values = np.clip(values, lower_bound, upper_bound)
        
        mean_val = np.mean(clipped_values)
        std_val = np.std(clipped_values)
        
        if std_val > 0:
            normalized = (clipped_values - mean_val) / std_val
        else:
            normalized = np.zeros_like(clipped_values)
            
    elif method == 'min_max':
        # 简单min-max归一化
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(values)
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    return normalized


def calculate_percentiles(values: Union[np.ndarray, pd.Series],
                         percentiles: list = [10, 25, 50, 75, 90, 95, 99]) -> dict:
    """
    计算分位数统计
    
    Args:
        values: 数值
        percentiles: 要计算的分位数列表
        
    Returns:
        分位数字典
    """
    result = {}
    for p in percentiles:
        result[f'p{p}'] = np.percentile(values, p)
    
    return result


def validate_prediction_matrix(predictions: np.ndarray, 
                              labels: np.ndarray) -> Tuple[bool, str]:
    """
    验证预测矩阵的有效性
    
    Args:
        predictions: 预测概率矩阵 (n_samples, n_models)
        labels: 真实标签 (n_samples,)
        
    Returns:
        (是否有效, 错误信息)
    """
    # 检查形状
    if len(predictions.shape) != 2:
        return False, "预测矩阵必须是二维的"
    
    n_samples, n_models = predictions.shape
    
    if len(labels) != n_samples:
        return False, f"标签数量({len(labels)})与样本数量({n_samples})不匹配"
    
    # 检查概率范围
    if np.any((predictions < 0) | (predictions > 1)):
        return False, "预测概率必须在[0, 1]范围内"
    
    # 检查标签
    unique_labels = np.unique(labels)
    if not all(label in [0, 1] for label in unique_labels):
        return False, "标签必须是0或1"
    
    # 检查模型数量
    if n_models < 2:
        return False, "至少需要2个模型的预测结果"
    
    # 检查是否有NaN值
    if np.any(np.isnan(predictions)) or np.any(np.isnan(labels)):
        return False, "数据中包含NaN值"
    
    return True, "验证通过"


def create_version_tag(params: dict) -> str:
    """
    创建版本标记，记录所有关键参数
    
    Args:
        params: 参数字典
        
    Returns:
        版本标记字符串
    """
    param_strs = []
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            param_strs.append(f"{key}={value:.4f}")
        else:
            param_strs.append(f"{key}={value}")
    
    return "_".join(param_strs)


def filter_extreme_samples(predictions: np.ndarray, 
                          labels: np.ndarray,
                          min_variance_threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    过滤极端样本（全对或全错的样本）
    
    Args:
        predictions: 预测概率矩阵 (n_samples, n_models)
        labels: 真实标签
        min_variance_threshold: 最小方差阈值
        
    Returns:
        (过滤后的预测, 过滤后的标签, 保留的索引)
    """
    # 转换为二值预测
    binary_preds = (predictions > 0.5).astype(int)
    
    # 计算每个样本在所有模型上的方差
    variances = np.var(binary_preds, axis=1)
    
    # 保留有足够方差的样本
    valid_mask = variances >= min_variance_threshold
    
    if np.sum(valid_mask) == 0:
        warnings.warn("所有样本都被过滤掉了，将返回原始数据")
        return predictions, labels, np.arange(len(predictions))
    
    return predictions[valid_mask], labels[valid_mask], np.where(valid_mask)[0]


def safe_divide(numerator: Union[np.ndarray, float], 
                denominator: Union[np.ndarray, float],
                default_value: float = 0.0) -> Union[np.ndarray, float]:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default_value: 当分母为0时的默认值
        
    Returns:
        除法结果
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.divide(numerator, denominator)
        
        # 处理除零和无穷大
        if np.isscalar(result):
            if not np.isfinite(result):
                result = default_value
        else:
            result = np.where(np.isfinite(result), result, default_value)
    
    return result


def compute_correlation_metrics(difficulty1: np.ndarray, 
                               difficulty2: np.ndarray) -> dict:
    """
    计算两个难度度量之间的相关性指标
    
    Args:
        difficulty1: 第一个难度度量
        difficulty2: 第二个难度度量
        
    Returns:
        相关性指标字典
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    metrics = {}
    
    try:
        # Pearson相关系数
        r_pearson, p_pearson = pearsonr(difficulty1, difficulty2)
        metrics['pearson_r'] = r_pearson
        metrics['pearson_p'] = p_pearson
        
        # Spearman相关系数
        r_spearman, p_spearman = spearmanr(difficulty1, difficulty2)
        metrics['spearman_r'] = r_spearman
        metrics['spearman_p'] = p_spearman
        
        # Kendall's tau
        tau, p_tau = kendalltau(difficulty1, difficulty2)
        metrics['kendall_tau'] = tau
        metrics['kendall_p'] = p_tau
        
    except Exception as e:
        warnings.warn(f"计算相关性时出错: {e}")
        metrics = {
            'pearson_r': np.nan, 'pearson_p': np.nan,
            'spearman_r': np.nan, 'spearman_p': np.nan,
            'kendall_tau': np.nan, 'kendall_p': np.nan
        }
    
    return metrics