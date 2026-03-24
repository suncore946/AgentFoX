from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class FeatureConfig:
    """图像特征提取配置"""

    batch_size: int = 512
    features: List[str] = None
    num_workers: int = 8
    use_gpu: bool = True

    def __post_init__(self):
        if self.features is None:
            self.features = [
                "LaplacianVar",  # 清晰度/锐度
                "JPEG_Q",  # 压缩质量估计
                "HF_LF_Ratio",  # 高低频比率
                "EdgeDensity",  # 边缘密度
                "Colorfulness",  # 颜色丰富度
                "MeanIntensity",  # 平均亮度
                "Contrast_P95_P5",  # 对比度(P95-P5)
            ]


@dataclass
class ClusteringConfig:
    """聚类配置"""

    # 聚类参数
    min_difference_threshold: float = 0.05
    significant_difference_threshold: float = 0.10

    # HDBSCAN核心参数
    min_cluster_size_ratio: float = 0.005  # 0.5% * N_IID
    min_cluster_size_abs: int = 10
    min_samples: int = 15
    cluster_selection_epsilon: float = 0.1
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"  # 改为eom方法

    # 特征预处理参数
    max_features: int = 20
    missing_threshold: float = 0.5  # 缺失率超过50%的特征将被移除

    # PCA降维参数
    n_components_pca: float = None  # 保持95%方差
    pca_n_components_max: int = 64  # PCA降维的最大维度

    # 质量评分权重
    scoring_weights: Dict[str, float] = None

    # 评分标准
    scoring_thresholds: Dict[str, float] = None

    # 聚类结果保存路径
    save_dir: str = "outputs/cluster_models"
    
    # 是否强制重新聚类
    force_clustering: bool = False

    def __post_init__(self):
        if self.scoring_weights is None:
            self.scoring_weights = {
                "cluster_count": 0.3,  # 簇数量权重
                "silhouette": 0.3,  # 轮廓系数权重
                "coverage": 0.2,  # 覆盖率权重
                "noise_ratio": 0.2,  # 噪声比例权重
            }

        if self.scoring_thresholds is None:
            self.scoring_thresholds = {
                "max_clusters_for_full_score": 10.0,  # 10个簇得满分
                "silhouette_full_score": 0.5,  # 轮廓系数0.5得满分
                "coverage_full_score": 0.8,  # 80%覆盖率得满分
                "max_noise_ratio": 0.5,  # 50%噪声得0分
            }


@dataclass
class DifficultyConfig:
    """难度度量配置"""

    # 数值稳定性参数
    eps: float = 1e-6  # 概率裁剪阈值

    # 基础难度计算参数
    use_calibrated_probs: bool = True  # 是否使用校准概率
    calibrated_prob_column: str = "pred_calibrated"  # 校准概率列名
    raw_prob_column: str = "pred_prob"  # 原始概率列名

    # 融合参数
    epistemic_weight: float = 0.3  # 认知不确定性权重 λ1

    # 标准化参数
    standardization_method: str = "quantile"  # 标准化方法: 'quantile', 'z_score', 'min_max'
    quantile_lower: float = 0.01  # 下分位数
    quantile_upper: float = 0.99  # 上分位数

    # 噪声检测阈值
    label_noise_confidence_threshold: float = 0.9  # 标签噪声检测的置信度阈值
    label_noise_disagree_threshold: float = 0.2  # 标签噪声检测的分歧度阈值
    ood_epistemic_threshold: float = 0.9  # OOD检测的认知不确定性阈值
    ood_var_threshold: float = 0.9  # OOD检测的方差阈值
    ambiguous_aleatoric_threshold: float = 0.7  # 含糊样本的数据不确定性阈值
    ambiguous_prob_threshold: float = 0.1  # 含糊样本的概率阈值(距离0.5的距离)

    # IRT模型参数
    irt_max_iter: int = 1000  # IRT最大迭代次数
    irt_tolerance: float = 1e-6  # IRT收敛容差
    irt_min_variance_threshold: float = 1e-6  # 过滤极端样本的方差阈值

    # 分位数统计参数
    percentiles: List[int] = None

    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = [10, 25, 50, 75, 90, 95, 99]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "eps": self.eps,
            "use_calibrated_probs": self.use_calibrated_probs,
            "calibrated_prob_column": self.calibrated_prob_column,
            "raw_prob_column": self.raw_prob_column,
            "epistemic_weight": self.epistemic_weight,
            "standardization_method": self.standardization_method,
            "quantile_lower": self.quantile_lower,
            "quantile_upper": self.quantile_upper,
            "label_noise_confidence_threshold": self.label_noise_confidence_threshold,
            "label_noise_disagree_threshold": self.label_noise_disagree_threshold,
            "ood_epistemic_threshold": self.ood_epistemic_threshold,
            "ood_var_threshold": self.ood_var_threshold,
            "ambiguous_aleatoric_threshold": self.ambiguous_aleatoric_threshold,
            "ambiguous_prob_threshold": self.ambiguous_prob_threshold,
            "irt_max_iter": self.irt_max_iter,
            "irt_tolerance": self.irt_tolerance,
            "irt_min_variance_threshold": self.irt_min_variance_threshold,
            "percentiles": self.percentiles,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DifficultyConfig":
        """从字典创建配置对象"""
        return cls(**config_dict)


@dataclass
class ProfileConfig:
    """模型画像配置"""

    llm: Dict[str, Any]
    model_profile: Path = Path("./forensic_agent/configs/profiles/model_profiles.json")  # 模型画像文件路径
    output_dir: Path = Path("./outputs/model_profiles")  # 输出目录
    performance_threshold: float = 0.05  # 性能差异阈值
    significance_threshold: float = 0.10  # 显著差异阈值
    min_samples_for_analysis: int = 50
    max_workers: int = 3  # 最大线程数


@dataclass
class CalibrationConfig:
    """
    校准相关配置
    """

    # 数据路径配置
    save_dir: Union[str, Path] = "outputs/calibration"
    # 数据划分配置
    random_state: int = 42
    # 校准配置
    calibration_methods: Optional[List[str]] = None
    # ECE计算配置
    ece_bins: int = 10
    ece_threshold: float = 0.05
    ace_adaptive: bool = True
    # BMA配置
    enable_bma: bool = True
    bma_method: str = "bic"

    # 大模型配置
    llm: Dict[str, Any] = None
    calibration_analysis_prompt_path: str = "./forensic_agent/configs/prompts/calibration_analysis_prompt.txt"
    model_profile_prompt_path: str = "./forensic_agent/configs/prompts/model_profile_prompt.txt"

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.calibration_methods is None:
            self.calibration_methods = [
                "temperature_scaling",
                "platt_scaling",
                "isotonic_regression",
                "histogram_binning",
                "beta_calibration",
            ]
