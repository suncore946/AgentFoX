import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import pickle


@dataclass
class PreprocessingConfig:
    """预处理配置"""

    # 标准化方法
    scaling_method: str = "standard"  # 'standard', 'minmax', 'robust', 'none'

    # 缺失值处理
    imputation_method: str = "median"  # 'mean', 'median', 'mode', 'knn', 'drop'
    knn_neighbors: int = 5

    # 异常值处理
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation', 'none'
    outlier_threshold: float = 3.0
    outlier_action: str = "clip"  # 'clip', 'remove', 'flag'

    # 特征选择
    remove_constant_features: bool = True
    remove_low_variance_features: bool = True
    variance_threshold: float = 0.01

    # 数据变换
    apply_log_transform: bool = True
    log_transform_features: Optional[List[str]] = None

    # 其他设置
    random_state: int = 42


@dataclass
class PreprocessingResult:
    """预处理结果"""

    processed_data: pd.DataFrame
    scaler: Optional[Any] = None
    imputer: Optional[Any] = None
    feature_names_in: List[str] = None
    feature_names_out: List[str] = None
    removed_features: List[str] = None
    preprocessing_stats: Dict[str, Any] = None

    def get_summary(self) -> str:
        """获取预处理摘要"""
        n_features_in = len(self.feature_names_in) if self.feature_names_in else 0
        n_features_out = len(self.feature_names_out) if self.feature_names_out else 0
        n_removed = len(self.removed_features) if self.removed_features else 0

        summary = f"""
预处理摘要:
- 输入特征数: {n_features_in}
- 输出特征数: {n_features_out}  
- 移除特征数: {n_removed}
- 处理后数据形状: {self.processed_data.shape}
        """.strip()

        if self.removed_features:
            summary += f"\n移除的特征: {', '.join(self.removed_features)}"

        return summary


class DataPreprocessor:
    """数据预处理器

    提供全面的数据预处理功能
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """初始化数据预处理器

        Args:
            config: 预处理配置
        """
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)

        # 预处理器组件
        self.scaler = None
        self.imputer = None
        self.is_fitted = False

        # 预处理统计信息
        self.preprocessing_stats = {}

    def fit_transform(self, data: pd.DataFrame, target_column: Optional[str] = None) -> PreprocessingResult:
        """拟合并转换数据

        Args:
            data: 输入数据
            target_column: 目标列（不参与预处理）

        Returns:
            PreprocessingResult: 预处理结果
        """
        if data.empty:
            raise ValueError("输入数据为空")

        self.logger.info(f"开始数据预处理: {data.shape}")

        # 分离特征和目标
        if target_column and target_column in data.columns:
            feature_data = data.drop(columns=[target_column])
            target_data = data[target_column]
        else:
            feature_data = data.copy()
            target_data = None

        original_feature_names = list(feature_data.columns)

        # 预处理步骤
        processed_data = feature_data.copy()
        removed_features = []

        # 1. 处理缺失值
        processed_data, imputer = self._handle_missing_values(processed_data)

        # 2. 移除常数特征和低方差特征
        if self.config.remove_constant_features or self.config.remove_low_variance_features:
            processed_data, removed_variance = self._remove_low_variance_features(processed_data)
            removed_features.extend(removed_variance)

        # 3. 异常值处理
        if self.config.outlier_method != "none":
            processed_data = self._handle_outliers(processed_data)

        # 4. 数据变换
        if self.config.apply_log_transform:
            processed_data = self._apply_log_transform(processed_data)

        # 5. 特征标准化
        if self.config.scaling_method != "none":
            processed_data, scaler = self._scale_features(processed_data)
        else:
            scaler = None

        # 重新添加目标列
        if target_data is not None:
            processed_data[target_column] = target_data

        # 保存预处理器状态
        self.scaler = scaler
        self.imputer = imputer
        self.is_fitted = True

        # 生成预处理统计
        stats = self._generate_preprocessing_stats(
            feature_data,
            processed_data,
            removed_features,
        )

        result = PreprocessingResult(
            processed_data=processed_data,
            scaler=scaler,
            imputer=imputer,
            feature_names_in=original_feature_names,
            feature_names_out=list(processed_data.columns),
            removed_features=removed_features,
            preprocessing_stats=stats,
        )

        self.logger.info(result.get_summary())
        return result

    def transform(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """仅转换数据（不拟合）

        Args:
            data: 输入数据
            target_column: 目标列

        Returns:
            pd.DataFrame: 转换后的数据
        """
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合，请先调用fit_transform")

        if data.empty:
            return data

        # 分离特征和目标
        if target_column and target_column in data.columns:
            feature_data = data.drop(columns=[target_column])
            target_data = data[target_column]
        else:
            feature_data = data.copy()
            target_data = None

        processed_data = feature_data.copy()

        # 应用已拟合的预处理器

        # 1. 缺失值处理
        if self.imputer:
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                processed_data[numeric_columns] = self.imputer.transform(processed_data[numeric_columns])

        # 2. 移除已标记的特征
        if hasattr(self, "_removed_features"):
            processed_data = processed_data.drop(columns=self._removed_features, errors="ignore")

        # 3. 数据变换（如果需要）
        if self.config.apply_log_transform:
            processed_data = self._apply_log_transform(processed_data)

        # 4. 特征标准化
        if self.scaler:
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                processed_data[numeric_columns] = self.scaler.transform(processed_data[numeric_columns])

        # 重新添加目标列
        if target_data is not None:
            processed_data[target_column] = target_data

        return processed_data

    def _handle_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Any]]:
        """处理缺失值"""
        if data.isnull().sum().sum() == 0:
            return data, None

        method = self.config.imputation_method
        processed_data = data.copy()
        imputer = None

        # 只对数值列进行插值
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) == 0:
            return processed_data, None

        if method == "drop":
            processed_data = processed_data.dropna()
            self.logger.info(f"删除包含缺失值的行，剩余 {len(processed_data)} 行")

        elif method in ["mean", "median"]:
            imputer = SimpleImputer(strategy=method)
            processed_data[numeric_columns] = imputer.fit_transform(processed_data[numeric_columns])
            self.logger.info(f"使用 {method} 填充缺失值")

        elif method == "knn":
            imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            processed_data[numeric_columns] = imputer.fit_transform(processed_data[numeric_columns])
            self.logger.info(f"使用 KNN(k={self.config.knn_neighbors}) 填充缺失值")

        # 处理非数值列的缺失值（使用众数）
        categorical_columns = processed_data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_columns:
            if processed_data[col].isnull().any():
                mode_value = processed_data[col].mode()
                if len(mode_value) > 0:
                    processed_data[col] = processed_data[col].fillna(mode_value[0])

        return processed_data, imputer

    def _remove_low_variance_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """移除常数和低方差特征"""
        removed_features = []
        processed_data = data.copy()

        for column in data.columns:
            if processed_data[column].dtype in [np.number]:
                # 检查常数特征
                if self.config.remove_constant_features:
                    if processed_data[column].nunique() <= 1:
                        removed_features.append(column)
                        continue

                # 检查低方差特征
                if self.config.remove_low_variance_features:
                    variance = processed_data[column].var()
                    if variance < self.config.variance_threshold:
                        removed_features.append(column)
                        continue

        if removed_features:
            processed_data = processed_data.drop(columns=removed_features)
            self.logger.info(f"移除 {len(removed_features)} 个低方差特征")

        # 保存移除的特征列表，供transform使用
        self._removed_features = removed_features

        return processed_data, removed_features

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        method = self.config.outlier_method
        action = self.config.outlier_action
        threshold = self.config.outlier_threshold

        processed_data = data.copy()
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns

        outliers_detected = 0

        for column in numeric_columns:
            values = processed_data[column].dropna()
            if len(values) < 10:  # 样本太少，跳过
                continue

            # 检测异常值
            if method == "zscore":
                z_scores = np.abs(stats.zscore(values))
                outlier_mask = z_scores > threshold

            elif method == "iqr":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (values < lower_bound) | (values > upper_bound)

            else:
                continue

            outliers_count = outlier_mask.sum()
            if outliers_count > 0:
                outliers_detected += outliers_count

                # 处理异常值
                if action == "clip":
                    # 截断异常值
                    if method == "iqr":
                        processed_data[column] = processed_data[column].clip(lower_bound, upper_bound)
                    elif method == "zscore":
                        mean_val = values.mean()
                        std_val = values.std()
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        processed_data[column] = processed_data[column].clip(lower_bound, upper_bound)

                elif action == "remove":
                    # 移除异常值行
                    outlier_indices = values[outlier_mask].index
                    processed_data = processed_data.drop(index=outlier_indices)

                elif action == "flag":
                    # 添加异常值标记列
                    flag_column = f"{column}_outlier_flag"
                    processed_data[flag_column] = False
                    processed_data.loc[outlier_mask, flag_column] = True

        if outliers_detected > 0:
            self.logger.info(f"处理了 {outliers_detected} 个异常值 (方法: {method}, 动作: {action})")

        return processed_data

    def _apply_log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用对数变换"""
        processed_data = data.copy()

        # 确定需要变换的特征
        if self.config.log_transform_features:
            transform_features = [f for f in self.config.log_transform_features if f in processed_data.columns]
        else:
            # 自动选择适合对数变换的特征（正偏态、正值）
            transform_features = []
            for column in processed_data.select_dtypes(include=[np.number]).columns:
                values = processed_data[column].dropna()
                if len(values) > 10 and values.min() > 0:  # 必须为正值
                    skewness = stats.skew(values)
                    if skewness > 1:  # 正偏态
                        transform_features.append(column)

        # 应用对数变换
        transformed_count = 0
        for column in transform_features:
            try:
                # 确保值为正
                min_val = processed_data[column].min()
                if min_val <= 0:
                    # 添加常数使所有值为正
                    shift = abs(min_val) + 1e-8
                    processed_data[column] = processed_data[column] + shift

                # 应用log(1+x)变换
                processed_data[column] = np.log1p(processed_data[column])
                transformed_count += 1

            except Exception as e:
                self.logger.warning(f"特征 {column} 对数变换失败: {e}")

        if transformed_count > 0:
            self.logger.info(f"对 {transformed_count} 个特征应用了对数变换")

        return processed_data

    def _scale_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """特征标准化"""
        method = self.config.scaling_method
        processed_data = data.copy()

        # 只对数值列进行标准化
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) == 0:
            return processed_data, None

        # 选择标准化方法
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return processed_data, None

        # 应用标准化
        processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])

        self.logger.info(f"使用 {method} 方法标准化 {len(numeric_columns)} 个数值特征")

        return processed_data, scaler

    def _generate_preprocessing_stats(
        self, original_data: pd.DataFrame, processed_data: pd.DataFrame, removed_features: List[str]
    ) -> Dict[str, Any]:
        """生成预处理统计信息"""
        stats = {
            "original_shape": original_data.shape,
            "processed_shape": processed_data.shape,
            "n_removed_features": len(removed_features),
            "removed_features": removed_features,
            "scaling_method": self.config.scaling_method,
            "imputation_method": self.config.imputation_method,
            "outlier_method": self.config.outlier_method,
        }

        # 计算缺失值变化
        original_missing = original_data.isnull().sum().sum()
        processed_missing = processed_data.select_dtypes(include=[np.number]).isnull().sum().sum()
        stats["missing_values_handled"] = original_missing - processed_missing

        # 计算数值特征统计
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats["numeric_features_stats"] = {
                "mean": processed_data[numeric_columns].mean().to_dict(),
                "std": processed_data[numeric_columns].std().to_dict(),
                "min": processed_data[numeric_columns].min().to_dict(),
                "max": processed_data[numeric_columns].max().to_dict(),
            }

        self.preprocessing_stats = stats
        return stats

    def save_preprocessor(self, file_path: str):
        """保存预处理器到文件

        Args:
            file_path: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合，无法保存")

        preprocessor_data = {
            "config": self.config,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "removed_features": getattr(self, "_removed_features", []),
            "preprocessing_stats": self.preprocessing_stats,
            "is_fitted": self.is_fitted,
        }

        with open(file_path, "wb") as f:
            pickle.dump(preprocessor_data, f)

        self.logger.info(f"预处理器已保存到: {file_path}")

    def load_preprocessor(self, file_path: str):
        """从文件加载预处理器

        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, "rb") as f:
                preprocessor_data = pickle.load(f)

            self.config = preprocessor_data["config"]
            self.scaler = preprocessor_data["scaler"]
            self.imputer = preprocessor_data["imputer"]
            self._removed_features = preprocessor_data.get("removed_features", [])
            self.preprocessing_stats = preprocessor_data.get("preprocessing_stats", {})
            self.is_fitted = preprocessor_data.get("is_fitted", False)

            self.logger.info(f"预处理器已从文件加载: {file_path}")

        except Exception as e:
            self.logger.error(f"加载预处理器失败: {e}")
            raise

    def get_feature_importance_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算特征重要性权重（基于方差和相关性）

        Args:
            data: 输入数据

        Returns:
            Dict[str, float]: 特征权重字典
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        weights = {}

        for column in numeric_columns:
            values = data[column].dropna()
            if len(values) < 10:
                weights[column] = 0.0
                continue

            # 基于方差的权重（归一化）
            variance_weight = values.var() / (values.var() + 1e-8)

            # 基于非零值比例的权重
            non_zero_ratio = (values != 0).sum() / len(values)

            # 综合权重
            weights[column] = variance_weight * non_zero_ratio

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights
