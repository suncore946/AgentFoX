import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

from ...utils.logger import get_logger
from ...configs.config_dataclass import ClusteringConfig


class ClusteringPreprocessor:
    """数据预处理器 - 专门处理聚类前的数据清洗和特征工程"""

    VARIANCE_THRESHOLD = 0.01
    CORRELATION_THRESHOLD = 0.9
    VIF_THRESHOLD = 10.0
    SKEWNESS_THRESHOLD = 2.0

    def __init__(self, config: ClusteringConfig):
        self.logger = get_logger(__name__)
        self.config = config
        self.scaler = RobustScaler()

    def preprocess_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[int, int]]:
        """集成所有预处理步骤的主函数"""
        preprocessing_info = {
            "original_shape": data.shape,
            "preprocessing_steps": [],
            "removed_features": [],
            "transformed_features": [],
            "vif_removed": [],
            "outlier_info": {},
        }
        index_mapping = {}  # 用于存储原始索引到预处理后索引的映射关系

        current_data = data.copy()

        # 如果数据中包含"XXX_result"列，则该列不作为后续预处理的对象
        result_cols = [col for col in current_data.columns if col.endswith("_result")]
        if result_cols:
            self.logger.info(f"检测到结果列，预处理时将排除: {result_cols}")
            current_data = current_data.drop(columns=result_cols)

        # 处理嵌套数组结构
        current_data, preprocessing_info = self._handle_nested_arrays(current_data, preprocessing_info)
        
        init_shape = current_data.shape

        # 步骤1: 异常值检测和处理
        current_data, preprocessing_info = self._detect_and_handle_outliers(current_data, preprocessing_info)

        # 步骤2: 移除高缺失率特征
        current_data, preprocessing_info = self._remove_high_missing_features(current_data, preprocessing_info)

        if current_data.empty:
            return current_data, index_mapping

        # 步骤3: 填充缺失值
        current_data, preprocessing_info = self._impute_missing_values(current_data, preprocessing_info)

        # 步骤4: 长尾分布变换
        current_data, preprocessing_info = self._apply_distribution_transforms(current_data, preprocessing_info)

        # # 步骤5: 移除低方差特征
        # current_data, preprocessing_info = self._remove_low_variance_features(current_data.copy(), preprocessing_info)

        if current_data.empty:
            return current_data, preprocessing_info, index_mapping

        # 步骤6: 标准化
        current_data, preprocessing_info = self._standardize_features(current_data, preprocessing_info)

        # 步骤7: 共线性控制
        if len(current_data.columns) > 1:
            current_data, preprocessing_info = self._remove_collinear_features(current_data, preprocessing_info)

        # 最终验证
        if not self._validate_final_data(current_data):
            self.logger.warning("最终数据验证失败")
            return pd.DataFrame(), preprocessing_info, index_mapping

        # # 记录索引映射关系(其实索引不会变, 这个代码是多余的)
        # index_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(current_data.index)}

        if result_cols:
            # 根据索引匹配来添加结果列，确保索引对齐
            result_data = data[result_cols].loc[current_data.index]
            current_data = pd.concat([current_data, result_data], axis=1)
            self.logger.info(f"重新添加结果列: {result_cols}")

        preprocessing_info.update({"final_shape": current_data.shape, "final_features": list(current_data.columns)})
        self.logger.info(
            f"预处理完成: 扩展形状:{init_shape}, 最终形状: {current_data.shape}, "
            f"移除特征数: {len(preprocessing_info['removed_features'])}"
            f"{', 变换特征数: ' + str(len(preprocessing_info['transformed_features'])) if preprocessing_info['transformed_features'] else ''}"
        )
        return current_data

    def _validate_final_data(self, current_data: pd.DataFrame) -> bool:
        """验证最终预处理后的数据质量"""
        if current_data.empty:
            return False

        if len(current_data.columns) < 1:
            return False

        if current_data.isnull().any().any():
            self.logger.warning("最终数据仍存在缺失值")
            return False

        if not np.isfinite(current_data.values).all():
            self.logger.warning("最终数据存在无穷值")
            return False

        return True

    def _detect_and_handle_outliers(
        self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """检测和处理异常值"""
        outlier_info = {"outliers_detected": 0, "outliers_capped": 0, "methods_used": []}

        for col in current_data.select_dtypes(include=[np.number]).columns:
            try:
                # IQR方法
                Q1 = current_data[col].quantile(0.25)
                Q3 = current_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (current_data[col] < lower_bound) | (current_data[col] > upper_bound)
                outlier_count = outliers.sum()

                # 修正代码以确保数据类型兼容
                if outlier_count > 0:
                    outlier_info["outliers_detected"] += outlier_count

                    # 使用winsorization代替简单截断
                    lower_bound = lower_bound.astype(current_data[col].dtype)  # 显式转换为列的数据类型
                    upper_bound = upper_bound.astype(current_data[col].dtype)  # 显式转换为列的数据类型
                    current_data.loc[current_data[col] < lower_bound, col] = lower_bound
                    current_data.loc[current_data[col] > upper_bound, col] = upper_bound
                    outlier_info["outliers_capped"] += outlier_count

                    if "IQR_winsorization" not in outlier_info["methods_used"]:
                        outlier_info["methods_used"].append("IQR_winsorization")

            except Exception as e:
                self.logger.debug(f"异常值处理失败 {col}: {e}")
                continue

        if outlier_info["outliers_detected"] > 0:
            preprocessing_info["preprocessing_steps"].append("outlier_handling")
            preprocessing_info["outlier_info"] = outlier_info
            self.logger.debug(f"处理异常值: {outlier_info['outliers_detected']} 个")

        return current_data, preprocessing_info

    def _handle_nested_arrays(self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """处理嵌套数组结构"""
        expanded_data = []  # 用于存储展开后的列
        for col in current_data.columns:
            # 检查列是否包含嵌套的 list 或 array
            if current_data[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                self.logger.info(f"检测到列 {col} 包含嵌套的 list 或 array，展开处理")
                try:
                    # 将嵌套的 list 或 array 展开为多列
                    expanded_cols = pd.DataFrame(current_data[col].tolist(), index=current_data.index)
                    expanded_cols.columns = [f"{col}_{i}" for i in range(expanded_cols.shape[1])]
                    expanded_data.append(expanded_cols)
                except Exception as e:
                    self.logger.warning(f"列 {col} 展开失败，原因: {e}")
            else:
                # 如果列不包含嵌套数据，直接保留
                expanded_data.append(current_data[[col]])

        # 合并所有处理后的列
        current_data = pd.concat(expanded_data, axis=1)

        # 数据类型转换和数值处理
        current_data = current_data.apply(pd.to_numeric, errors="coerce")  # 将所有列转换为数值类型
        current_data = current_data.fillna(0)  # 填充缺失值为 0
        current_data = np.clip(current_data, -1e6, 1e6)  # 限制数值范围

        preprocessing_info["preprocessing_steps"].append("handle_nested_arrays")
        self.logger.info(f"嵌套数组处理完成，当前数据形状: {current_data.shape}")
        return current_data, preprocessing_info

    def _remove_high_missing_features(
        self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """移除高缺失率特征"""
        missing_rates = current_data.isnull().mean()
        valid_features = missing_rates[missing_rates <= self.config.missing_threshold].index
        removed_missing = set(current_data.columns) - set(valid_features)

        if removed_missing:
            current_data = current_data[valid_features]
            preprocessing_info["removed_features"].extend(list(removed_missing))
            preprocessing_info["preprocessing_steps"].append("remove_high_missing")
            self.logger.debug(f"移除高缺失率特征: {len(removed_missing)} 个")

        return current_data, preprocessing_info

    def _impute_missing_values(self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """填充缺失值"""
        if current_data.isnull().any().any():
            imputer = SimpleImputer(strategy="median")
            imputed_data = imputer.fit_transform(current_data)
            current_data = pd.DataFrame(imputed_data, columns=current_data.columns, index=current_data.index)
            preprocessing_info["preprocessing_steps"].append("impute_missing")

        return current_data, preprocessing_info

    def _apply_distribution_transforms(
        self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """对长尾分布特征应用变换"""
        transformed_data = current_data.copy()
        transformed_features = []

        for col in current_data.columns:
            try:
                col_data = current_data[col].dropna()
                if len(col_data) < 10:  # 数据量太少跳过
                    continue

                # 计算偏度
                skewness = abs(stats.skew(col_data))

                if skewness > self.SKEWNESS_THRESHOLD:
                    # 对于正值数据，尝试log1p变换
                    if (col_data >= 0).all():
                        transformed_col = np.log1p(col_data)
                        new_skewness = abs(stats.skew(transformed_col))

                        if new_skewness < skewness:
                            transformed_data[col] = np.log1p(current_data[col].fillna(0))
                            transformed_features.append(f"{col}_log1p")
                            self.logger.debug(f"对 {col} 应用log1p变换，偏度从 {skewness:.3f} 降至 {new_skewness:.3f}")
                            continue

                    # 尝试Yeo-Johnson变换
                    try:
                        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                        transformed_col = transformer.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                        new_skewness = abs(stats.skew(transformed_col))

                        if new_skewness < skewness:
                            full_transformed = transformer.transform(
                                current_data[col].fillna(current_data[col].median()).values.reshape(-1, 1)
                            ).flatten()
                            transformed_data[col] = full_transformed
                            transformed_features.append(f"{col}_yeo_johnson")
                            self.logger.debug(f"对 {col} 应用Yeo-Johnson变换，偏度从 {skewness:.3f} 降至 {new_skewness:.3f}")
                    except Exception as e:
                        self.logger.debug(f"Yeo-Johnson变换失败 {col}: {e}")

            except Exception as e:
                self.logger.debug(f"特征 {col} 变换失败: {e}")
                continue

        if transformed_features:
            preprocessing_info["transformed_features"].extend(transformed_features)
            preprocessing_info["preprocessing_steps"].append("distribution_transform")

        return transformed_data, preprocessing_info

    def _remove_low_variance_features(
        self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """移除低方差特征"""
        variance_filter = VarianceThreshold(threshold=self.VARIANCE_THRESHOLD)
        try:
            variance_mask = variance_filter.fit(current_data).get_support()
            low_variance_features = current_data.columns[~variance_mask]

            if len(low_variance_features) > 0:
                selected_columns = current_data.columns[variance_mask]
                current_data = current_data[selected_columns]
                preprocessing_info["removed_features"].extend(list(low_variance_features))
                preprocessing_info["preprocessing_steps"].append("remove_low_variance")
                self.logger.debug(f"移除低方差特征: {len(low_variance_features)} 个")
        except Exception as e:
            self.logger.debug(f"方差筛选失败: {e}")

        return current_data, preprocessing_info

    def _standardize_features(self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """标准化特征"""
        # current_data中每一行如果是一个list， 则只对这一行list本身进行标准化
        scaled_data = self.scaler.fit_transform(current_data)
        current_data = pd.DataFrame(scaled_data, columns=current_data.columns, index=current_data.index)
        preprocessing_info["preprocessing_steps"].append("standardization")
        return current_data, preprocessing_info

    def _remove_collinear_features(
        self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """移除共线性特征 - 结合相关性和VIF"""
        collinearity_info = {"removed_corr": [], "removed_vif": []}

        # 步骤1: 移除高相关性特征
        if len(current_data.columns) > 1:
            try:
                corr_matrix = current_data.corr().abs()
                upper_triangle = np.triu(corr_matrix.values, k=1)
                high_corr_pairs = np.where(upper_triangle > self.CORRELATION_THRESHOLD)

                features_to_remove = set()
                if len(high_corr_pairs[0]) > 0:
                    feature_variances = current_data.var().sort_values(ascending=False)
                    feature_names = current_data.columns

                    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                        feat1, feat2 = feature_names[i], feature_names[j]
                        if feat1 not in features_to_remove and feat2 not in features_to_remove:
                            # 移除方差较小的特征
                            if feature_variances[feat1] < feature_variances[feat2]:
                                features_to_remove.add(feat1)
                            else:
                                features_to_remove.add(feat2)

                if features_to_remove:
                    current_data = current_data.drop(columns=list(features_to_remove))
                    collinearity_info["removed_corr"] = list(features_to_remove)
                    self.logger.debug(f"移除高相关性特征: {len(features_to_remove)} 个")

            except Exception as e:
                self.logger.debug(f"相关性分析失败: {e}")

        # 步骤2: VIF检查
        if 2 <= len(current_data.columns) <= 50:
            try:
                vif_removed = self._remove_high_vif_features(current_data)
                if vif_removed:
                    current_data = current_data.drop(columns=vif_removed)
                    collinearity_info["removed_vif"] = vif_removed
                    self.logger.debug(f"移除高VIF特征: {len(vif_removed)} 个")

            except Exception as e:
                self.logger.debug(f"VIF计算失败: {e}")

        if collinearity_info["removed_corr"] or collinearity_info["removed_vif"]:
            preprocessing_info["removed_features"].extend(collinearity_info["removed_corr"])
            preprocessing_info["vif_removed"].extend(collinearity_info["removed_vif"])
            preprocessing_info["preprocessing_steps"].append("remove_collinear")

        return current_data, preprocessing_info

    def _remove_high_vif_features(self, data: pd.DataFrame) -> List[str]:
        """移除高VIF特征"""
        removed_features = []
        current_data = data.copy()

        max_iterations = min(len(current_data.columns), 20)

        for iteration in range(max_iterations):
            if len(current_data.columns) < 2:
                break

            try:
                vif_data = pd.DataFrame()
                vif_data["Feature"] = current_data.columns
                vif_data["VIF"] = [variance_inflation_factor(current_data.values, i) for i in range(len(current_data.columns))]

                vif_data = vif_data[np.isfinite(vif_data["VIF"])]

                if vif_data.empty:
                    break

                max_vif = vif_data["VIF"].max()

                if max_vif > self.VIF_THRESHOLD:
                    feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
                    current_data = current_data.drop(columns=[feature_to_remove])
                    removed_features.append(feature_to_remove)
                    self.logger.debug(f"移除高VIF特征 {feature_to_remove} (VIF: {max_vif:.2f})")
                else:
                    break

            except Exception as e:
                self.logger.debug(f"VIF计算迭代 {iteration} 失败: {e}")
                break

        return removed_features

    def _final_feature_selection(
        self, current_data: pd.DataFrame, preprocessing_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """最终特征数量控制（备用方法）"""
        if len(current_data.columns) > self.config.max_features:
            feature_variances = current_data.var().sort_values(ascending=False)
            selected_features = feature_variances.head(self.config.max_features).index
            variance_removed = set(current_data.columns) - set(selected_features)

            if variance_removed:
                current_data = current_data[selected_features]
                preprocessing_info["removed_features"].extend(list(variance_removed))
                preprocessing_info["preprocessing_steps"].append("final_feature_selection")
                self.logger.debug(f"最终特征选择移除: {len(variance_removed)} 个特征")

        return current_data, preprocessing_info
