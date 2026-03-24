"""
PCA降维工具类
提供智能化的主成分分析降维功能
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class PCAConfig:
    """PCA配置类"""
    
    # 基本参数
    n_components: Union[int, float, str] = 0.95  # 主成分数量或方差比例
    max_components: Optional[int] = 100          # 最大主成分数量限制
    random_state: int = 42                       # 随机种子
    
    # 预处理参数
    standardize: bool = True                     # 是否标准化
    handle_missing: bool = True                  # 是否处理缺失值
    missing_strategy: str = 'mean'               # 缺失值处理策略
    
    # 策略参数
    low_dim_threshold: int = 20                  # 低维数据阈值
    medium_dim_threshold: int = 100              # 中等维度数据阈值
    use_kaiser_criterion: bool = True            # 是否使用Kaiser准则
    use_elbow_method: bool = True                # 是否使用肘部法则
    
    # 自适应参数
    adaptive_medium_variance: float = 0.95       # 中等维度目标方差
    adaptive_high_ratio: float = 0.1             # 高维数据保留比例
    adaptive_ultra_components: int = 100         # 超高维固定组件数
    
    # 输出控制
    verbose: bool = True                         # 是否输出详细信息
    return_transformer: bool = False             # 是否返回拟合的PCA对象


class PCADimensionReducer:
    """
    智能PCA降维器
    
    提供多种降维策略：
    1. 基于方差解释比例
    2. 固定主成分数量
    3. Kaiser准则（特征值>1）
    4. 肘部法则
    5. 自适应策略
    """
    
    def __init__(self, config: Optional[PCAConfig] = None, logger: Optional[logging.Logger] = None):
        """
        初始化PCA降维器
        
        Args:
            config: PCA配置对象
            logger: 日志记录器
        """
        self.config = config or PCAConfig()
        self.logger = logger or self._setup_logger()
        
        # 存储拟合后的对象
        self.pca_transformer: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.last_strategy_info: Dict[str, Any] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置默认日志记录器"""
        logger = logging.getLogger(f"{__name__}.PCAReducer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        return logger
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        拟合并转换数据
        
        Args:
            data: 输入数据 (样本×特征)
            
        Returns:
            Tuple[降维后的数据, 降维信息字典]
        """
        # 数据类型转换
        if isinstance(data, pd.DataFrame):
            original_data = data
            data_array = data.values
        else:
            original_data = pd.DataFrame(data)
            data_array = data
            
        return self._perform_pca(original_data, data_array)
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'PCADimensionReducer':
        """
        仅拟合不转换
        
        Args:
            data: 输入数据
            
        Returns:
            self
        """
        self.fit_transform(data)
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用已拟合的转换器转换新数据
        
        Args:
            data: 待转换数据
            
        Returns:
            转换后的数据
        """
        if self.pca_transformer is None:
            raise ValueError("必须先调用fit()或fit_transform()方法")
        
        # 数据预处理
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
            
        # 应用预处理步骤
        if self.imputer is not None:
            data_array = self.imputer.transform(data_array)
        if self.scaler is not None:
            data_array = self.scaler.transform(data_array)
            
        # PCA转换
        return self.pca_transformer.transform(data_array)
    
    def _perform_pca(self, original_data: pd.DataFrame, data_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """执行PCA降维的主方法"""
        try:
            # 1. 数据预处理
            processed_data = self._preprocess_data(data_array)
            n_samples, n_features = processed_data.shape
            
            # 2. 确定最大可用组件数
            max_components = min(n_samples - 1, n_features)
            
            # 3. 确定降维策略
            strategy_info = self._determine_strategy(n_features, max_components)
            self.last_strategy_info = strategy_info
            
            # 4. 执行降维
            return self._execute_strategy(processed_data, strategy_info, original_data.shape)
            
        except Exception as e:
            self.logger.error(f"PCA降维失败: {e}")
            return data_array, self._create_error_info(original_data.shape, str(e))
    
    def _preprocess_data(self, data_array: np.ndarray) -> np.ndarray:
        """数据预处理"""
        processed_data = data_array.copy()
        
        # 处理缺失值
        if self.config.handle_missing and np.any(np.isnan(processed_data)):
            self.logger.info(f"检测到缺失值，使用{self.config.missing_strategy}策略填充")
            self.imputer = SimpleImputer(strategy=self.config.missing_strategy)
            processed_data = self.imputer.fit_transform(processed_data)
        
        # 标准化处理
        if self.config.standardize:
            self.logger.debug("对数据进行标准化处理")
            self.scaler = StandardScaler()
            processed_data = self.scaler.fit_transform(processed_data)
        
        return processed_data
    
    def _determine_strategy(self, n_features: int, max_components: int) -> Dict[str, Any]:
        """确定降维策略"""
        
        # 应用最大组件数限制
        if self.config.max_components is not None:
            max_components = min(max_components, self.config.max_components)
        
        # 策略1: 低维数据保持原状
        if n_features <= self.config.low_dim_threshold:
            return {
                'strategy': 'keep_original',
                'n_components': n_features,
                'reason': f'数据维度({n_features})低于阈值({self.config.low_dim_threshold})'
            }
        
        # 策略2: 用户指定方差比例
        if isinstance(self.config.n_components, float) and 0 < self.config.n_components < 1:
            return {
                'strategy': 'variance_based',
                'target_variance': self.config.n_components,
                'max_components': max_components,
                'reason': f'基于方差解释比例({self.config.n_components})'
            }
        
        # 策略3: 用户指定固定组件数
        if isinstance(self.config.n_components, int) and self.config.n_components > 0:
            n_components = min(self.config.n_components, max_components)
            return {
                'strategy': 'fixed_components',
                'n_components': n_components,
                'reason': f'用户指定组件数({n_components})'
            }
        
        # 策略4: 自适应策略
        return self._get_adaptive_strategy(n_features, max_components)
    
    def _get_adaptive_strategy(self, n_features: int, max_components: int) -> Dict[str, Any]:
        """获取自适应策略"""
        
        if n_features <= self.config.medium_dim_threshold:
            # 中等维度：综合多种方法
            return {
                'strategy': 'adaptive_medium',
                'target_variance': self.config.adaptive_medium_variance,
                'use_kaiser': self.config.use_kaiser_criterion,
                'use_elbow': self.config.use_elbow_method,
                'max_components': max_components,
                'reason': f'中等维度({n_features})，综合Kaiser+方差+肘部法则'
            }
        elif n_features <= 1000:
            # 高维度：保守降维
            target_components = min(max_components, max(20, int(n_features * self.config.adaptive_high_ratio)))
            return {
                'strategy': 'adaptive_high',
                'n_components': target_components,
                'backup_variance': 0.90,
                'reason': f'高维数据({n_features})，保留约{self.config.adaptive_high_ratio*100}%维度'
            }
        else:
            # 超高维度：大幅降维
            target_components = min(max_components, self.config.adaptive_ultra_components)
            return {
                'strategy': 'adaptive_ultra_high',
                'n_components': target_components,
                'backup_variance': 0.85,
                'reason': f'超高维数据({n_features})，固定降至{target_components}维'
            }
    
    def _execute_strategy(self, data_array: np.ndarray, strategy_info: Dict[str, Any], 
                         original_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """执行具体的降维策略"""
        
        strategy = strategy_info['strategy']
        
        if strategy == 'keep_original':
            return self._handle_keep_original(data_array, original_shape, strategy_info)
        elif strategy == 'variance_based':
            return self._handle_variance_based(data_array, original_shape, strategy_info)
        elif strategy in ['fixed_components', 'adaptive_high', 'adaptive_ultra_high']:
            return self._handle_fixed_components(data_array, original_shape, strategy_info)
        elif strategy == 'adaptive_medium':
            return self._handle_adaptive_medium(data_array, original_shape, strategy_info)
        else:
            raise ValueError(f"未知策略: {strategy}")
    
    def _handle_keep_original(self, data_array: np.ndarray, original_shape: Tuple[int, int],
                             strategy_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理保持原始维度的情况"""
        n_components = strategy_info['n_components']
        
        pca_info = {
            "pca_explained_variance": 1.0,
            "pca_n_components_used": n_components,
            "pca_explained_variance_ratio": [1.0 / n_components] * n_components,
            "pca_explained_variance_values": [1.0] * n_components,
            "original_shape": original_shape,
            "pca_shape": data_array.shape,
            "strategy_used": strategy_info['strategy'],
            "strategy_reason": strategy_info['reason'],
            "variance_by_component": list(np.linspace(1.0/n_components, 1.0, n_components))
        }
        
        self.logger.info(f"保持原始维度: {original_shape}, 理由: {strategy_info['reason']}")
        return data_array, pca_info
    
    def _handle_variance_based(self, data_array: np.ndarray, original_shape: Tuple[int, int],
                              strategy_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理基于方差的降维"""
        target_variance = strategy_info['target_variance']
        max_components = strategy_info['max_components']
        
        # 首先拟合完整PCA获取方差信息
        pca_full = PCA(random_state=self.config.random_state)
        pca_full.fit(data_array)
        
        # 找到满足目标方差的组件数
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.argmax(cumsum_variance >= target_variance) + 1)
        n_components = min(n_components, max_components)
        
        # 使用确定的组件数重新拟合
        self.pca_transformer = PCA(n_components=n_components, random_state=self.config.random_state)
        pca_data = self.pca_transformer.fit_transform(data_array)
        
        return self._create_pca_result(pca_data, self.pca_transformer, original_shape,
                                      strategy_info['strategy'], strategy_info['reason'])
    
    def _handle_fixed_components(self, data_array: np.ndarray, original_shape: Tuple[int, int],
                                strategy_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理固定组件数的降维"""
        n_components = strategy_info['n_components']
        
        self.pca_transformer = PCA(n_components=n_components, random_state=self.config.random_state)
        pca_data = self.pca_transformer.fit_transform(data_array)
        
        return self._create_pca_result(pca_data, self.pca_transformer, original_shape,
                                      strategy_info['strategy'], strategy_info['reason'])
    
    def _handle_adaptive_medium(self, data_array: np.ndarray, original_shape: Tuple[int, int],
                               strategy_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理中等维度的自适应策略"""
        
        # 首先拟合完整PCA
        pca_full = PCA(random_state=self.config.random_state)
        pca_full.fit(data_array)
        
        candidates = []
        
        # Kaiser准则：保留特征值>1的组件
        if strategy_info.get('use_kaiser'):
            kaiser_components = np.sum(pca_full.explained_variance_ > 1.0)
            if kaiser_components > 0:
                candidates.append(('Kaiser', kaiser_components))
        
        # 方差阈值
        target_variance = strategy_info['target_variance']
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        variance_components = int(np.argmax(cumsum_variance >= target_variance) + 1)
        candidates.append(('Variance', variance_components))
        
        # 肘部法则
        if strategy_info.get('use_elbow'):
            elbow_components = self._find_elbow_point(pca_full.explained_variance_ratio_)
            if elbow_components is not None:
                candidates.append(('Elbow', elbow_components))
        
        # 综合决策
        component_values = [c[1] for c in candidates]
        n_components = min(max(component_values), strategy_info['max_components'])
        
        candidate_info = ', '.join([f"{name}={val}" for name, val in candidates])
        self.logger.info(f"自适应中等维度策略 - 候选方案: {candidate_info}, 最终选择: {n_components}")
        
        # 使用确定的组件数
        self.pca_transformer = PCA(n_components=n_components, random_state=self.config.random_state)
        pca_data = self.pca_transformer.fit_transform(data_array)
        
        return self._create_pca_result(pca_data, self.pca_transformer, original_shape,
                                      strategy_info['strategy'], 
                                      f"{strategy_info['reason']} (最优: {n_components})")
    
    def _find_elbow_point(self, explained_variance_ratio: np.ndarray, 
                         max_search: int = 50) -> Optional[int]:
        """使用肘部法则找到最优组件数"""
        try:
            search_range = min(len(explained_variance_ratio), max_search)
            if search_range < 3:
                return search_range
            
            variance_subset = explained_variance_ratio[:search_range]
            
            # 计算每个点到起终点连线的距离
            n_points = len(variance_subset)
            cumsum_var = np.cumsum(variance_subset)
            
            # 起点和终点
            start_point = np.array([0, 0])
            end_point = np.array([n_points - 1, cumsum_var[-1]])
            
            # 计算每个点到直线的距离
            distances = []
            for i in range(1, n_points - 1):
                point = np.array([i, cumsum_var[i]])
                
                # 点到直线距离公式
                distance = np.abs(np.cross(end_point - start_point, start_point - point)) / np.linalg.norm(end_point - start_point)
                distances.append(distance)
            
            if distances:
                elbow_idx = np.argmax(distances) + 1  # +1因为从索引1开始
                return min(elbow_idx + 1, search_range)  # +1转为组件数
            
            return search_range // 2
            
        except Exception as e:
            self.logger.warning(f"肘部法则计算失败: {e}")
            return None
    
    def _create_pca_result(self, pca_data: np.ndarray, pca: PCA, 
                          original_shape: Tuple[int, int], strategy: str, 
                          reason: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """创建PCA结果"""
        
        pca_info = {
            "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "pca_n_components_used": int(pca.n_components_),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_explained_variance_values": pca.explained_variance_.tolist(),
            "original_shape": original_shape,
            "pca_shape": pca_data.shape,
            "strategy_used": strategy,
            "strategy_reason": reason,
            "variance_by_component": np.cumsum(pca.explained_variance_ratio_).tolist()
        }
        
        # 可选返回拟合的转换器
        if self.config.return_transformer:
            pca_info["pca_transformer"] = pca
            if self.scaler is not None:
                pca_info["scaler"] = self.scaler
            if self.imputer is not None:
                pca_info["imputer"] = self.imputer
        
        self.logger.info(f"PCA完成 ({strategy}): {original_shape} -> {pca_data.shape}, "
                        f"解释方差: {pca_info['pca_explained_variance']:.3f}, "
                        f"理由: {reason}")
        
        return pca_data, pca_info
    
    def _create_error_info(self, original_shape: Tuple[int, int], error: str) -> Dict[str, Any]:
        """创建错误信息"""
        return {
            "pca_error": error,
            "pca_explained_variance": 1.0,
            "pca_n_components_used": original_shape[1],
            "original_shape": original_shape,
            "pca_shape": original_shape,
            "strategy_used": "error_fallback"
        }
    
    def get_component_importance(self, feature_names: Optional[list] = None) -> Optional[pd.DataFrame]:
        """
        获取主成分的特征重要性
        
        Args:
            feature_names: 原始特征名称列表
            
        Returns:
            主成分重要性DataFrame
        """
        if self.pca_transformer is None:
            self.logger.warning("尚未拟合PCA，无法获取主成分重要性")
            return None
        
        components = self.pca_transformer.components_
        n_components, n_features = components.shape
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        elif len(feature_names) != n_features:
            self.logger.warning(f"特征名称数量({len(feature_names)})与特征数量({n_features})不匹配")
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # 创建重要性矩阵
        importance_data = []
        for i in range(n_components):
            pc_name = f'PC_{i+1}'
            explained_var = self.pca_transformer.explained_variance_ratio_[i]
            
            for j, feature_name in enumerate(feature_names):
                importance_data.append({
                    'Principal_Component': pc_name,
                    'Feature': feature_name,
                    'Loading': components[i, j],
                    'Abs_Loading': abs(components[i, j]),
                    'Explained_Variance_Ratio': explained_var
                })
        
        importance_df = pd.DataFrame(importance_data)
        return importance_df
    
    def plot_explained_variance(self, max_components: int = 50) -> None:
        """
        绘制解释方差比例图
        
        Args:
            max_components: 最大显示的主成分数量
        """
        if self.pca_transformer is None:
            self.logger.warning("尚未拟合PCA，无法绘制解释方差图")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            explained_var_ratio = self.pca_transformer.explained_variance_ratio_
            cumsum_var_ratio = np.cumsum(explained_var_ratio)
            
            n_show = min(len(explained_var_ratio), max_components)
            components_range = range(1, n_show + 1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 个别方差解释比例
            ax1.bar(components_range, explained_var_ratio[:n_show])
            ax1.set_xlabel('主成分')
            ax1.set_ylabel('解释方差比例')
            ax1.set_title('各主成分解释方差比例')
            ax1.grid(True, alpha=0.3)
            
            # 累积方差解释比例
            ax2.plot(components_range, cumsum_var_ratio[:n_show], 'bo-')
            ax2.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
            ax2.axhline(y=0.90, color='orange', linestyle='--', label='90%阈值')
            ax2.set_xlabel('主成分数量')
            ax2.set_ylabel('累积解释方差比例')
            ax2.set_title('累积解释方差比例')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib未安装，无法绘制图表")


# 便捷函数入口
def pca_reduce_dimensions(data: Union[pd.DataFrame, np.ndarray], 
                         n_components: Union[int, float] = 0.95,
                         max_components: Optional[int] = 100,
                         standardize: bool = True,
                         random_state: int = 42,
                         verbose: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    便捷的PCA降维函数
    
    Args:
        data: 输入数据
        n_components: 主成分数量或方差比例
        max_components: 最大主成分数量
        standardize: 是否标准化
        random_state: 随机种子
        verbose: 是否输出详细信息
        
    Returns:
        降维后的数据和信息字典
    """
    config = PCAConfig(
        n_components=n_components,
        max_components=max_components,
        standardize=standardize,
        random_state=random_state,
        verbose=verbose
    )
    
    reducer = PCADimensionReducer(config)
    return reducer.fit_transform(data)