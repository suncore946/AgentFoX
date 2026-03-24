"""
基础校准方法实现
包含各种校准算法的核心实现
"""

from enum import Enum
import numpy as np
from typing import Dict, Tuple, Any, Optional
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import beta
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score

from .calibration_metrics import CalibrationMetrics as Metrics
from .calibration_exceptions import CalibrationOptimizationError
from ..configs.config_dataclass import CalibrationConfig


class CalibrationMethod(Enum):
    """校准方法枚举"""

    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    HISTOGRAM_BINNING = "histogram_binning"
    BETA_CALIBRATION = "beta_calibration"
    NONE = "none"


class CalibrationMethods:
    """
    基础校准方法类 - 概率校准核心算法实现

    概率校准的目标是使模型输出的概率更接近真实的置信度。
    一个校准良好的模型应该满足：预测概率为p时，真实标签为正类的比例也应该接近p。

    本类实现了5种经典的二分类概率校准方法：
    1. Temperature Scaling (温度缩放) - 参数化方法，适用于深度学习模型
    2. Platt Scaling (Platt缩放) - 参数化方法，使用sigmoid函数
    3. Isotonic Regression (等距回归) - 非参数化方法，学习单调映射
    4. Histogram Binning (直方图分箱) - 非参数化方法，分段常数函数
    5. Beta Calibration (Beta校准) - 参数化方法，基于Beta分布假设

    算法特点：
    - Temperature Scaling: 保持相对概率顺序，全局缩放
    - Platt Scaling: 灵活的sigmoid映射，适合小数据集
    - Isotonic Regression: 无参数假设，适合复杂概率分布
    - Histogram Binning: 简单直观，但可能过拟合
    - Beta Calibration: 基于分布假设，计算效率高

    使用场景：
    - 训练阶段：使用验证集学习校准参数
    - 推理阶段：将学习到的参数应用到新预测值
    """

    # 统一的数值稳定性常量，防止log(0)和除零错误
    EPSILON = 1e-15

    def __init__(self, config: CalibrationConfig):
        """
        初始化校准方法类

        Args:
            config: 校准配置对象，包含ECE分箱数等超参数
        """
        self.config = config

    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        """
        将概率转换为logits（sigmoid函数的反函数）

        数学公式：logit(p) = log(p / (1-p))
        这是sigmoid函数的反函数，用于将[0,1]区间的概率值映射到(-∞,+∞)区间

        数值稳定性处理：
        - 使用epsilon裁剪避免log(0)和log(∞)
        - 确保输入概率在(ε, 1-ε)范围内

        Args:
            probs: 概率值数组，期望范围[0,1]

        Returns:
            logits数组，范围(-∞,+∞)

        Note:
            此函数主要用于Platt Scaling中将概率转换为logits供逻辑回归使用
        """
        # 数值稳定性处理：避免极端值导致的数值问题
        probs_clipped = np.clip(probs, self.EPSILON, 1 - self.EPSILON)
        # logit变换：log(p/(1-p))
        return np.log(probs_clipped / (1 - probs_clipped))

    def _validate_inputs(self, predictions: np.ndarray, labels: np.ndarray, expected_type: str = "probs") -> Tuple[np.ndarray, np.ndarray]:
        """
        验证和预处理输入数据的核心方法

        这是所有校准方法的入口检查点，确保输入数据的格式正确性、
        类型一致性和数值有效性。通过统一的验证流程，提高代码健壮性。

        验证步骤：
        1. 数据类型转换：将输入转换为numpy数组并扁平化
        2. 长度一致性检查：确保预测值和标签数量匹配
        3. 标签格式验证：确保是有效的二分类标签
        4. 预测值类型检查：根据expected_type验证数值范围
        5. 异常值处理：检测和处理NaN、Inf等无效值

        标签处理逻辑：
        - 标准格式：{0, 1} 或 {False, True}
        - 非标准格式：自动转换（如{-1, 1} → {0, 1}）
        - 多类标签：报错，仅支持二分类

        预测值处理：
        - "probs"类型：期望[0,1]范围，超出范围将裁剪并警告
        - "logits"类型：允许任意实数，检查NaN/Inf

        Args:
            predictions: 模型预测值数组，可以是概率或logits
            labels: 真实标签数组，期望为二分类标签
            expected_type: 期望的预测值类型 ("probs" 或 "logits")

        Returns:
            tuple: (处理后的预测值, 处理后的标签)

        Raises:
            ValueError: 当输入数据不满足要求时
        """
        # Step 1: 数据类型标准化
        predictions = np.asarray(predictions).flatten()
        labels = np.asarray(labels).flatten()

        # Step 2: 长度一致性检查
        if len(predictions) != len(labels):
            raise ValueError(f"预测值和标签长度不匹配: {len(predictions)} vs {len(labels)}")

        # Step 3: 标签格式验证和标准化
        unique_labels = np.unique(labels)
        if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [False, True]):
            # 如果标签不是标准的0/1，尝试转换
            if len(unique_labels) == 2:
                logger.warning(f"标签值 {unique_labels} 不是标准的0/1，尝试自动转换")
                labels = (labels == unique_labels[1]).astype(int)
            else:
                raise ValueError("仅支持二分类标签 (0/1 或 False/True)")
        labels = labels.astype(int)

        # Step 4: 预测值类型验证和处理
        if expected_type == "probs":
            if np.any(predictions < 0) or np.any(predictions > 1):
                logger.warning("预测值超出[0,1]范围，将进行裁剪")
                predictions = np.clip(predictions, self.EPSILON, 1 - self.EPSILON)
        elif expected_type == "logits":
            # logits可以是任何实数值，但要检查是否有无效值
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                raise ValueError("logits包含无效值(NaN或Inf)")

        return predictions, labels

    def calculate_metrics(self, labels: np.ndarray, calibrated_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        计算校准后的综合评估指标

        该方法计算多种评估指标来全面评估校准效果，包括：
        1. 校准质量指标：ECE (期望校准误差) 和 MCE (最大校准误差)
        2. 预测质量指标：Brier Score (布里尔分数)

        指标说明：
        - ECE: 衡量预测置信度与实际准确率的平均偏差，越小越好
        - MCE: 衡量预测置信度与实际准确率的最大偏差，越小越好
        - Brier Score: 衡量概率预测的准确性，越小越好

        计算细节：
        - ECE和MCE使用配置的分箱数计算
        - Brier Score基于概率预测计算

        Args:
            labels: 真实二分类标签 {0,1}
            calibrated_probs: 校准后的概率预测值 [0,1]

        Returns:
            包含各项评估指标的字典:
            - "ece": Expected Calibration Error
            - "mce": Maximum Calibration Error
            - "brier_score": Brier Score

        Note:
            该方法依赖于外部Metrics类的实现，确保指标计算的一致性
        """
        # 计算校准质量专门指标
        calibrated_labels = (calibrated_probs >= threshold).astype(int)
        res = {
            "ece": Metrics.calculate_ece(labels, calibrated_probs, self.config.ece_bins),
            # "mce": Metrics.calculate_mce(labels, calibrated_probs, self.config.ece_bins),
            "brier_score": Metrics.brier_score(labels, calibrated_probs),
            "f1_score": f1_score(labels, calibrated_labels, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, calibrated_labels),
        }
        return res

    def temperature_scaling(self, logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        温度缩放校准 (Temperature Scaling)

        算法原理：
        Temperature Scaling是一种简单而有效的校准方法，通过在softmax函数中引入温度参数T：
        P_calibrated = softmax(logits / T) = sigmoid(logits / T)  # 二分类情况

        特点：
        - 保持预测的相对顺序不变（单调变换）
        - 只需要一个全局参数T，参数效率高
        - 特别适合深度学习模型的校准
        - T > 1时使概率分布更平滑，T < 1时使分布更尖锐

        优化目标：
        最小化Expected Calibration Error (ECE)，衡量校准质量

        适用场景：
        - 深度神经网络的置信度校准
        - 需要保持预测排序的场景
        - 计算资源受限的环境

        Args:
            logits: 模型输出的原始logits (未经sigmoid变换)
                   注意：必须是logits而非概率，因为温度缩放作用于softmax之前
            labels: 真实二分类标签 {0,1}

        Returns:
            校准结果字典，包含：
            - method: 方法名称
            - method_params: {"temperature": 最优温度值}
            - metrics: 校准后的评估指标(ECE, MCE, Brier Score等)
            - success: 是否成功

        Mathematical Note:
            对于二分类：P(y=1|x) = σ(logits/T) = 1/(1 + exp(-logits/T))
            其中σ是sigmoid函数，T是待优化的温度参数
        """
        try:
            # 输入验证：确保logits和labels格式正确且长度匹配
            logits, labels = self._validate_inputs(logits, labels, "logits")

            def objective(temperature):
                # 温度必须为正数，否则失去物理意义
                if temperature <= 0:
                    return np.inf
                try:
                    # 应用温度缩放：将logits除以温度后通过sigmoid得到校准概率
                    calibrated_probs = expit(logits / temperature)
                    # 计算ECE分数作为优化目标
                    return Metrics.calculate_ece(labels, calibrated_probs, self.config.ece_bins)
                except Exception:
                    # 数值计算异常时返回无穷大，避免优化器崩溃
                    return np.inf

            # 使用有界标量优化器寻找最优温度
            # 搜索范围[0.1, 10.0]覆盖了大多数实际场景的合理温度值
            result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")

            if not result.success:
                raise CalibrationOptimizationError("温度缩放优化失败")

            optimal_temp = result.x

            # 使用最优温度计算最终的校准概率
            calibrated_probs = expit(logits / optimal_temp)
            # 计算校准后的各项评估指标
            metrics = self.calculate_metrics(labels, calibrated_probs)

            return {
                "method": "temperature_scaling",
                "method_params": {"temperature": optimal_temp},
                "calibration_metrics": metrics,
                "success": True,
            }

        except Exception as e:
            logger.error(f"温度缩放失败: {e}")
            return {"method": "temperature_scaling", "success": False, "error": str(e)}

    def platt_scaling(self, pred_probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Platt缩放校准 (Platt Scaling)

        算法原理：
        Platt Scaling通过训练一个额外的逻辑回归模型来校准概率：
        P_calibrated = sigmoid(A * logit(P_original) + B)
        其中A和B是通过最大似然估计学习的参数

        数学形式：
        - 将原始概率转换为logits: z = logit(p) = log(p/(1-p))
        - 逻辑回归映射: P_cal = 1/(1 + exp(-(A*z + B)))

        特点：
        - 参数化方法，只需学习两个参数(A, B)
        - 基于最大似然估计，有理论支撑
        - 适合小数据集，避免过拟合
        - 可以学习非线性的校准映射

        适用场景：
        - 传统机器学习模型(SVM, Random Forest等)
        - 小样本数据集
        - 需要可解释校准参数的场景

        历史背景：
        最初由John Platt提出用于SVM输出的概率校准

        Args:
            pred_probs: 模型预测的概率值，范围[0,1]
            labels: 真实二分类标签 {0,1}

        Returns:
            校准结果字典，包含：
            - method: 方法名称
            - method_params: {"coef": 系数A, "intercept": 截距B}
            - metrics: 校准后的评估指标
            - success: 是否成功

        Mathematical Details:
            优化目标：max Σ[y*log(σ(Az+B)) + (1-y)*log(1-σ(Az+B))]
            其中z = logit(p), σ是sigmoid函数
        """
        try:
            # 输入验证：确保概率值在[0,1]范围内
            pred_probs, labels = self._validate_inputs(pred_probs, labels, "probs")

            # Step 1: 将概率转换为logits，为逻辑回归做准备
            # logits = log(p/(1-p))，这样可以将[0,1]映射到(-∞,+∞)
            logits = self._probs_to_logits(pred_probs)

            # Step 2: 训练逻辑回归模型进行校准
            # 逻辑回归会学习 P(y=1) = sigmoid(A * logits + B)
            # 其中A和B是需要学习的校准参数
            platt_model = LogisticRegression(random_state=42, max_iter=1000)  # 确保结果可重复

            # 拟合校准模型：输入是logits，输出是真实标签
            platt_model.fit(logits.reshape(-1, 1), labels)

            # Step 3: 使用训练好的模型计算校准后的概率
            # predict_proba返回[P(y=0), P(y=1)]，我们需要P(y=1)
            calibrated_probs = platt_model.predict_proba(logits.reshape(-1, 1))[:, 1]

            # Step 4: 计算校准后的评估指标
            metrics = self.calculate_metrics(labels, calibrated_probs)

            return {
                "method": "platt_scaling",
                "method_params": {
                    # 保存校准参数，用于后续推理阶段
                    "coef": float(platt_model.coef_[0][0]),  # 系数A
                    "intercept": float(platt_model.intercept_[0]),  # 截距B
                },
                "calibration_metrics": metrics,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Platt缩放失败: {e}")
            return {"method": "platt_scaling", "success": False, "error": str(e)}

    def isotonic_regression(self, pred_probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        等距回归校准 (Isotonic Regression Calibration)

        算法原理：
        等距回归学习一个单调递增的分段常数函数来校准概率：
        P_calibrated = f(P_original)
        其中f是通过保序回归(isotonic regression)学习的单调递增函数

        数学约束：
        对于任意 p1 < p2，必须满足 f(p1) ≤ f(p2) (单调性约束)

        特点：
        - 非参数化方法，无函数形式假设
        - 保证单调性：置信度高的样本校准后置信度不会更低
        - 可以学习复杂的非线性校准关系
        - 对异常值相对鲁棒
        - 适合大数据集，小数据集容易过拟合

        算法细节：
        使用Pool Adjacent Violators (PAV)算法求解保序回归问题
        时间复杂度：O(n)，空间复杂度：O(n)

        适用场景：
        - 大数据集的概率校准
        - 概率分布复杂、非线性的情况
        - 不想假设特定校准函数形式的场景
        - 需要保证预测排序的应用

        Args:
            pred_probs: 模型预测的概率值，范围[0,1]
            labels: 真实二分类标签 {0,1}

        Returns:
            校准结果字典，包含：
            - method: 方法名称
            - method_params: 包含校准映射关键点的字典
            - metrics: 校准后的评估指标
            - success: 是否成功

        Note:
            校准函数以分段常数形式存储，通过线性插值应用到新数据
        """
        try:
            # 输入验证：确保概率值在合理范围内
            pred_probs, labels = self._validate_inputs(pred_probs, labels, "probs")

            # Step 1: 创建等距回归模型
            # out_of_bounds="clip"确保预测值在[0,1]范围内
            iso_reg = IsotonicRegression(out_of_bounds="clip")

            # Step 2: 拟合保序回归模型
            # 学习从预测概率到真实标签的单调递增映射
            # 内部使用PAV算法求解约束优化问题：
            # min Σ(f(pi) - yi)² subject to f(p1) ≤ f(p2) ≤ ... ≤ f(pn)
            iso_reg.fit(pred_probs, labels)

            # Step 3: 应用学习到的校准函数
            # transform方法应用学习到的分段常数函数
            calibrated_probs = iso_reg.transform(pred_probs)

            # Step 4: 计算校准后的评估指标
            metrics = self.calculate_metrics(labels, calibrated_probs)

            # Step 5: 保存校准映射的关键点用于后续预测
            # X_thresholds_: 分段点的横坐标(输入概率值)
            # y_thresholds_: 分段点的纵坐标(校准后概率值)
            # 这些点定义了分段常数校准函数
            calibration_points = list(zip(iso_reg.X_thresholds_.tolist(), iso_reg.y_thresholds_.tolist()))

            return {
                "method": "isotonic_regression",
                "method_params": {
                    # 保存校准函数的关键点，用于推理阶段的线性插值
                    "thresholds": iso_reg.X_thresholds_.tolist(),
                    "calibration_points": calibration_points,
                },
                "calibration_metrics": metrics,
                "success": True,
            }

        except Exception as e:
            logger.error(f"等距回归失败: {e}")
            return {"method": "isotonic_regression", "success": False, "error": str(e)}

    def histogram_binning(self, pred_probs: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> Dict[str, Any]:
        """
        直方图分箱校准 (Histogram Binning Calibration)

        算法原理：
        将预测概率空间[0,1]分成若干个等宽的bins，每个bin内的所有预测值
        都被校准为该bin内真实标签的平均值（即该bin的经验准确率）

        数学表示：
        对于bin i: P_calibrated = Σ(y_j) / |bin_i|  ∀ p_j ∈ bin_i
        其中y_j是bin内样本的真实标签，|bin_i|是bin内样本数量

        特点：
        - 非参数化方法，分段常数函数
        - 直观易懂，每个bin的校准值就是该bin的准确率
        - 计算简单高效，内存占用少
        - 容易过拟合，特别是在小数据集上
        - 分箱边界可能造成不连续性

        优缺点：
        ✓ 简单直观，易于理解和实现
        ✓ 计算效率高，适合大规模数据
        ✓ 不需要参数调优（除了bin数量）
        ✗ 容易过拟合，特别是bins过多时
        ✗ 校准函数不连续，可能有跳跃
        ✗ 对bin数量敏感，需要仔细选择

        适用场景：
        - 大数据集，每个bin有足够样本
        - 概率分布相对均匀的情况
        - 计算资源受限的环境
        - 需要快速原型验证的场景

        Args:
            pred_probs: 模型预测的概率值，范围[0,1]
            labels: 真实二分类标签 {0,1}
            n_bins: 分箱数量，默认10个bins
                   建议每个bin至少有10-20个样本以保证统计稳定性

        Returns:
            校准结果字典，包含：
            - method: 方法名称
            - method_params: 分箱边界和每个bin的校准概率
            - metrics: 校准后的评估指标
            - success: 是否成功

        Implementation Details:
            - 使用等宽分箱：[0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
            - 第一个bin包含0，最后一个bin包含1
            - 空bin使用bin中点作为校准值
        """
        try:
            # 输入验证：确保数据格式正确
            pred_probs, labels = self._validate_inputs(pred_probs, labels, "probs")

            # Step 1: 自适应调整分箱数量
            # 确保每个bin有足够的样本，避免过拟合
            # 经验法则：每个bin至少5个样本
            n_bins = max(2, min(n_bins, len(pred_probs) // 5))

            # Step 2: 创建等宽分箱边界
            # [0, 1/n_bins, 2/n_bins, ..., 1]
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]  # [0, 1/n_bins, ..., (n-1)/n_bins]
            bin_uppers = bin_boundaries[1:]  # [1/n_bins, 2/n_bins, ..., 1]

            # Step 3: 计算每个分箱的校准概率（经验准确率）
            bin_calibrated_probs = []  # 每个bin的校准概率
            bin_counts = []  # 每个bin的样本数量

            for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
                # 边界处理：第一个分箱包含下边界，其他分箱不包含下边界但包含上边界
                # 这样确保边界值0和1被正确分配，避免遗漏
                if i == 0:
                    # 第一个bin: [0, upper]，包含0
                    in_bin = (pred_probs >= bin_lower) & (pred_probs <= bin_upper)
                else:
                    # 其他bin: (lower, upper]，不包含下边界
                    in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)

                bin_count = in_bin.sum()
                bin_counts.append(int(bin_count))

                if bin_count > 0:
                    # 该分箱的校准概率 = 该分箱中真实标签的平均值
                    # 这就是该bin的经验准确率
                    bin_accuracy = labels[in_bin].mean()
                    bin_calibrated_probs.append(float(bin_accuracy))
                else:
                    # 处理空分箱：使用分箱中点作为校准概率
                    # 这是一个合理的默认值，避免校准函数有空洞
                    bin_mid_point = (bin_lower + bin_upper) / 2
                    bin_calibrated_probs.append(float(bin_mid_point))
                    logger.debug(f"Bin [{bin_lower:.2f}, {bin_upper:.2f}] 为空，使用中点 {bin_mid_point:.2f}")

            # Step 4: 应用校准映射到所有预测值
            calibrated_probs = np.zeros_like(pred_probs)
            for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
                # 使用相同的边界处理逻辑
                if i == 0:
                    in_bin = (pred_probs >= bin_lower) & (pred_probs <= bin_upper)
                else:
                    in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)

                # 将该bin内的所有预测值设置为该bin的校准概率
                calibrated_probs[in_bin] = bin_calibrated_probs[i]

            # Step 5: 计算校准后的评估指标
            metrics = self.calculate_metrics(labels, calibrated_probs)

            return {
                "method": "histogram_binning",
                "method_params": {
                    "n_bins": n_bins,
                    "bin_boundaries": bin_boundaries.tolist(),
                    "bin_calibrated_probs": bin_calibrated_probs,
                    "bin_counts": bin_counts,  # 记录每个bin的样本数，用于诊断
                },
                "calibration_metrics": metrics,
                "success": True,
            }

        except Exception as e:
            logger.error(f"直方图分箱失败: {e}")
            return {"method": "histogram_binning", "success": False, "error": str(e)}

    def beta_calibration(self, pred_probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Beta校准 (Beta Calibration) - 基于Beta分布的概率校准方法

        算法原理：
        Beta校准假设校准函数是Beta分布的累积分布函数(CDF)：
        P_calibrated = Beta_CDF(P_original; α, β)
        其中α和β是需要学习的Beta分布参数

        数学基础：
        Beta分布CDF: F(x; α, β) = B(x; α, β) / B(α, β)
        其中B是Beta函数，x ∈ [0,1]

        参数含义：
        - α > 0: 形状参数，控制分布在0附近的行为
        - β > 0: 形状参数，控制分布在1附近的行为
        - α = β = 1时退化为均匀分布
        - α > β时分布右偏，α < β时分布左偏

        特点：
        ✓ 参数化方法，只需要两个参数(α, β)
        ✓ 基于概率分布理论，有数学支撑
        ✓ 可以建模各种形状的校准曲线
        ✓ 计算效率高，适合大规模数据
        ✗ 假设校准函数是Beta分布CDF，可能不适合所有情况
        ✗ 参数估计可能不稳定，特别是极端数据分布

        参数估计策略（多层次后备）：
        1. 最大似然估计(MLE) - 精度最高，首选方法
        2. 矩估计法(Method of Moments) - 稳定可靠，备选方法
        3. 简化优化 - 基于对数似然的启发式优化
        4. 默认参数 - 保守的最后备选(α=1, β=1)

        适用场景：
        - 大数据集，有足够样本支持参数估计
        - 预测概率分布不太极端的情况
        - 需要平滑校准函数的场景
        - 计算资源充足的环境

        Args:
            pred_probs: 模型预测的概率值，范围[0,1]
            labels: 真实二分类标签 {0,1}

        Returns:
            校准结果字典，包含：
            - method: 方法名称
            - method_params: {"alpha": α参数, "beta": β参数, "estimation_method": 使用的估计方法}
            - metrics: 校准后的评估指标
            - success: 是否成功

        Mathematical Details:
            MLE目标: max Σ[log(f(pi; α, β))]
            矩估计: α = μ*((μ(1-μ)/σ²) - 1), β = (1-μ)*((μ(1-μ)/σ²) - 1)
            其中μ和σ²是样本均值和方差
        """
        try:
            # 输入验证：确保数据格式正确
            pred_probs, labels = self._validate_inputs(pred_probs, labels, "probs")

            # Step 1: 数据质量检查和预警
            # 检查预测概率的多样性
            unique_probs = len(np.unique(pred_probs))
            if unique_probs < 10:
                logger.warning(f"预测概率只有{unique_probs}个不同值，Beta校准可能效果有限")

            # 检查概率分布是否过于极端
            # 极端概率(接近0或1)会导致Beta分布参数估计困难
            extreme_count = np.sum((pred_probs < 0.01) | (pred_probs > 0.99))
            extreme_ratio = extreme_count / len(pred_probs)
            if extreme_ratio > 0.8:
                logger.warning(f"预测概率过于极端({extreme_ratio:.1%}在[0,0.01]或[0.99,1]范围)，Beta校准效果可能有限")

            # Step 2: 计算基本统计量，用于后续参数估计
            mean_prob = np.mean(pred_probs)  # 样本均值μ
            var_prob = np.var(pred_probs)  # 样本方差σ²

            # 初始化参数和估计方法标记
            alpha, beta_param = 1.0, 1.0  # 默认参数(均匀分布)
            estimation_method = "fallback"  # 参数估计方法标记

            # Step 3: 参数估计策略1 - 最大似然估计(MLE)
            # MLE是理论上最优的参数估计方法
            try:
                logger.debug("尝试使用最大似然估计...")
                # scipy.stats.beta.fit使用MLE估计参数
                # floc=0, fscale=1固定支撑区间为[0,1]
                alpha_mle, beta_mle, _, _ = beta.fit(pred_probs, floc=0, fscale=1)

                # 验证MLE参数的合理性
                # 检查参数是否为正数、有限值且在合理范围内
                if (
                    alpha_mle > 0
                    and beta_mle > 0  # 必须为正数
                    and alpha_mle < 1000
                    and beta_mle < 1000  # 避免极端值
                    and not np.isnan(alpha_mle)
                    and not np.isnan(beta_mle)  # 非NaN
                ):
                    alpha, beta_param = alpha_mle, beta_mle
                    estimation_method = "MLE"
                    logger.debug(f"MLE成功: alpha={alpha:.4f}, beta={beta_param:.4f}")
                else:
                    raise ValueError(f"MLE参数超出合理范围: alpha={alpha_mle}, beta={beta_mle}")

            except Exception as mle_error:
                logger.debug(f"MLE失败: {mle_error}")

                # Step 4: 参数估计策略2 - 矩估计法
                # 当MLE失败时的稳定备选方案
                try:
                    logger.debug("使用矩估计法...")
                    # 矩估计法要求方差大于0且均值在(0,1)内
                    if var_prob > 0 and 0 < mean_prob < 1:
                        # 标准矩估计公式推导自:
                        # E[X] = α/(α+β) = μ
                        # Var[X] = αβ/((α+β)²(α+β+1)) = σ²
                        # 解得: α+β = μ(1-μ)/σ² - 1
                        moment_ratio = mean_prob * (1 - mean_prob) / var_prob - 1

                        if moment_ratio > 0:
                            # 根据矩估计公式计算参数
                            alpha_moment = mean_prob * moment_ratio
                            beta_moment = (1 - mean_prob) * moment_ratio

                            # 参数范围限制，确保数值稳定性
                            alpha_moment = max(0.1, min(alpha_moment, 100.0))
                            beta_moment = max(0.1, min(beta_moment, 100.0))

                            alpha, beta_param = alpha_moment, beta_moment
                            estimation_method = "moment_estimation"
                            logger.debug(f"矩估计成功: alpha={alpha:.4f}, beta={beta_param:.4f}")
                        else:
                            raise ValueError(f"矩估计计算出负的比值: {moment_ratio}")
                    else:
                        raise ValueError(f"矩估计条件不满足: var={var_prob}, mean={mean_prob}")

                except Exception as moment_error:
                    logger.debug(f"矩估计失败: {moment_error}")

                    # Step 5: 参数估计策略3 - 简化优化
                    # 最后的备选方案，使用启发式优化
                    try:
                        logger.debug("使用简化优化...")

                        def simplified_objective(alpha_val):
                            """
                            基于负对数似然的简化目标函数

                            使用简单的参数关系: β ≈ α*(1-μ)/μ
                            这个关系来自于Beta分布均值的近似
                            """
                            # 参数范围检查
                            if alpha_val <= 0.1 or alpha_val >= 50.0:
                                return np.inf

                            try:
                                # 使用均值关系估计beta参数
                                beta_val = max(0.1, alpha_val * (1 - mean_prob) / mean_prob)
                                if beta_val >= 50.0:
                                    return np.inf

                                # 计算校准后概率
                                calibrated_probs = beta.cdf(pred_probs, alpha_val, beta_val)
                                calibrated_probs = np.clip(calibrated_probs, self.EPSILON, 1 - self.EPSILON)

                                # 计算负对数似然作为优化目标
                                log_likelihood = np.sum(labels * np.log(calibrated_probs) + (1 - labels) * np.log(1 - calibrated_probs))
                                return -log_likelihood  # 最小化负对数似然
                            except:
                                return np.inf

                        # 使用有界优化器搜索最优alpha
                        result = minimize_scalar(simplified_objective, bounds=(0.5, 20.0), method="bounded")

                        if result.success and result.fun < np.inf:
                            alpha = result.x
                            beta_param = max(0.1, alpha * (1 - mean_prob) / mean_prob)
                            estimation_method = "simplified_optimization"
                            logger.debug(f"简化优化成功: alpha={alpha:.4f}, beta={beta_param:.4f}")
                        else:
                            raise ValueError(f"优化失败: success={result.success}, fun={result.fun}")

                    except Exception as opt_error:
                        # Step 6: 最后的保险策略 - 使用默认参数
                        logger.warning(f"所有参数估计方法都失败，使用默认参数: {opt_error}")
                        alpha, beta_param = 1.0, 1.0  # 均匀分布参数
                        estimation_method = "default"

            # Step 7: 最终参数验证
            # 确保参数的有效性，这是最后的安全检查
            if alpha <= 0 or beta_param <= 0:
                logger.warning(f"参数异常，重置为默认值: alpha={alpha}, beta={beta_param}")
                alpha, beta_param = 1.0, 1.0
                estimation_method = "default(Exceptional reset)"

            # Step 8: 计算校准后的概率
            try:
                # 使用学习到的Beta分布参数进行校准
                # beta.cdf计算Beta分布的累积分布函数值
                calibrated_probs = beta.cdf(pred_probs, alpha, beta_param)
                # 数值稳定性处理，确保概率在有效范围内
                calibrated_probs = np.clip(calibrated_probs, self.EPSILON, 1 - self.EPSILON)
            except Exception as calib_error:
                logger.error(f"概率校准计算失败: {calib_error}")
                # 如果校准失败，返回原始概率作为后备
                calibrated_probs = pred_probs
                estimation_method += "(Fault)"

            # Step 9: 计算校准后的评估指标
            metrics = self.calculate_metrics(labels, calibrated_probs)

            return {
                "method": "beta_calibration",
                "method_params": {
                    "estimation_method": estimation_method,  # 记录使用的估计方法
                    "alpha": float(alpha),  # Alpha参数
                    "beta": float(beta_param),  # Beta参数
                },
                "calibration_metrics": metrics,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Beta校准失败: {e}")
            return {"method": "beta_calibration", "success": False, "error": str(e), "estimation_method": "ERROR"}

    def execute_calibration(self, method_name: str, pred_probs: np.ndarray, true_labels: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        安全执行校准方法

        Args:
            method_name: 校准方法名称
            pred_probs: 预测概率数组
            true_labels: 真实标签数组

        Returns:
            统一格式的校准结果字典，包含：
            - method: 方法名称
            - method_params: 校准参数
            - calibration_metrics: 校准后评估指标
            - success: 是否成功
            - error: 错误信息(如果失败)
        """
        try:
            # 对于temperature_scaling，需要将概率转换为logits
            if method_name == CalibrationMethod.TEMPERATURE_SCALING.value:
                # 将概率转换为logits: logit(p) = log(p / (1-p))
                logits = self._probs_to_logits(pred_probs)
                result = self.temperature_scaling(logits, true_labels)
            else:
                # 其他方法直接使用概率
                method_map = {
                    CalibrationMethod.PLATT_SCALING.value: lambda: self.platt_scaling(pred_probs, true_labels),
                    CalibrationMethod.ISOTONIC_REGRESSION.value: lambda: self.isotonic_regression(pred_probs, true_labels),
                    CalibrationMethod.HISTOGRAM_BINNING.value: lambda: self.histogram_binning(pred_probs, true_labels),
                    CalibrationMethod.BETA_CALIBRATION.value: lambda: self.beta_calibration(pred_probs, true_labels),
                }

                if method_name not in method_map:
                    logger.warning(f"未知校准方法: {method_name}")
                    return None

                result = method_map[method_name]()

            # 统一处理返回结果
            if result and result.get("success", False):
                # 统一返回值格式，确保使用正确的键名
                calibration_metrics = result.get("calibration_metrics", result.get("metrics", {}))
                logger.info(f"{method_name} 校准成功 - {calibration_metrics}")
                return result

            else:
                error_msg = result.get("error", "未知错误") if result else "校准方法返回None"
                logger.warning(f"{method_name} 校准失败: {error_msg}")
                return {"method": method_name, "method_params": {}, "calibration_metrics": {}, "success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"{method_name} 校准出错: {e}")
            return {"method": method_name, "method_params": {}, "calibration_metrics": {}, "success": False, "error": str(e)}

    def apply_calibration_params(
        self,
        model_df: pd.DataFrame,
        calibration: Dict[str, Any],
        input_type: str = "probs",
        override_params: Dict[str, Any] = None,
        target_column: str = "pred_label",
        output_column: str = "calibration_prob",
    ) -> pd.DataFrame:
        # Step 1: 解析校准方法和参数
        method = calibration.get("method", "none")

        # 获取训练阶段学习到的校准参数
        params = calibration.get("method_params", {})

        # 应用用户指定的参数覆盖（用于实验或微调）
        if override_params:
            params = {**params, **override_params}  # 兼容Python 3.5+的字典合并语法
            logger.debug(f"使用覆盖参数: {override_params}")

        # Step 2: 验证输入数据的有效性
        if target_column not in model_df.columns:
            available_columns = list(model_df.columns)
            raise ValueError(f"指定的目标列 '{target_column}' 不存在于DataFrame中。可用列: {available_columns}")

        # 提取需要校准的预测值
        pred_values = model_df[target_column].values
        pred_values = np.asarray(pred_values).flatten()

        # Step 3: 处理边界情况
        if len(pred_values) == 0:
            # 空数据情况：创建与DataFrame行数匹配的空列
            model_df[output_column] = pd.Series(dtype=float, index=model_df.index)
            logger.warning("输入数据为空，返回空的校准结果")
            return model_df

        # 处理无效数值（NaN, Inf等）
        if np.any(np.isnan(pred_values)) or np.any(np.isinf(pred_values)):
            logger.warning("输入分数包含无效值，将被替换为默认值")
            # 使用合理的默认值替换无效值
            pred_values = np.nan_to_num(
                pred_values, nan=0.5, posinf=1.0, neginf=0.0  # NaN替换为中性概率  # +Inf替换为最大概率  # -Inf替换为最小概率
            )

        # Step 4: 应用校准变换
        # 调用专门的数组校准方法，保持代码模块化
        calibrated_values = self._apply_calibration_to_array(pred_values, method, params, input_type)

        # Step 5: 将校准结果添加到DataFrame
        model_df[output_column] = calibrated_values

        return model_df

    def _apply_calibration_to_array(
        self,
        pred_values: np.ndarray,
        method: str,
        params: Dict[str, Any],
        input_type: str,
    ) -> np.ndarray:
        """
        对数组应用校准方法的核心实现函数

        这是校准应用的核心算法实现，负责将训练好的校准参数
        应用到预测值数组上。每种校准方法都有其特定的应用逻辑。

        实现细节：
        - 统一的输入类型处理：自动在概率和logits之间转换
        - 数值稳定性保证：所有计算都包含epsilon裁剪
        - 参数验证：确保校准参数的完整性和有效性
        - 异常处理：方法失败时返回合理的默认值

        各方法的应用逻辑：
        1. Temperature Scaling: logits/T → sigmoid
        2. Platt Scaling: A*logits + B → sigmoid
        3. Isotonic Regression: 线性插值校准映射
        4. Histogram Binning: 分箱查找表映射
        5. Beta Calibration: Beta分布CDF变换

        Args:
            pred_values: 待校准的预测值数组
            method: 校准方法名称
            params: 校准参数字典（来自训练阶段）
            input_type: 输入类型 ("probs" 或 "logits")

        Returns:
            校准后的概率值数组，范围[0,1]

        Note:
            所有校准方法最终都输出概率值，即使输入是logits
        """

        if method == "temperature_scaling":
            """
            温度缩放应用：将温度参数应用到logits

            公式：P_cal = sigmoid(logits / T)
            其中T是训练阶段学习到的温度参数
            """
            temperature = params.get("temperature")
            if temperature is None:
                raise ValueError("温度缩放缺少必需的 'temperature' 参数")
            if temperature <= 0:
                raise ValueError(f"温度参数必须为正数，当前值: {temperature}")

            logger.debug(f"应用温度缩放: temperature={temperature}")

            if input_type == "probs":
                # 输入是概率：先转换为logits，再应用温度缩放
                logits = self._probs_to_logits(pred_values)
                return expit(logits / temperature)
            else:
                # 输入已经是logits：直接应用温度缩放
                return expit(pred_values / temperature)

        elif method == "platt_scaling":
            """
            Platt缩放应用：应用逻辑回归变换

            公式：P_cal = sigmoid(A * logits + B)
            其中A和B是训练阶段学习到的逻辑回归参数
            """
            coef = params.get("coef")
            intercept = params.get("intercept")
            if coef is None or intercept is None:
                raise ValueError("Platt缩放缺少必需的 'coef' 和 'intercept' 参数")

            logger.debug(f"应用Platt缩放: coef={coef}, intercept={intercept}")

            if input_type == "probs":
                # 输入是概率：转换为logits，应用线性变换，再sigmoid
                logits = self._probs_to_logits(pred_values)
                return expit(coef * logits + intercept)
            else:
                # 输入已经是logits：直接应用线性变换
                return expit(coef * pred_values + intercept)

        elif method == "isotonic_regression":
            """
            等距回归应用：使用校准点进行分段线性插值

            校准函数以分段常数形式存储为(x, y)点对，
            使用线性插值在这些点之间进行平滑连接
            """
            # 等距回归期望概率输入，因为训练时使用的是概率
            if input_type == "logits":
                pred_values = expit(pred_values)  # 转换为概率

            # 获取校准映射点
            points = params.get("calibration_points")
            if points is None:
                raise ValueError("等距回归缺少必需的 'calibration_points' 参数。请确保训练时正确保存了校准点。")

            logger.debug(f"应用等距回归: 使用{len(points)}个校准点")

            # 分离x和y坐标
            x_vals = [p[0] for p in points]  # 输入概率值
            y_vals = [p[1] for p in points]  # 校准后概率值

            # 验证单调性（等距回归的基本要求）
            if not all(x_vals[i] <= x_vals[i + 1] for i in range(len(x_vals) - 1)):
                logger.warning("校准点X值不是单调递增的，这可能导致插值错误")

            # 使用线性插值在校准点之间进行映射
            # numpy.interp会自动处理超出范围的值
            return np.interp(pred_values, x_vals, y_vals)

        elif method == "histogram_binning":
            """
            直方图分箱应用：根据分箱边界进行映射

            每个预测值根据其所在的bin被映射到该bin的校准概率
            这实现了分段常数校准函数
            """
            # 分箱校准期望概率输入
            if input_type == "logits":
                pred_values = expit(pred_values)  # 转换为概率

            bin_boundaries = params.get("bin_boundaries")
            bin_probs = params.get("bin_calibrated_probs")
            if bin_boundaries is None or bin_probs is None:
                raise ValueError("分箱校准缺少必需的 'bin_boundaries' 和 'bin_calibrated_probs' 参数")

            bin_boundaries = np.array(bin_boundaries)
            logger.debug(f"应用分箱校准: {len(bin_boundaries)-1}个分箱")

            # 初始化校准结果数组
            calibrated = np.zeros_like(pred_values)

            # 为每个分箱应用相应的校准概率
            for i in range(len(bin_boundaries) - 1):
                # 使用与训练时相同的边界处理逻辑
                if i == 0:
                    # 第一个bin包含下边界
                    mask = (pred_values >= bin_boundaries[i]) & (pred_values <= bin_boundaries[i + 1])
                else:
                    # 其他bin不包含下边界
                    mask = (pred_values > bin_boundaries[i]) & (pred_values <= bin_boundaries[i + 1])

                # 将该bin内的所有值设为该bin的校准概率
                calibrated[mask] = bin_probs[i]

            return calibrated

        elif method == "beta_calibration":
            """
            Beta校准应用：使用Beta分布CDF进行变换

            公式：P_cal = Beta_CDF(P_orig; α, β)
            使用训练阶段学习到的Beta分布参数
            """
            # Beta校准期望概率输入
            if input_type == "logits":
                pred_values = expit(pred_values)  # 转换为概率

            alpha = params.get("alpha")
            beta_param = params.get("beta")
            if alpha is None or beta_param is None:
                raise ValueError("Beta校准缺少必需的 'alpha' 和 'beta' 参数")
            if alpha <= 0 or beta_param <= 0:
                raise ValueError(f"Beta分布参数必须为正数，当前值: alpha={alpha}, beta={beta_param}")

            logger.debug(f"应用Beta校准: alpha={alpha}, beta={beta_param}")

            # 使用Beta分布累积分布函数进行校准
            calibrated_probs = beta.cdf(pred_values, alpha, beta_param)
            # 确保数值稳定性，避免极端值
            return np.clip(calibrated_probs, self.EPSILON, 1 - self.EPSILON)

        else:
            """
            未知校准方法的处理：返回合理的默认值

            如果遇到未实现的校准方法，记录警告并返回概率形式的原始值
            """
            logger.warning(f"未知校准方法: {method}，返回原始分数")

            if input_type == "logits":
                # 如果输入是logits，转换为概率后返回
                return expit(pred_values)
            else:
                # 如果输入已经是概率，直接返回
                return pred_values
