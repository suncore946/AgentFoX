class ModelCalibrationError(Exception):
    """模型校准相关异常"""

    pass


class DataValidationError(ModelCalibrationError):
    """数据验证异常"""

    pass


class CalibrationOptimizationError(ModelCalibrationError):
    """校准优化异常"""

    pass
