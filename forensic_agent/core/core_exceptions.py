"""
AFA异常定义 - 统一的错误处理机制，使用现代Python特性
"""

from typing import Optional


class ForensicTemplateError(Exception):
    """取证模板相关异常"""

    pass


class JSONParsingError(Exception):
    """JSON 解析失败异常"""

    pass


class AFAException(Exception):
    """AFA系统基础异常类 - 使用现代Python异常链"""

    def __init__(self, message: str, error_code: Optional[str] = None, *, cause: Optional[Exception] = None) -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

        # 使用Python 3的异常链机制
        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AFAException):
    """配置相关错误"""

    def __init__(self, message: str, *, config_key: Optional[str] = None) -> None:
        error_code = f"CONFIG_ERROR_{config_key}" if config_key else "CONFIG_ERROR"
        super().__init__(message, error_code)
        self.config_key = config_key


class FeatureExtractionError(AFAException):
    """特征提取错误"""

    def __init__(self, message: str, *, feature_type: Optional[str] = None) -> None:
        error_code = f"FEATURE_ERROR_{feature_type}" if feature_type else "FEATURE_ERROR"
        super().__init__(message, error_code)
        self.feature_type = feature_type


class ModelExecutionError(AFAException):
    """模型执行错误"""

    def __init__(self, message: str, *, model_name: Optional[str] = None) -> None:
        error_code = f"MODEL_ERROR_{model_name}" if model_name else "MODEL_ERROR"
        super().__init__(message, error_code)
        self.model_name = model_name


class ReasoningError(AFAException):
    """Agent推理错误"""

    def __init__(self, message: str, *, step: Optional[str] = None) -> None:
        error_code = f"REASONING_ERROR_{step}" if step else "REASONING_ERROR"
        super().__init__(message, error_code)
        self.step = step


class StrategyPlanningError(AFAException):
    """策略规划错误"""

    def __init__(self, message: str, *, strategy: Optional[str] = None) -> None:
        error_code = f"STRATEGY_ERROR_{strategy}" if strategy else "STRATEGY_ERROR"
        super().__init__(message, error_code)
        self.strategy = strategy


class ReportGenerationError(AFAException):
    """报告生成错误"""

    def __init__(self, message: str, *, report_type: Optional[str] = None) -> None:
        error_code = f"REPORT_ERROR_{report_type}" if report_type else "REPORT_ERROR"
        super().__init__(message, error_code)
        self.report_type = report_type


class KnowledgeBaseError(AFAException):
    """知识库操作错误"""

    def __init__(self, message: str, *, operation: Optional[str] = None) -> None:
        error_code = f"KB_ERROR_{operation}" if operation else "KB_ERROR"
        super().__init__(message, error_code)
        self.operation = operation


class ValidationError(AFAException):
    """数据验证错误"""

    def __init__(self, message: str, *, field: Optional[str] = None) -> None:
        error_code = f"VALIDATION_ERROR_{field}" if field else "VALIDATION_ERROR"
        super().__init__(message, error_code)
        self.field = field


class OrchestrationError(AFAException):
    """编排流程错误"""

    def __init__(self, message: str, *, stage: Optional[str] = None) -> None:
        error_code = f"ORCHESTRATION_ERROR_{stage}" if stage else "ORCHESTRATION_ERROR"
        super().__init__(message, error_code)
        self.stage = stage


class WorkflowExceptionError(AFAException):
    """工作流错误"""

    def __init__(self, message: str, *, workflow_stage: Optional[str] = None) -> None:
        error_code = f"WORKFLOW_ERROR_{workflow_stage}" if workflow_stage else "WORKFLOW_ERROR"
        super().__init__(message, error_code)
        self.workflow_stage = workflow_stage


class ModelRunnerError(AFAException):
    """数据验证错误"""

    def __init__(self, message: str, *, field: Optional[str] = None) -> None:
        error_code = f"MODEL_ERROR_{field}" if field else "MODEL_ERROR"
        super().__init__(message, error_code)
        self.field = field
