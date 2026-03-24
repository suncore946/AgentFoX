from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class ConflictResolutionStrategy(Enum):
    """冲突解决策略枚举"""

    HIGHEST_CONFIDENCE = "highest_confidence"
    STATIC_WEIGHT_VOTING = "static_weight_voting"
    CKB_ARBITRATION = "ckb_arbitration"
    PROFILE_ARBITRATION = "profile_arbitration"
    UMPIRE_MODEL = "umpire_model"
    EMBRACE_UNCERTAINTY = "embrace_uncertainty"


@dataclass
class ModelOutput:
    """模型输出数据"""

    model_name: str
    prediction: str
    raw_logits: np.ndarray
    calibrated_confidence: float
    reasoning: Optional[str] = None


@dataclass
class ConflictResolutionResult:
    """冲突解决结果"""

    final_prediction: str
    final_confidence: float
    strategy_used: ConflictResolutionStrategy
    reasoning: str
    evidence: List[str]
    model_contributions: Dict[str, float]
    needs_human_review: bool = False
    uncertainty_score: float = 0.0


@dataclass
class StageResult:
    """阶段结果数据类"""

    stage_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class ProcessingResult:
    """处理结果数据类"""

    image_path: str | Path
    classification: str
    confidence: float
    processing_time: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[str] = None
    decision_chain: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BatchProcessingResult:
    """批处理结果数据类"""

    total_processed: int
    successful: int
    failed: int
    results: List[ProcessingResult] = field(default_factory=list)
    processing_time: float = 0.0


class FinalResponse(BaseModel):
    is_success: bool = Field(
        ...,
        description="Audit Status: True if logic is consistent; False if contradictions are detected.",
    )
    reasoning: str = Field(
        ...,
        description="Audit Justification: If consistent, summarize facts supporting the verdict. If inconsistent, strictly list the logical conflicts.",
    )
    finally_result: int = Field(..., description="Verdict: 0 = Authentic, 1 = Fake. (Valid only when is_success is True).")


class CheckingResult(BaseModel):
    pred_result: int = Field(..., description="Final conclusion: 0 = authentic, 1 = fake/forged")
    is_success: bool = Field(
        ..., description="Indicates whether the analysis was ultimately completed successfully, regardless of any intermediate failures."
    )
