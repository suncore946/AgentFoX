"""Shared data models for the minimal AgentFoX runtime.

中文说明: 这里只保留 test/analyze 路径需要的数据结构, 不包含训练、校准或专家融合模型。
English: This file keeps only the data structures needed by test/analyze and
does not include training, calibration, or expert-fusion models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class ProcessingResult:
    """Optional single-image processing summary.

    中文说明: 主要用于类型标注和未来 API 封装, CLI 当前保存 dict 结果。
    English: Mainly used for typing and future API wrappers; the CLI currently
    writes dict results.
    """

    image_path: str | Path
    classification: str
    confidence: float
    processing_time: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[str] = None
    decision_chain: List[Dict[str, Any]] = field(default_factory=list)


class FinalResponse(BaseModel):
    """Structured final verdict produced by the reporter.

    中文说明: finally_result 必须遵循 0=真实图像, 1=AI 生成/伪造图像。
    English: finally_result must follow 0=authentic image, 1=AI-generated or
    forged image.
    """

    is_success: bool = Field(
        ...,
        description="True if the report is internally consistent and usable.",
    )
    reasoning: str = Field(
        ...,
        description="Concise forensic justification grounded in the conversation history.",
    )
    finally_result: int = Field(
        ...,
        ge=0,
        le=1,
        description="Verdict: 0 = authentic, 1 = AI-generated/forged.",
    )


class CheckingResult(BaseModel):
    """Compatibility result for simple binary checks.

    中文说明: 保留轻量兼容结构, 不引入旧专家工具依赖。
    English: Kept as a lightweight compatibility structure without old expert
    tool dependencies.
    """

    pred_result: int = Field(..., ge=0, le=1, description="Final conclusion: 0 = authentic, 1 = fake/forged")
    is_success: bool = Field(..., description="Whether the analysis completed successfully.")
