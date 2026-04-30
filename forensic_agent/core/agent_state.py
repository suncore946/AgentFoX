"""Agent state definitions.

中文说明: StageEnum 是 Agent 与执行图之间的轻量状态协议。
English: StageEnum is the lightweight state protocol between the agent and the
execution graph.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, NotRequired, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt.chat_agent_executor import AgentState

from .forensic_dataclass import FinalResponse


class StageEnum(Enum):
    """Supported workflow stages.

    中文说明: 名称全部使用大写枚举成员, 方便解析 `update stage to: xxx`。
    English: Enum member names are uppercase so `update stage to: xxx` can be
    parsed reliably.
    """

    INITIAL = "initial"
    SEMANTIC_LEVEL = "semantic_level"
    FINALLY_REPORT = "finally_report"
    DONE = "done"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"


class CustomAgentState(AgentState):
    """State carried through the LangGraph workflow.

    中文说明: image_path 是工具层读取当前图片的唯一入口。
    English: image_path is the single source used by tools to identify the
    current image.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    image_path: str
    origin_input: Dict[str, Any]
    workflow_id: NotRequired[int]
    current_stage: StageEnum
    final_response: FinalResponse | None = None
    metrics: NotRequired[Dict[str, Any]]
    remaining_steps: NotRequired[RemainingSteps]
