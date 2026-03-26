from typing import Annotated, Dict, Any, NotRequired, Optional, Sequence
from enum import Enum
from langchain.agents import AgentState
from typing import Any, Optional
from .forensic_dataclass import FinalResponse
from langgraph.managed import RemainingSteps
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class StageEnum(Enum):
    INITIAL = "initial"
    SEMANTIC_LEVEL = "semantic_level"
    EXPERT_PROFILEs = "expert_profiles"
    expert_RESULTS = "expert_results"
    EXPERT_ANALYSIS = "expert_analysis"
    CLUSTERING_ANALYSIS = "clustering_analysis"
    # CONFLICTING_RESOLUTION = "conflicting_resolution"
    FINALLY_REPORT = "finally_report"
    DONE = "done"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"


class CustomAgentState(AgentState):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    image_path: str
    origin_input: Dict[str, Any]
    current_stage: StageEnum
    final_response: FinalResponse = None
    remaining_steps: NotRequired[RemainingSteps]
