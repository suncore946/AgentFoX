"""Shared schema for LangGraph tool calls.

中文说明: 工具不让用户手动输入 image_path, 而是从 Agent 状态中注入。
English: Tools do not require users to pass image_path manually; it is injected
from the agent state.
"""

from typing import Annotated

from langgraph.prebuilt import InjectedState
from pydantic import BaseModel


class BaseSchema(BaseModel):
    """Base tool schema with injected state.

    中文说明: LangGraph 会填充 state 字段, 工具从中读取当前图片路径。
    English: LangGraph fills the state field, and tools read the current image
    path from it.
    """

    state: Annotated[dict, InjectedState]

    def get_image_path(self) -> str:
        """Return the current image path.

        中文说明: image_path 由 agent_pipeline 的 batch 分组传入。
        English: image_path is supplied by the batch grouping in agent_pipeline.
        """
        return self.state["image_path"]
