from typing import Annotated
from pydantic import BaseModel
from langgraph.prebuilt import InjectedState


class BaseSchema(BaseModel):
    """Base schema with injected state"""

    state: Annotated[dict, InjectedState]

    def get_image_path(self):
        return self.state["image_path"]
