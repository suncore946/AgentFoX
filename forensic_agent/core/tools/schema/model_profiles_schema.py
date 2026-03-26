from typing import List, Optional

from pydantic import Field, field_validator
from .base_schema import BaseSchema


class ModelProfilesSchema(BaseSchema):
    """query_model_profiles工具的输入结构"""

    model_names: Optional[List[str]] = Field(
        default=None,
        description="Specify model name(s); if not specified, returns profile information for all models. Supports comma-separated string input.",
    )

    @field_validator("model_names", mode="before")
    @classmethod
    def split_comma_separated(cls, v):
        """处理逗号分隔的字符串输入"""
        if isinstance(v, str):
            # 按逗号分割并去除空格
            return [name.strip() for name in v.split(",") if name.strip()]
        return v
