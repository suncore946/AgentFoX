from typing import Any, Dict, List

from pydantic import Field
from .base_schema import BaseSchema


class ClusteringProfilesSchema(BaseSchema):
    """query_model_profiles工具的输入结构"""

    clustering_result: Dict[str, List[Any]] = Field(
        ...,
        description="Clustering result. The key is the clustering method, value is the list of image feature clusters corresponding to that method",
    )
