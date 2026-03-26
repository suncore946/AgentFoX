from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Profile:
    type: str
    model: str
    analogy: str
    core_mechanism: str
    primary_use_case: str
    major_limitation: str
    advantages_senior: List[str] = field(default_factory=list)
    disadvantages_senior: List[str] = field(default_factory=list)


@dataclass
class ModelProfiles:
    schema_version: str
    description: str
    profiles: List[Profile] = field(default_factory=list)
    model_evaluation: Optional[Dict] = None
