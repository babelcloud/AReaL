"""CUA task types and metadata enums."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskCategory(str, Enum):
    """Task categories."""

    SYSTEM = "system"
    NAVIGATION = "navigation"
    SETTINGS = "settings"
    APP = "app"
    INPUT = "input"


@dataclass
class CUATask:
    """A task for CUA agent to complete."""

    id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    category: TaskCategory
    max_steps: int = 10

    # Validation config
    validation_type: str = "state"  # "state", "screenshot", "api"
    validation_query: Optional[str] = None
    expected_result: Optional[Any] = None

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "category": self.category.value,
            "max_steps": self.max_steps,
            "validation_type": self.validation_type,
            "validation_query": self.validation_query,
            "expected_result": self.expected_result,
            "tags": self.tags,
            "prerequisites": self.prerequisites,
        }


__all__ = [
    "CUATask",
    "TaskDifficulty",
    "TaskCategory",
]
