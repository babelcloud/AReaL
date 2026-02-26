"""
Task loader for CUA RL training (gym-only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from cua_rl.demo_tasks import CUATask
from cua_rl.gym_task_source import load_genv_tasks

logger = logging.getLogger(__name__)


@dataclass
class TaskSourceConfig:
    """Configuration for loading gym tasks."""

    # Source type must be "gym".
    source_type: str

    # Gym configuration.
    gym_base_url: Optional[str] = None
    gym_id: Optional[str] = None
    train_ratio: float = 0.8
    split_type: Optional[str] = None  # "train" | "eval" | None
    task_names: Optional[str] = None  # Comma-separated filter list

    # Limit number of tasks (optional sampling).
    limit: Optional[int] = None

    # Eval by task number range (e.g. "001-032" -> eval=001..032, train=rest). Overrides train_ratio split.
    eval_number_range: Optional[str] = None

    # Tags filter: comma-separated list of tags (e.g. "easy,normal"). Tasks with ANY of these tags are included.
    tags: Optional[str] = None

    # Random seed for sampling and splitting.
    seed: Optional[int] = 42


def load_tasks_from_config(config: TaskSourceConfig) -> List[CUATask]:
    """Load tasks based on gym configuration."""
    if config.source_type != "gym":
        raise ValueError(
            f"Unsupported source_type: {config.source_type}. Only 'gym' is supported."
        )

    tasks = load_genv_tasks(
        gym_base_url=config.gym_base_url,
        gym_id=config.gym_id,
        seed=config.seed or 42,
        train_ratio=config.train_ratio,
        split_type=config.split_type,
        task_names=config.task_names,
        limit=config.limit,
        eval_number_range=config.eval_number_range,
        tags=config.tags,
    )
    logger.info(
        "Loaded %d tasks from source_type='gym' (base_url=%s, split=%s, limit=%s)",
        len(tasks),
        config.gym_base_url,
        config.split_type,
        config.limit,
    )
    return tasks


def load_tasks_from_multiple_sources(configs: List[TaskSourceConfig]) -> List[CUATask]:
    """Load tasks from multiple gym configs."""
    tasks: List[CUATask] = []
    for config in configs:
        tasks.extend(load_tasks_from_config(config))
    return tasks
