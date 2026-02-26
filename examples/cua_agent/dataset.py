"""Build HuggingFace Dataset from gym tasks for CUA Agent training."""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)


def get_cua_task_dataset(
    *,
    gym_base_url: str,
    gym_id: str,
    split_type: str | None = "train",
    train_ratio: float = 0.8,
    limit: int | None = None,
    seed: int = 42,
    task_names: str | None = None,
    eval_number_range: str | None = None,
    tags: str | None = None,
) -> Dataset:
    """Load gym tasks and return a HuggingFace Dataset of task dicts (one row per task)."""
    from cua_rl.task_loader import TaskSourceConfig, load_tasks_from_config

    config = TaskSourceConfig(
        source_type="gym",
        gym_base_url=gym_base_url,
        gym_id=gym_id,
        train_ratio=train_ratio,
        split_type=split_type,
        limit=limit,
        seed=seed,
        task_names=task_names,
        eval_number_range=eval_number_range,
        tags=tags,
    )
    tasks = load_tasks_from_config(config)
    rows = []
    for t in tasks:
        # Get task_number from genv_task_data if available
        task_number = None
        if hasattr(t, 'genv_task_data') and isinstance(t.genv_task_data, dict):
            meta = t.genv_task_data.get('meta', {})
            task_number = meta.get('number')
        rows.append({
            "task_id": t.id,
            "name": t.name,
            "description": t.description,
            "instruction": getattr(t, 'instruction', None) or t.description,
            "task_number": task_number,
            "tags": t.tags if hasattr(t, 'tags') else [],
            "gym_base_url": gym_base_url,
            "gym_id": gym_id,
            "difficulty": t.difficulty.value,
            "category": t.category.value,
        })
    logger.info("CUA task dataset: %d tasks (split=%s)", len(rows), split_type)
    return Dataset.from_list(rows)


def get_cua_train_and_valid_datasets(
    gym_base_url: str,
    gym_id: str,
    train_ratio: float = 0.8,
    limit: int | None = None,
    seed: int = 42,
    task_names: str | None = None,
    eval_number_range: str | None = None,
    tags: str | None = None,
) -> tuple[Dataset, Dataset]:
    """Return (train_dataset, valid_dataset) for CUA."""
    train_ds = get_cua_task_dataset(
        gym_base_url=gym_base_url,
        gym_id=gym_id,
        split_type="train",
        train_ratio=train_ratio,
        limit=limit,
        seed=seed,
        task_names=task_names,
        eval_number_range=eval_number_range,
        tags=tags,
    )
    valid_ds = get_cua_task_dataset(
        gym_base_url=gym_base_url,
        gym_id=gym_id,
        split_type="eval",
        train_ratio=train_ratio,
        limit=limit,
        seed=seed,
        task_names=task_names,
        eval_number_range=eval_number_range,
        tags=tags,
    )
    return train_ds, valid_ds
