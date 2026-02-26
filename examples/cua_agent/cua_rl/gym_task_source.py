from __future__ import annotations

import logging
import os
import random
import re
from typing import Any, Optional

from cua_rl.core.genv_http_client import GenvHttpClient
from cua_rl.demo_tasks import CUATask, TaskCategory, TaskDifficulty

logger = logging.getLogger(__name__)


def _task_meta(task: dict[str, Any]) -> dict[str, Any]:
    meta = task.get("meta")
    return meta if isinstance(meta, dict) else {}


def _get_task_identifier(task: dict[str, Any]) -> str | None:
    """
    Extract the task identifier used by the gym API routes:
      - GET  /api/v1/{gym_id}/tasks/{task_identifier}
      - POST /api/v1/{gym_id}/tasks/{task_identifier}

    Backward/forward compatible with:
      - legacy: {"task_id": "...", "name": "..."}
      - new schema: {"meta": {"id": "<uuid>", "number": "011", ...}, ...}
    """
    task_id = task.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    task_id = task.get("id")
    if isinstance(task_id, str) and task_id:
        return task_id
    meta = _task_meta(task)
    meta_id = meta.get("id")
    if isinstance(meta_id, str) and meta_id:
        return meta_id
    return None


def _extract_digits(value: str) -> str | None:
    import re

    m = re.search(r"(\d+)", value)
    return m.group(1) if m else None


def _get_task_number(task: dict[str, Any], *, identifier: str) -> str | None:
    # Current gym API flattens number to top-level `task_number`.
    num = task.get("task_number")
    if isinstance(num, str) and num.strip():
        return num.strip()
    meta = _task_meta(task)
    num = meta.get("number")
    if isinstance(num, str) and num.strip():
        return num.strip()
    # Fallbacks for older shapes: numeric prefix in identifier/name
    digits = _extract_digits(identifier)
    if digits:
        return digits
    name = task.get("name")
    if isinstance(name, str):
        digits = _extract_digits(name)
        if digits:
            return digits
    return None


def _get_task_name(task: dict[str, Any], *, identifier: str) -> str:
    name = task.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    # Current gym API uses `task_name` (slug) + optional `display_name`.
    disp2 = task.get("display_name")
    if isinstance(disp2, str) and disp2.strip():
        return disp2.strip()
    name2 = task.get("task_name")
    if isinstance(name2, str) and name2.strip():
        return name2.strip()
    meta = _task_meta(task)
    disp = meta.get("displayName")
    if isinstance(disp, str) and disp.strip():
        return disp.strip()
    meta_name = meta.get("name")
    if isinstance(meta_name, str) and meta_name.strip():
        return meta_name.strip()
    return identifier


def _get_task_description(
    task: dict[str, Any],
    *,
    identifier: str,
    task_name: str,
) -> str:
    """
    Extract a human-readable task description.

    New schema (per `task.schema.json`): description is stored in `meta.description`.
    Legacy schema: may have top-level `description`.
    """
    desc = task.get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    meta = _task_meta(task)
    meta_desc = meta.get("description")
    if isinstance(meta_desc, str) and meta_desc.strip():
        return meta_desc.strip()
    # Do NOT fall back to instruction here — instruction is a separate field.
    return task_name or identifier


def _needs_full_task(task: dict[str, Any]) -> bool:
    """
    The gym list endpoint typically returns a summary without `description`,
    `env_config`, or `evaluation`. Fetch the full task payload when needed.
    """
    if not isinstance(task, dict):
        return True
    if isinstance(task.get("description"), str) and task.get("description", "").strip():
        return False
    # If any of these exist, we likely already have the full task payload.
    if "env_config" in task or "evaluation" in task:
        return False
    return True


def list_genv_gym_tasks(*, gym_id: str, gym_base_url: str | None) -> list[Any]:
    """List tasks from the Gym server using the HTTP API."""
    client = GenvHttpClient(gym_id=gym_id, base_url=gym_base_url)
    try:
        return client.list_tasks()
    finally:
        client.close()


def _iter_split(
    identifiers: list[str],
    *,
    seed: int,
    train_ratio: float,
    split_type: Optional[str],
) -> list[str]:
    rng = random.Random(seed)
    shuffled = identifiers.copy()
    rng.shuffle(shuffled)
    if split_type is None:
        return shuffled
    split_type_norm = split_type.strip().lower()
    if split_type_norm not in {"train", "eval"}:
        raise ValueError(f"Invalid split_type: {split_type!r} (expected 'train' or 'eval' or None)")
    cut = int(len(shuffled) * float(train_ratio))
    if split_type_norm == "train":
        return shuffled[:cut]
    return shuffled[cut:]


def _parse_number_range(s: str) -> tuple[int, int] | None:
    """Parse "001-032" or "1-32" to (1, 32) inclusive. Returns None if invalid."""
    if not s or not isinstance(s, str):
        return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s.strip())
    if not m:
        return None
    low, high = int(m.group(1)), int(m.group(2))
    if low > high:
        low, high = high, low
    return (low, high)


def _partition_by_number_range(
    task_map: dict[str, dict],
    eval_number_range: str,
) -> tuple[list[str], list[str]]:
    """Split identifiers into (eval_identifiers, train_identifiers) by task number range.
    eval_number_range e.g. "001-032" -> eval = numbers in [1,32], train = rest.
    """
    r = _parse_number_range(eval_number_range)
    if r is None:
        return [], list(task_map.keys())
    low, high = r
    eval_ids: list[str] = []
    train_ids: list[str] = []
    for identifier, task in task_map.items():
        num_str = _get_task_number(task, identifier=identifier)
        if num_str is None:
            train_ids.append(identifier)
            continue
        digits = _extract_digits(num_str)
        if not digits:
            train_ids.append(identifier)
            continue
        try:
            n = int(digits.lstrip("0") or "0")
        except ValueError:
            train_ids.append(identifier)
            continue
        if low <= n <= high:
            eval_ids.append(identifier)
        else:
            train_ids.append(identifier)
    return eval_ids, train_ids


def load_genv_tasks(
    *,
    gym_base_url: str | None = None,
    gym_id: str | None = None,
    seed: int,
    train_ratio: float = 0.8,
    split_type: Optional[str] = None,
    task_names: Optional[str] = None,
    limit: Optional[int] = None,
    eval_number_range: Optional[str] = None,
    tags: Optional[str] = None,
) -> list[CUATask]:
    """
    Load tasks from the Gym server.
    """
    resolved_gym_id = gym_id or os.getenv("GYM_ID") or "umetrip"
    gym_tasks = list_genv_gym_tasks(gym_id=resolved_gym_id, gym_base_url=gym_base_url)
    task_map: dict[str, dict[str, Any]] = {}
    for task in gym_tasks or []:
        if not isinstance(task, dict):
            continue
        identifier = _get_task_identifier(task)
        if isinstance(identifier, str) and identifier:
            task_map[identifier] = task

    all_identifiers = list(task_map.keys())
    if eval_number_range and eval_number_range.strip():
        eval_ids, train_ids = _partition_by_number_range(task_map, eval_number_range.strip())
        if split_type and split_type.strip().lower() == "eval":
            chosen_identifiers = sorted(eval_ids)
        else:
            chosen_identifiers = sorted(train_ids)
        logger.info(
            "[gym] split by eval_number_range=%r: eval=%d, train=%d; using %s (%d)",
            eval_number_range,
            len(eval_ids),
            len(train_ids),
            split_type or "train",
            len(chosen_identifiers),
        )
    else:
        chosen_identifiers = _iter_split(
            all_identifiers,
            seed=seed,
            train_ratio=train_ratio,
            split_type=split_type,
        )
    if not gym_tasks:
        logger.warning(
            "[gym] No tasks returned from gym_id=%s (base_url=%s)",
            resolved_gym_id,
            gym_base_url,
        )

    # Optional filter: restrict to a subset of tasks (comma-separated).
    #
    # For genv tasks, this filter accepts either:
    # - task directory identifiers (e.g. "019_check_recent_c919_negtive")
    # - numeric prefixes (e.g. "019" or "19")
    # - meta.id strings (e.g. "task-019") -> matched by numeric prefix
    if task_names:
        import re

        tokens = [t.strip() for t in str(task_names).split(",") if t.strip()]
        token_set = set(tokens)

        def _norm_num(s: str) -> str:
            s = s.strip()
            s = re.sub(r"^0+", "", s)
            return s or "0"

        def _matches(identifier: str) -> bool:
            task = task_map.get(identifier) or {}
            meta = _task_meta(task) if isinstance(task, dict) else {}

            # Candidate strings to match against.
            candidates: set[str] = {identifier}
            if isinstance(task, dict):
                for k in ("name", "task_name", "display_name", "task_number", "seed_id"):
                    v = task.get(k)
                    if isinstance(v, str) and v:
                        candidates.add(v)
            for key in ("id", "number", "name", "displayName"):
                val = meta.get(key)
                if isinstance(val, str) and val:
                    candidates.add(val)

            if token_set.intersection(candidates):
                return True

            # Prefix match (useful for legacy directory-like identifiers).
            for tok in token_set:
                if not tok:
                    continue
                for cand in candidates:
                    if cand.startswith(tok):
                        return True

            # Numeric matching: token "11" matches task number "011", etc.
            task_num = _get_task_number(task, identifier=identifier) if isinstance(task, dict) else None
            if isinstance(task_num, str) and task_num:
                task_num_norm = _norm_num(task_num)
                for tok in token_set:
                    tok_digits = _extract_digits(tok) if isinstance(tok, str) else None
                    if tok_digits and _norm_num(tok_digits) == task_num_norm:
                        return True
            return False

        chosen_identifiers = [ident for ident in chosen_identifiers if _matches(ident)]
        logger.info(
            "[gym] task_names filter applied: %r -> %d tasks",
            task_names,
            len(chosen_identifiers),
        )
    # Tags filter: restrict to tasks with specific tags (comma-separated, e.g. "easy,normal").
    # A task matches if it has ANY of the specified tags.
    if tags:
        tag_tokens = [t.strip().lower() for t in str(tags).split(",") if t.strip()]
        if tag_tokens:
            def _has_tag(identifier: str) -> bool:
                task = task_map.get(identifier)
                if not isinstance(task, dict):
                    return False
                task_tags = task.get("tags")
                if not isinstance(task_tags, list):
                    return False
                task_tags_lower = [str(t).lower() for t in task_tags]
                return any(tok in task_tags_lower for tok in tag_tokens)

            chosen_identifiers = [ident for ident in chosen_identifiers if _has_tag(ident)]
            logger.info(
                "[gym] tags filter applied: %r -> %d tasks",
                tags,
                len(chosen_identifiers),
            )

    if limit is not None and limit < len(chosen_identifiers):
        rng = random.Random(seed)
        chosen_identifiers = rng.sample(chosen_identifiers, int(limit))

    tasks: list[CUATask] = []

    for identifier in chosen_identifiers:
        task = task_map.get(identifier)
        if task is None:
            continue
        # Fetch full task payload (includes description/evaluation/env_config) if needed.
        if _needs_full_task(task):
            client = GenvHttpClient(gym_id=resolved_gym_id, base_url=gym_base_url)
            try:
                full = client.get_task(identifier)
            finally:
                client.close()
            if isinstance(full, dict) and full:
                task = {**task, **full}
                task_map[identifier] = task
        task_name = _get_task_name(task, identifier=identifier)
        description = _get_task_description(task, identifier=identifier, task_name=task_name)
        # Keep instruction separate for display purposes
        instruction = task.get("instruction")
        tags = ["gym"]
        task_tags = task.get("tags") if isinstance(task.get("tags"), list) else []
        for tag in task_tags:
            if tag not in tags:
                tags.append(tag)

        cua_task = CUATask(
            id=identifier,
            name=task_name,
            description=description,
            difficulty=TaskDifficulty.MEDIUM,
            category=TaskCategory.APP,
            max_steps=20,
            validation_type="genv_graphql",
            validation_query=None,
            expected_result=None,
            tags=tags,
        )
        cua_task.genv_identifier = identifier
        # Store instruction as an attribute for saving to DB
        cua_task.instruction = instruction
        meta_in = _task_meta(task)
        meta_out = dict(meta_in) if isinstance(meta_in, dict) else {}
        meta_out.setdefault("id", identifier)
        meta_out.setdefault("name", task_name)
        task_number = _get_task_number(task, identifier=identifier)
        if isinstance(task_number, str) and task_number:
            meta_out.setdefault("number", task_number)
        display_name = task.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            meta_out.setdefault("displayName", display_name.strip())
        raw_data = {
            "meta": meta_out,
            "description": description,
            # Keep instruction for backward compatibility in raw_data
            "instruction": instruction or description,
            "env": task.get("env_config") or task.get("env") or {},
            "evaluation": task.get("evaluation") or {},
        }
        cua_task.genv_task_data = raw_data
        tasks.append(cua_task)

    logger.info(
        "[gym] Loaded %d tasks from gym_id=%s (base_url=%s, split=%s, limit=%s)",
        len(tasks),
        resolved_gym_id,
        gym_base_url,
        split_type,
        limit,
    )
    return tasks
