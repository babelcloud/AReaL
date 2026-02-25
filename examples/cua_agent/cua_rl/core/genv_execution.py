from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import httpx

from cua_rl.core.genv_http_client import AsyncGenvHttpClient

logger = logging.getLogger(__name__)
# Use RLTrainer logger for create_execution so logs show in training script output
_rl_logger = logging.getLogger("RLTrainer")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _extract_task_identifier(env: Any) -> str | None:
    task = getattr(env, "task", None)
    if task is None:
        return None
    genv_identifier = getattr(task, "genv_identifier", None)
    if isinstance(genv_identifier, str) and genv_identifier:
        return genv_identifier
    task_id = getattr(task, "id", None)
    if isinstance(task_id, str) and task_id:
        return task_id
    task_name = getattr(task, "name", None)
    if isinstance(task_name, str) and task_name:
        return task_name
    return None


def _extract_numeric_prefix(task_id: str) -> str | None:
    match = re.search(r"(\d+)", task_id)
    if not match:
        return None
    return match.group(1).lstrip("0") or match.group(1)


def _pick_task_identifier(task_hint: str | None, task_items: Iterable[dict]) -> str | None:
    task_ids = []
    task_names = {}
    task_numbers = {}
    for item in task_items:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        task_id = item.get("task_id") or item.get("id") or meta.get("id")
        if isinstance(task_id, str):
            task_ids.append(task_id)
        name = item.get("name") or meta.get("displayName") or meta.get("name")
        if isinstance(name, str):
            task_names[name] = task_id
        number = meta.get("number")
        if isinstance(number, str) and number.strip() and isinstance(task_id, str):
            task_numbers[number.strip()] = task_id

    if task_hint:
        if task_hint in task_ids:
            return task_hint
        if task_hint in task_names:
            return task_names[task_hint]
        if task_hint in task_numbers:
            return task_numbers[task_hint]

        hint_prefix = _extract_numeric_prefix(task_hint)
        if hint_prefix:
            # Prefer explicit meta.number matching when available.
            for num, tid in task_numbers.items():
                if _extract_numeric_prefix(num) == hint_prefix:
                    return tid
            for task_id in task_ids:
                if task_id.startswith(hint_prefix.zfill(len(task_id))):
                    return task_id
            for task_id in task_ids:
                if task_id.startswith(hint_prefix):
                    return task_id

    return task_ids[0] if task_ids else None


def _get_task_number_from_tasks(task_identifier: str, task_items: Iterable[dict]) -> Optional[str]:
    """Return task number (e.g. 088, 072) from task list, or None."""
    for item in task_items:
        if not isinstance(item, dict):
            continue
        tid = item.get("task_id") or item.get("id") or (item.get("meta") or {}).get("id")
        if tid != task_identifier:
            continue
        num = item.get("task_number")
        if isinstance(num, str) and num.strip():
            return num.strip()
        num = (item.get("meta") or {}).get("number")
        if isinstance(num, str) and num.strip():
            return num.strip()
    return None

def _build_env_build_details(
    *,
    start_time: float,
    execution_response: dict[str, Any],
    task_identifier: str,
    gym_id: str,
    gym_base_url: str,
) -> dict[str, Any]:
    duration = max(0.0, time.time() - start_time)
    return {
        "status": "success",
        "total_time": duration,
        "stages": [
            {
                "name": "create_execution",
                "status": "success",
                "duration": duration,
                "details": {
                    "gym_id": gym_id,
                    "gym_base_url": gym_base_url,
                    "task_identifier": task_identifier,
                    "execution_response": execution_response,
                },
            }
        ],
        "prehook_executed": False,
        "prehook_output": None,
    }


@dataclass
class GenvExecutionContext:
    gym_base_url: str
    gym_id: str
    execution_id: str
    task_identifier: str
    env_build: dict[str, Any]
    task_number: Optional[str] = None

    def agent_env_payload(self) -> dict[str, Any]:
        return {
            "type": "genv",
            "gymBaseUrl": self.gym_base_url,
            "gymId": self.gym_id,
            "executionId": self.execution_id,
        }


async def list_tasks(
    base_url: str, gym_id: str, *, max_retries: int = 3, retry_delay: float = 2.0
) -> list[dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        client = AsyncGenvHttpClient(gym_id=gym_id, base_url=base_url)
        try:
            return await client.list_tasks()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"list_tasks attempt {attempt}/{max_retries} failed for {base_url}: {exc}"
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * attempt)
        finally:
            await client.close()
    raise last_exc  # type: ignore[misc]


async def create_execution(
    *,
    base_url: str,
    gym_id: str,
    task_identifier: str,
    options: Optional[dict[str, Any]] = None,
    task_number: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> dict[str, Any]:
    t0 = time.monotonic()
    task_label = ("task-%s" % task_number) if task_number else ("task=%s" % task_identifier)
    _rl_logger.info("create_execution start for %s", task_label)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        client = AsyncGenvHttpClient(gym_id=gym_id, base_url=base_url)
        try:
            result = await client.create_execution(
                task_identifier=task_identifier,
                options=options or {},
            )
            duration = time.monotonic() - t0
            execution_id = result.get("execution_id", "")
            # Gym may return box/device id under different keys (umetrip uses device.device_serial)
            box_id = (
                result.get("box_id")
                or result.get("gbox_id")
                or result.get("boxId")
                or result.get("container_id")
                or result.get("env_id")
                or result.get("environment_id")
            )
            if not box_id and isinstance(result.get("device"), dict):
                box_id = result["device"].get("device_serial") or result["device"].get("box_id") or result["device"].get("id")
            if not box_id and isinstance(result.get("environment"), dict):
                box_id = result["environment"].get("id") or result["environment"].get("box_id") or result["environment"].get("boxId")
            if not box_id and isinstance(result.get("box"), dict):
                box_id = result["box"].get("id") or result["box"].get("box_id")
            box_id = box_id or ""
            if not box_id and isinstance(result, dict):
                _rl_logger.info(
                    "create_execution done for %s, duration=%.2fs, execution_id=%s, box_id= (response keys: %s)",
                    task_label, duration, execution_id, list(result.keys()),
                )
            else:
                _rl_logger.info(
                    "create_execution done for %s, duration=%.2fs, execution_id=%s, box_id=%s",
                    task_label, duration, execution_id, box_id,
                )
            return result
        except Exception as exc:
            last_exc = exc
            err_msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            logger.warning(
                f"create_execution attempt {attempt}/{max_retries} failed for "
                f"task={task_identifier}: {err_msg}",
                exc_info=True,
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * attempt)
        finally:
            await client.close()
    raise last_exc  # type: ignore[misc]


async def evaluate_execution(
    *,
    base_url: str,
    gym_id: str,
    execution_id: str,
    answer_text: Optional[str] = None,
    context_text: Optional[str] = None,
    facts: Optional[dict[str, Any]] = None,
    execution_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if answer_text is None and isinstance(execution_data, dict):
        answer_from_execution = execution_data.get("answer_text")
        if isinstance(answer_from_execution, str):
            answer_text = answer_from_execution
    if answer_text is not None:
        payload["answer_text"] = answer_text
    if context_text is not None:
        payload["context_text"] = context_text
    if facts is not None:
        payload["facts"] = facts
    if isinstance(execution_data, dict):
        payload["execution_data"] = execution_data
    client = AsyncGenvHttpClient(gym_id=gym_id, base_url=base_url)
    try:
        return await client.evaluate_execution(
            execution_id=execution_id,
            execution_data=payload.get("execution_data"),
            answer_text=payload.get("answer_text"),
            context_text=payload.get("context_text"),
            facts=payload.get("facts"),
        )
    finally:
        await client.close()


async def terminate_execution(
    *, base_url: str, gym_id: str, execution_id: str
) -> dict[str, Any]:
    client = AsyncGenvHttpClient(gym_id=gym_id, base_url=base_url)
    try:
        return await client.terminate_execution(execution_id=execution_id)
    finally:
        await client.close()


def normalize_genv_env_config(env_payload: dict | None) -> dict[str, Any]:
    env_payload = env_payload or {}
    base_url = env_payload.get("gymBaseUrl") or env_payload.get("baseUrl") or "http://localhost:5010"
    gym_id = env_payload.get("gymId") or env_payload.get("gym_id") or "umetrip"
    create_gym_payload = env_payload.get("createGym") or env_payload.get("create_gym")
    return {
        "gym_base_url": str(base_url),
        "gym_id": str(gym_id),
        "create_gym_payload": create_gym_payload if isinstance(create_gym_payload, dict) else None,
    }


async def start_genv_execution(
    *,
    env: Any,
    gym_base_url: str,
    gym_id: str,
    create_gym_payload: dict | None = None,
    rollout_logger = None,
    preloaded_tasks: list[dict] | None = None,
) -> GenvExecutionContext:
    gym_base_url = _normalize_base_url(gym_base_url)
    start_time = time.time()

    if create_gym_payload:
        async with httpx.AsyncClient(base_url=gym_base_url, timeout=120.0) as client:
            response = await client.post("/api/v1/gyms", json=create_gym_payload)
            response.raise_for_status()

    if preloaded_tasks is not None and len(preloaded_tasks) > 0:
        tasks = preloaded_tasks
    else:
        tasks = await list_tasks(gym_base_url, gym_id)
    task_hint = _extract_task_identifier(env)
    task_identifier = _pick_task_identifier(task_hint, tasks)
    if not task_identifier:
        raise RuntimeError("Unable to resolve genv task identifier from gym tasks.")

    task_number = _get_task_number_from_tasks(task_identifier, tasks)
    execution_response = await create_execution(
        base_url=gym_base_url,
        gym_id=gym_id,
        task_identifier=task_identifier,
        task_number=task_number,
    )

    execution_id = execution_response.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        raise RuntimeError(f"Invalid execution response from gym: {execution_response}")

    env_build = _build_env_build_details(
        start_time=start_time,
        execution_response=execution_response,
        task_identifier=task_identifier,
        gym_id=gym_id,
        gym_base_url=gym_base_url,
    )

    if rollout_logger is not None:
        rollout_logger.trajectory_data["env_build"] = env_build

    return GenvExecutionContext(
        gym_base_url=gym_base_url,
        gym_id=gym_id,
        execution_id=execution_id,
        task_identifier=task_identifier,
        env_build=env_build,
        task_number=task_number,
    )
