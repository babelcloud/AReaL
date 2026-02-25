"""
HTTP-only rollout recorder for training-monitor ingest.

This recorder mirrors RolloutRecorder's public methods but writes via
IngestClient instead of a local database session.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from cua_rl.database.ingest_client import (
    IngestClient,
    _sanitize_payload_for_log,
)

logger = logging.getLogger(__name__)


class HttpRolloutRecorder:
    def __init__(
        self,
        ingest_client: IngestClient,
        rollout_uuid: str,
        training_id: Optional[int],
        source_type: str,
        model_path: str,
        step_id: Optional[int] = None,
        eval_id: Optional[int] = None,
        baseline_id: Optional[int] = None,
    ) -> None:
        self.ingest_client = ingest_client
        self.rollout_uuid = rollout_uuid
        self.training_id = training_id
        self.source_type = source_type
        self.model_path = model_path
        self.step_id = step_id
        self.eval_id = eval_id
        self.baseline_id = baseline_id

        self.task_client_id: Optional[str] = None
        self.env_client_id: Optional[str] = None
        self.remote_task_id: Optional[int] = None
        self.remote_env_id: Optional[int] = None
        self.remote_rollout_id: Optional[int] = None
        self.turn_client_ids: Dict[int, str] = {}

    def start_rollout(
        self,
        task_id_str: str,
        task_description: str,
        model_path: str,
        env_type: str,
        source_type: str,
        step_id: Optional[int] = None,
        eval_id: Optional[int] = None,
        baseline_id: Optional[int] = None,
        batch: Optional[int] = None,
        group_num: Optional[int] = None,
        env_index: Optional[int] = None,
        is_eval: bool = False,
        group_id: Optional[int] = None,
        box_type: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> bool:
        self.source_type = source_type
        self.step_id = step_id
        self.eval_id = eval_id
        self.baseline_id = baseline_id
        self.model_path = model_path

        if source_type == "step" and step_id is None:
            logger.warning(
                "[HttpRolloutRecorder] step_id is None; rollout may be filtered by Monitor or not linked to step."
            )

        task_name = task_description or task_id_str
        task_payload = {
            "task_id": task_id_str,
            "name": task_name,
            "description": task_description or task_name,
            "source_type": "gym",
        }
        self.task_client_id = task_id_str
        self.remote_task_id = self.ingest_client.post_task(task_payload, self.task_client_id)
        if self.remote_task_id is None:
            logger.warning("[HttpRolloutRecorder] Failed to create task via ingest.")
            return False

        env_payload = {
            "env_type": env_type,
            "status": "created",
            "box_type": box_type,
            "creation_time": datetime.utcnow().isoformat(),
        }
        self.env_client_id = f"{self.rollout_uuid}:env"
        self.remote_env_id = self.ingest_client.post_environment(env_payload, self.env_client_id)
        if self.remote_env_id is None:
            logger.warning("[HttpRolloutRecorder] Failed to create environment via ingest.")
            return False

        rollout_payload = {
            "rollout_id": self.rollout_uuid,
            "source_type": source_type,
            "step_id": step_id,
            "eval_id": eval_id,
            "baseline_id": baseline_id,
            "env_id": self.remote_env_id,
            "task_id": self.remote_task_id,
            "model_path": model_path,
            "is_eval": is_eval,
            "batch": batch,
            "group_num": group_num,
            "env_index": env_index,
            "group_id": group_id,
            "max_turns": max_turns,
            "status": "running",
            "current_phase": "env_creation",
            "start_time": datetime.utcnow().isoformat(),
        }
        self.remote_rollout_id = self.ingest_client.post_rollout(rollout_payload, self.rollout_uuid)
        if self.remote_rollout_id is None:
            logger.warning(
                "[HttpRolloutRecorder] post_rollout failed; rollout will not appear in Monitor."
            )
        else:
            logger.debug(
                "Monitor: rollout created, rollout_id=%s, group_num=%s, env_index=%s",
                self.rollout_uuid,
                group_num,
                env_index,
            )
        return self.remote_rollout_id is not None

    def update_environment(
        self,
        *,
        gbox_id: Optional[str] = None,
        status: Optional[str] = None,
        status_message: Optional[str] = None,
        error_message: Optional[str] = None,
        termination_time: Optional[str] = None,
        config_json: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Update environment record in monitor (e.g. gbox run_id, termination)."""
        if self.env_client_id is None:
            return
        payload: Dict[str, Any] = {
            "env_type": "android",
            "status": "created",
            "box_type": "android",
            "creation_time": datetime.utcnow().isoformat(),
        }
        if gbox_id is not None:
            payload["gbox_id"] = gbox_id
        if status is not None:
            payload["status"] = status
        if status_message is not None:
            payload["status_message"] = status_message
        if error_message is not None:
            payload["error_message"] = error_message
        if termination_time is not None:
            payload["termination_time"] = termination_time
        if config_json is not None:
            payload["config_json"] = config_json
        payload.update(kwargs)
        self.ingest_client.post_environment(payload, self.env_client_id)

    def update(self, **kwargs: Any) -> None:
        self._post_rollout_update(kwargs)

    def update_status(
        self,
        status: Optional[str] = None,
        current_phase: Optional[str] = None,
        status_message: Optional[str] = None,
        error_message: Optional[str] = None,
        progress_percent: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        payload: Dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if current_phase is not None:
            payload["current_phase"] = current_phase
        if status_message is not None:
            payload["status_message"] = status_message
        if error_message is not None:
            payload["error_message"] = error_message
        if progress_percent is not None:
            payload["progress_percent"] = progress_percent
        payload.update(kwargs)
        self._post_rollout_update(payload)

    def start_turn(self, turn_num: int, start_time: Optional[datetime] = None) -> Optional[int]:
        resolved_start = start_time or datetime.utcnow()
        return self._ensure_turn(turn_num, start_time=resolved_start)

    def end_turn(
        self,
        turn_num: int,
        end_time: Optional[datetime] = None,
        turn_time: Optional[float] = None,
        reward: Optional[float] = None,
        episode_done: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if "metrics" in kwargs and "metrics_json" not in kwargs:
            metrics_val = kwargs.pop("metrics")
            # Serialize to JSON string for DB TEXT column (stage_timings, etc.)
            kwargs["metrics_json"] = (
                json.dumps(metrics_val, default=str)
                if isinstance(metrics_val, dict)
                else metrics_val
            )
        if "model_specific_input" in kwargs and "model_specific_input_json" not in kwargs:
            kwargs["model_specific_input_json"] = kwargs.pop("model_specific_input")
        self._post_turn(
            turn_num,
            {
                "end_time": end_time.isoformat() if isinstance(end_time, datetime) else end_time,
                "turn_time": turn_time,
                "reward": reward,
                "episode_done": episode_done,
                **kwargs,
            },
        )

    def record_action(
        self,
        turn_num: int,
        action_type: Optional[str] = None,
        action: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[int]:
        turn_id = self._ensure_turn(turn_num)
        if turn_id is None:
            return None
        payload: Dict[str, Any] = {"turn_id": turn_id}
        if action_type:
            payload["action_type"] = action_type
        if action and isinstance(action, dict):
            payload.update(
                {
                    "action_name": action.get("name"),
                    "action_args": action.get("args"),
                    "action_call_id": action.get("callId"),
                    "action_json": action,
                }
            )
            if action.get("name") and not action_type:
                payload["action_type"] = action.get("name")
        payload.update(kwargs)
        try:
            return self.ingest_client.post_action(payload)
        except Exception as exc:
            logger.warning(
                "[HttpRolloutRecorder] Failed to post action. "
                f"error={exc}, payload={_sanitize_payload_for_log(payload)}"
            )
            return None

    def record_observation(self, turn_num: int, **kwargs: Any) -> Optional[int]:
        turn_id = self._ensure_turn(turn_num)
        if turn_id is None:
            return None
        if "model_input" in kwargs and "model_input_json" not in kwargs:
            kwargs["model_input_json"] = kwargs.pop("model_input")
        payload = {"turn_id": turn_id}
        payload.update(kwargs)
        self.ingest_client.post_observation(payload)
        return turn_id

    def record_validation(
        self,
        validator_id: Optional[int] = None,
        validation_time: Optional[datetime] = None,
        validation_query: Optional[str] = None,
        expected_result: Optional[str] = None,
        actual_result: Optional[str] = None,
        success: Optional[bool] = None,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None,
        details_json: Optional[Dict[str, Any]] = None,
        screenshot_uri: Optional[str] = None,
    ) -> None:
        if success is None:
            return
        if self.remote_rollout_id is None:
            return
        resolved_time = validation_time or datetime.utcnow()
        payload = {
            "rollout_id": self.remote_rollout_id,
            "validator_id": validator_id,
            "validation_time": resolved_time.isoformat(),
            "validation_query": validation_query,
            "expected_result": expected_result,
            "actual_result": actual_result,
            "success": success,
            "execution_time": execution_time,
            "error_message": error_message,
            "details_json": details_json,
            "screenshot_uri": screenshot_uri,
        }
        self.ingest_client.post_validation(payload)

    def complete_rollout(self, **kwargs: Any) -> None:
        payload = {"status": "completed", "end_time": datetime.utcnow().isoformat(), **kwargs}
        self._post_rollout_update(payload)

    def _post_rollout_update(self, updates: Dict[str, Any]) -> None:
        if self.remote_env_id is None or self.remote_task_id is None:
            return
        payload = {
            "rollout_id": self.rollout_uuid,
            "source_type": self.source_type,
            "step_id": self.step_id,
            "eval_id": self.eval_id,
            "baseline_id": self.baseline_id,
            "env_id": self.remote_env_id,
            "task_id": self.remote_task_id,
            "model_path": self.model_path,
        }
        payload.update(updates)
        self.ingest_client.post_rollout(payload, self.rollout_uuid)

    def _ensure_turn(self, turn_num: int, start_time: Optional[datetime] = None) -> Optional[int]:
        if self.remote_rollout_id is None:
            return None
        client_id = self.turn_client_ids.get(turn_num)
        if client_id:
            return self.ingest_client._map_id("turn", client_id)
        return self._post_turn(turn_num, {"start_time": start_time.isoformat() if start_time else None})

    def _post_turn(self, turn_num: int, updates: Dict[str, Any]) -> Optional[int]:
        if self.remote_rollout_id is None:
            return None
        client_id = self.turn_client_ids.get(turn_num) or f"{self.rollout_uuid}:turn:{turn_num}"
        payload = {
            "rollout_id": self.remote_rollout_id,
            "turn": turn_num,
        }
        payload.update({k: v for k, v in updates.items() if v is not None})
        turn_id = self.ingest_client.post_turn(payload, client_id)
        self.turn_client_ids[turn_num] = client_id
        return turn_id
