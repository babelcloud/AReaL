"""
HTTP ingest client for training-monitor.

Uses token-based auth and maps local DB ids to remote ids.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
_BASE64_IMAGE_RE = re.compile(r"data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")


def _serialize_dt(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _sanitize_payload_for_log(payload: Dict[str, Any]) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=True, default=str)
    except Exception:
        return "<unserializable payload>"
    text = _BASE64_IMAGE_RE.sub("[base64 image omitted]", text)
    if len(text) > 2000:
        return f"{text[:2000]}...<truncated>"
    return text


def _extract_first_mapped_id(response: Dict[str, Any], key: str) -> Optional[int]:
    items = response.get(key)
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    if not isinstance(first, dict):
        return None
    try:
        return int(first.get("id")) if first.get("id") is not None else None
    except Exception:
        return None


class IngestClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.id_map: Dict[str, Dict[str, int]] = {
            "training": {},
            "task": {},
            "validator": {},
            "environment": {},
            "step": {},
            "baseline": {},
            "eval": {},
            "rollout": {},
            "turn": {},
            "action": {},
            "observation": {},
            "validation": {},
        }

    def _map_id(self, kind: str, local_id: Optional[Any]) -> Optional[int]:
        if local_id is None:
            return None
        return self.id_map.get(kind, {}).get(str(local_id))

    def _update_map(self, kind: str, items: list[dict[str, Any]]) -> None:
        mapping = self.id_map.setdefault(kind, {})
        for item in items:
            try:
                local_id = str(item["client_id"])
                remote_id = int(item["id"])
            except Exception:
                continue
            mapping[local_id] = remote_id

    def _post_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/api/ingest/batch"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        retries = 5
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                return json.loads(body)
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                logger.error(f"HTTP {e.code} error from {url}: {error_body}")
                raise
            except (socket.timeout, urllib.error.URLError) as e:
                logger.error(f"Failed to post batch to {url}: {e}")
                if attempt < retries:
                    time.sleep(1.0 * attempt)
                    continue
                raise
            except Exception as e:
                logger.error(f"Failed to post batch to {url}: {e}")
                raise

    def record_training(self, training: Any) -> None:
        payload = {
            "training": {
                "run_name": training.run_name,
                "log_path": training.log_path,
                "model_name": training.model_name,
                "lora_rank": training.lora_rank,
                "learning_rate": training.learning_rate,
                "batch_size": training.batch_size,
                "group_size": training.group_size,
                "groups_per_batch": training.groups_per_batch,
                "max_tokens": training.max_tokens,
                "temperature": training.temperature,
                "kl_penalty_coef": training.kl_penalty_coef,
                "num_substeps": training.num_substeps,
                "max_turns": training.max_turns,
                "seed": training.seed,
                "box_type": training.box_type,
                "renderer_name": training.renderer_name,
                "wandb_project": training.wandb_project,
                "wandb_name": training.wandb_name,
                "status": training.status,
                "progress_percent": training.progress_percent,
                "current_step": training.current_step,
                "total_steps": training.total_steps,
                "current_phase": training.current_phase,
                "status_message": training.status_message,
                "error_message": training.error_message,
                "start_time": _serialize_dt(training.start_time),
                "end_time": _serialize_dt(training.end_time),
                "last_heartbeat": _serialize_dt(training.last_heartbeat),
                "config_json": training.config_json,
                "avg_turn_time": training.avg_turn_time,
                "estimated_total_time": training.estimated_total_time,
                "estimated_remaining_time": training.estimated_remaining_time,
                "created_at": _serialize_dt(training.created_at),
                "updated_at": _serialize_dt(training.updated_at),
            }
        }
        response = self._post_batch(payload)
        if "training_id" in response:
            self.id_map["training"][training.id] = int(response["training_id"])

    def post_training(self, data: Dict[str, Any]) -> Optional[int]:
        """Post a training payload directly without local DB objects."""
        payload = {"training": data}
        response = self._post_batch(payload)
        training_id = response.get("training_id")
        return int(training_id) if training_id is not None else None

    def _post_single_with_map(
        self, kind: str, response_key: str, payload_key: str, data: Dict[str, Any], client_id: str
    ) -> Optional[int]:
        payload = {payload_key: [{**data, "client_id": client_id}]}
        response = self._post_batch(payload)
        self._update_map(kind, response.get(response_key, []))
        mapped = self._map_id(kind, client_id)
        if mapped is None:
            logger.warning(
                f"[IngestClient] Missing {kind} mapping after ingest. "
                f"response_key={response_key}, payload={_sanitize_payload_for_log(payload)}"
            )
        return mapped

    def post_task(self, data: Dict[str, Any], client_id: str) -> Optional[int]:
        return self._post_single_with_map("task", "tasks", "tasks", data, client_id)

    def post_environment(self, data: Dict[str, Any], client_id: str) -> Optional[int]:
        return self._post_single_with_map("environment", "environments", "environments", data, client_id)

    def post_step(self, data: Dict[str, Any], client_id: str) -> Optional[int]:
        return self._post_single_with_map("step", "steps", "steps", data, client_id)

    def post_baseline(self, data: Dict[str, Any], client_id: str) -> Optional[int]:
        return self._post_single_with_map("baseline", "baselines", "baselines", data, client_id)

    def post_eval(self, data: Dict[str, Any], client_id: str) -> Optional[int]:
        return self._post_single_with_map("eval", "evals", "evals", data, client_id)

    def post_rollout(self, data: Dict[str, Any], client_id: str) -> Optional[int]:
        return self._post_single_with_map("rollout", "rollouts", "rollouts", data, client_id)

    def post_turn(self, data: Dict[str, Any], client_id: str) -> Optional[int]:
        return self._post_single_with_map("turn", "turns", "turns", data, client_id)

    def post_action(self, data: Dict[str, Any]) -> Optional[int]:
        payload = {"actions": [data]}
        response = self._post_batch(payload)
        action_id = _extract_first_mapped_id(response, "actions")
        if action_id is None:
            warning = (
                "[IngestClient] Missing action id after ingest. "
                f"payload={_sanitize_payload_for_log(payload)} "
                f"response={_sanitize_payload_for_log(response)}"
            )
            logger.warning(warning)
            print(warning)
        return action_id

    def post_observation(self, data: Dict[str, Any]) -> Optional[int]:
        payload = {"observations": [data]}
        response = self._post_batch(payload)
        obs_id = _extract_first_mapped_id(response, "observations")
        if obs_id is None:
            logger.warning(
                "[IngestClient] Missing observation id after ingest. "
                f"payload={_sanitize_payload_for_log(payload)}"
            )
        return obs_id

    def post_validation(self, data: Dict[str, Any]) -> Optional[int]:
        payload = {"validations": [data]}
        response = self._post_batch(payload)
        validation_id = _extract_first_mapped_id(response, "validations")
        if validation_id is None:
            logger.warning(
                "[IngestClient] Missing validation id after ingest. "
                f"payload={_sanitize_payload_for_log(payload)}"
            )
        return validation_id

    def record_task(self, task: Any) -> None:
        payload = {
            "tasks": [
                {
                    "client_id": str(task.id),
                    "task_id": task.task_id,
                    "name": task.name,
                    "instruction": task.instruction,
                    "description": task.description,
                    "difficulty": task.difficulty,
                    "category": task.category,
                    "max_steps": task.max_steps,
                    "validation_type": task.validation_type,
                    "validation_query": task.validation_query,
                    "expected_result": task.expected_result,
                    "tags": task.tags,
                    "prerequisites": task.prerequisites,
                    "app_name": task.app_name,
                    "source_type": task.source_type,
                    "created_at": _serialize_dt(task.created_at),
                    "updated_at": _serialize_dt(task.updated_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("task", response.get("tasks", []))

    def record_validator(self, validator: Any) -> None:
        remote_task_id = self._map_id("task", validator.task_id)
        if remote_task_id is None:
            logger.warning("Ingest validator skipped: task mapping missing")
            return
        payload = {
            "validators": [
                {
                    "client_id": str(validator.id),
                    "task_id": remote_task_id,
                    "validator_type": validator.validator_type,
                    "validation_query": validator.validation_query,
                    "validation_method": validator.validation_method,
                    "config_json": validator.config_json,
                    "created_at": _serialize_dt(validator.created_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("validator", response.get("validators", []))

    def record_environment(self, env: Any) -> None:
        payload = {
            "environments": [
                {
                    "client_id": str(env.id),
                    "env_type": env.env_type,
                    "status": env.status,
                    "gbox_id": env.gbox_id,
                    "box_type": env.box_type,
                    "creation_time": _serialize_dt(env.creation_time),
                    "termination_time": _serialize_dt(env.termination_time),
                    "status_message": env.status_message,
                    "error_message": env.error_message,
                    "config_json": env.config_json,
                    "created_at": _serialize_dt(env.created_at),
                    "updated_at": _serialize_dt(env.updated_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("environment", response.get("environments", []))

    def record_step(self, step: Any) -> None:
        remote_training_id = self._map_id("training", step.training_id)
        if remote_training_id is None:
            logger.warning("Ingest step skipped: training mapping missing")
            return
        payload = {
            "steps": [
                {
                    "client_id": str(step.id),
                    "training_id": remote_training_id,
                    "step": step.step,
                    "batch": step.batch,
                    "status": step.status,
                    "progress_percent": step.progress_percent,
                    "current_phase": step.current_phase,
                    "rollout_progress": step.rollout_progress,
                    "training_progress": step.training_progress,
                    "status_message": step.status_message,
                    "error_message": step.error_message,
                    "start_time": _serialize_dt(step.start_time),
                    "end_time": _serialize_dt(step.end_time),
                    "rollout_start_time": _serialize_dt(step.rollout_start_time),
                    "rollout_end_time": _serialize_dt(step.rollout_end_time),
                    "training_start_time": _serialize_dt(step.training_start_time),
                    "training_end_time": _serialize_dt(step.training_end_time),
                    "learning_rate": step.learning_rate,
                    "model_path": step.model_path,
                    "checkpoint_path": step.checkpoint_path,
                    "loss": step.loss,
                    "kl_divergence": step.kl_divergence,
                    "policy_gradient_norm": step.policy_gradient_norm,
                    "reward_mean": step.reward_mean,
                    "reward_std": step.reward_std,
                    "num_trajectories": step.num_trajectories,
                    "num_tokens": step.num_tokens,
                    "metrics_json": step.metrics_json,
                    "avg_turn_time": step.avg_turn_time,
                    "estimated_total_time": step.estimated_total_time,
                    "estimated_remaining_time": step.estimated_remaining_time,
                    "created_at": _serialize_dt(step.created_at),
                    "updated_at": _serialize_dt(step.updated_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("step", response.get("steps", []))

    def record_baseline(self, baseline: Any) -> None:
        remote_training_id = self._map_id("training", baseline.training_id)
        if remote_training_id is None:
            logger.warning("Ingest baseline skipped: training mapping missing")
            return
        payload = {
            "baselines": [
                {
                    "client_id": str(baseline.id),
                    "training_id": remote_training_id,
                    "model_path": baseline.model_path,
                    "status": baseline.status,
                    "progress_percent": baseline.progress_percent,
                    "current_task_index": baseline.current_task_index,
                    "total_tasks": baseline.total_tasks,
                    "completed_tasks": baseline.completed_tasks,
                    "current_phase": baseline.current_phase,
                    "status_message": baseline.status_message,
                    "error_message": baseline.error_message,
                    "start_time": _serialize_dt(baseline.start_time),
                    "end_time": _serialize_dt(baseline.end_time),
                    "eval_time": _serialize_dt(baseline.eval_time),
                    "success_rate": baseline.success_rate,
                    "avg_reward": baseline.avg_reward,
                    "avg_turns": baseline.avg_turns,
                    "successful_tasks": baseline.successful_tasks,
                    "metrics_json": baseline.metrics_json,
                    "avg_turn_time": baseline.avg_turn_time,
                    "estimated_total_time": baseline.estimated_total_time,
                    "estimated_remaining_time": baseline.estimated_remaining_time,
                    "created_at": _serialize_dt(baseline.created_at),
                    "updated_at": _serialize_dt(baseline.updated_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("baseline", response.get("baselines", []))

    def record_eval(self, eval_obj: Any) -> None:
        remote_training_id = self._map_id("training", eval_obj.training_id)
        if remote_training_id is None:
            logger.warning("Ingest eval skipped: training mapping missing")
            return
        payload = {
            "evals": [
                {
                    "client_id": str(eval_obj.id),
                    "training_id": remote_training_id,
                    "step": eval_obj.step,
                    "model_path": eval_obj.model_path,
                    "status": eval_obj.status,
                    "progress_percent": eval_obj.progress_percent,
                    "current_task_index": eval_obj.current_task_index,
                    "total_tasks": eval_obj.total_tasks,
                    "completed_tasks": eval_obj.completed_tasks,
                    "current_phase": eval_obj.current_phase,
                    "status_message": eval_obj.status_message,
                    "error_message": eval_obj.error_message,
                    "start_time": _serialize_dt(eval_obj.start_time),
                    "end_time": _serialize_dt(eval_obj.end_time),
                    "eval_time": _serialize_dt(eval_obj.eval_time),
                    "success_rate": eval_obj.success_rate,
                    "avg_reward": eval_obj.avg_reward,
                    "avg_turns": eval_obj.avg_turns,
                    "successful_tasks": eval_obj.successful_tasks,
                    "metrics_json": eval_obj.metrics_json,
                    "avg_turn_time": eval_obj.avg_turn_time,
                    "estimated_total_time": eval_obj.estimated_total_time,
                    "estimated_remaining_time": eval_obj.estimated_remaining_time,
                    "created_at": _serialize_dt(eval_obj.created_at),
                    "updated_at": _serialize_dt(eval_obj.updated_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("eval", response.get("evals", []))

    def record_rollout(self, rollout: Any) -> None:
        remote_env_id = self._map_id("environment", rollout.env_id)
        remote_task_id = self._map_id("task", rollout.task_id)
        remote_step_id = self._map_id("step", rollout.step_id)
        remote_eval_id = self._map_id("eval", rollout.eval_id)
        remote_baseline_id = self._map_id("baseline", rollout.baseline_id)

        if remote_env_id is None or remote_task_id is None:
            logger.warning("Ingest rollout skipped: env/task mapping missing")
            return
        if rollout.source_type == "baseline" and remote_baseline_id is None:
            logger.warning("Ingest rollout skipped: baseline mapping missing")
            return
        if rollout.source_type == "eval" and remote_eval_id is None:
            logger.warning("Ingest rollout skipped: eval mapping missing")
            return
        if rollout.source_type == "step" and remote_step_id is None:
            logger.warning("Ingest rollout skipped: step mapping missing")
            return
        if rollout.source_type not in {"baseline", "eval", "step"}:
            logger.warning("Ingest rollout skipped: unknown source_type")
            return

        payload = {
            "rollouts": [
                {
                    "client_id": str(rollout.id),
                    "source_type": rollout.source_type,
                    "step_id": remote_step_id,
                    "eval_id": remote_eval_id,
                    "baseline_id": remote_baseline_id,
                    "env_id": remote_env_id,
                    "rollout_id": rollout.rollout_id,
                    "batch": rollout.batch,
                    "group_num": rollout.group_num,
                    "env_index": rollout.env_index,
                    "task_id": remote_task_id,
                    "model_path": rollout.model_path,
                    "is_eval": rollout.is_eval,
                    "status": rollout.status,
                    "progress_percent": rollout.progress_percent,
                    "current_phase": rollout.current_phase,
                    "current_turn": rollout.current_turn,
                    "status_message": rollout.status_message,
                    "error_message": rollout.error_message,
                    "start_time": _serialize_dt(rollout.start_time),
                    "end_time": _serialize_dt(rollout.end_time),
                    "env_creation_time": _serialize_dt(rollout.env_creation_time),
                    "agent_init_time": _serialize_dt(rollout.agent_init_time),
                    "task_start_time": _serialize_dt(rollout.task_start_time),
                    "task_end_time": _serialize_dt(rollout.task_end_time),
                    "validation_time": _serialize_dt(rollout.validation_time),
                    "rollout_time": rollout.rollout_time,
                    "task_completed": rollout.task_completed,
                    "task_success": rollout.task_success,
                    "agent_reported_success": rollout.agent_reported_success,
                    "validation_passed": rollout.validation_passed,
                    "num_turns": rollout.num_turns,
                    "max_turns": rollout.max_turns,
                    "reward": rollout.reward,
                    "temperature": rollout.temperature,
                    "num_total_actions": rollout.num_total_actions,
                    "consecutive_repeated_actions": rollout.consecutive_repeated_actions,
                    "parse_errors": rollout.parse_errors,
                    "tool_name_errors": rollout.tool_name_errors,
                    "tool_arg_errors": rollout.tool_arg_errors,
                    "runtime_errors": rollout.runtime_errors,
                    "ran_out_of_turns": rollout.ran_out_of_turns,
                    "attempted_completion": rollout.attempted_completion,
                    "turn_first_success": rollout.turn_first_success,
                    "turn_task_completed": rollout.turn_task_completed,
                    "errors": rollout.errors,
                    "summary_json": rollout.summary_json,
                    "trajectory_path": rollout.trajectory_path,
                    "trajectory_data_json": rollout.trajectory_data_json,
                    "created_at": _serialize_dt(rollout.created_at),
                    "updated_at": _serialize_dt(rollout.updated_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("rollout", response.get("rollouts", []))

    def record_turn(self, turn: Any) -> None:
        remote_rollout_id = self._map_id("rollout", turn.rollout_id)
        if remote_rollout_id is None:
            logger.warning("Ingest turn skipped: rollout mapping missing")
            return
        payload = {
            "turns": [
                {
                    "client_id": str(turn.id),
                    "rollout_id": remote_rollout_id,
                    "turn": turn.turn,
                    "start_time": _serialize_dt(turn.start_time),
                    "end_time": _serialize_dt(turn.end_time),
                    "turn_time": turn.turn_time,
                    "reward": turn.reward,
                    "episode_done": turn.episode_done,
                    "metrics_json": turn.metrics_json,
                    "model_response": turn.model_response,
                    "screen_before": turn.screen_before,
                    "screen_after": turn.screen_after,
                    "thinking": turn.thinking,
                    "text": turn.text,
                    "action_json": turn.action_json,
                    "model_specific_input_json": turn.model_specific_input_json,
                    "created_at": _serialize_dt(turn.created_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("turn", response.get("turns", []))

    def record_action(self, action: Any) -> None:
        remote_turn_id = self._map_id("turn", action.turn_id)
        if remote_turn_id is None:
            logger.warning("Ingest action skipped: turn mapping missing")
            return
        payload = {
            "actions": [
                {
                    "client_id": str(action.id),
                    "turn_id": remote_turn_id,
                    "action_type": action.action_type,
                    "tool_name": action.tool_name,
                    "tool_args": action.tool_args,
                    "action_name": action.action_name,
                    "action_args": action.action_args,
                    "action_call_id": action.action_call_id,
                    "action_json": action.action_json,
                    "tokens": action.tokens,
                    "logprobs": action.logprobs,
                    "num_tokens": action.num_tokens,
                    "created_at": _serialize_dt(action.created_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("action", response.get("actions", []))

    def record_observation(self, obs: Any) -> None:
        remote_turn_id = self._map_id("turn", obs.turn_id)
        if remote_turn_id is None:
            logger.warning("Ingest observation skipped: turn mapping missing")
            return
        payload = {
            "observations": [
                {
                    "client_id": str(obs.id),
                    "turn_id": remote_turn_id,
                    "obs_type": obs.obs_type,
                    "screenshot_uri": obs.screenshot_uri,
                    "text_content": obs.text_content,
                    "model_input_json": obs.model_input_json,
                    "created_at": _serialize_dt(obs.created_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("observation", response.get("observations", []))

    def record_validation(self, validation: Any) -> None:
        remote_rollout_id = self._map_id("rollout", validation.rollout_id)
        if remote_rollout_id is None:
            logger.warning("Ingest validation skipped: rollout mapping missing")
            return
        remote_validator_id = self._map_id("validator", validation.validator_id)
        payload = {
            "validations": [
                {
                    "client_id": str(validation.id),
                    "rollout_id": remote_rollout_id,
                    "validator_id": remote_validator_id,
                    "validation_time": _serialize_dt(validation.validation_time),
                    "validation_query": validation.validation_query,
                    "expected_result": validation.expected_result,
                    "actual_result": validation.actual_result,
                    "success": validation.success,
                    "execution_time": validation.execution_time,
                    "error_message": validation.error_message,
                    "details_json": validation.details_json,
                    "created_at": _serialize_dt(validation.created_at),
                }
            ]
        }
        response = self._post_batch(payload)
        self._update_map("validation", response.get("validations", []))
