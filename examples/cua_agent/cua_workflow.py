"""
CUAAgentWorkflow: one episode = genv execution + gbox-mini-agent run + trajectory for PPO.
Path A: proxy + existing gbox-mini-agent (model.baseUrl points to AReaL vLLM/proxy).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import name_resolve, names

import httpx

from cua_rl.core.genv_execution import (
    evaluate_execution,
    start_genv_execution,
    terminate_execution,
)
from cua_rl.core.mini_agent_runner import run_mini_agent_rollout
from cua_rl.core.rollout_logger import RolloutLogger
from cua_rl.trajectory_utils import (
    extract_model_outputs_from_events,
    extract_tokens_and_logprobs,
)

logger = logging.getLogger(__name__)

# Cache model id from GET /v1/models per base_url to avoid repeated calls
_v1_models_id_cache: dict[str, str] = {}


async def _resolve_model_id_from_api(base_url: str, timeout: float = 5.0) -> str | None:
    """Fetch first model id from OpenAI-compatible GET /v1/models. Cached per base_url."""
    if not base_url:
        return None
    key = base_url.rstrip("/")
    if key in _v1_models_id_cache:
        return _v1_models_id_cache[key]
    try:
        url = f"{key}/v1/models" if "/v1" not in key else f"{key}/models"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        models = data.get("data") if isinstance(data, dict) else None
        if isinstance(models, list) and models:
            model_id = models[0].get("id") if isinstance(models[0], dict) else None
            if model_id:
                _v1_models_id_cache[key] = model_id
                return model_id
    except Exception as e:
        logger.debug("CUA could not resolve model id from %s/v1/models: %s", base_url, e)
    return None


def _make_env_with_task(task_id: str) -> Any:
    """Minimal env-like object so start_genv_execution can resolve task_identifier."""
    class Task:
        def __init__(self, genv_identifier: str):
            self.genv_identifier = genv_identifier
            self.id = genv_identifier
            self.name = genv_identifier

    class Env:
        pass

    env = Env()
    env.task = Task(task_id)
    return env


def _events_to_trajectory_tensors(
    events: list[dict],
    tokenizer: PreTrainedTokenizerFast,
    reward: float,
) -> dict[str, torch.Tensor] | None:
    """Convert gbox events to AReaL trajectory dict (input_ids, loss_mask, logprobs, versions, rewards)."""
    outputs_by_turn = extract_model_outputs_from_events(events)
    if not outputs_by_turn:
        return None

    all_input_ids: list[int] = []
    all_logprobs: list[float] = []
    all_loss_mask: list[int] = []

    bos_id = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "pad_token_id", 0)
    if bos_id is None:
        bos_id = 0

    for turn in sorted(outputs_by_turn.keys()):
        model_output = outputs_by_turn[turn]
        pair = extract_tokens_and_logprobs(model_output)
        if pair is not None:
            token_ids, logprobs = pair
        else:
            content = ""
            if isinstance(model_output.get("choices"), list) and model_output["choices"]:
                msg = (model_output["choices"][0] or {}).get("message") or {}
                content = msg.get("content") or ""
            token_ids = tokenizer.encode(content, add_special_tokens=False)
            logprobs = [0.0] * len(token_ids)

        if not token_ids:
            continue
        all_input_ids.extend(token_ids)
        all_logprobs.extend(logprobs)
        all_loss_mask.extend([1] * len(token_ids))

    if not all_input_ids:
        return None

    all_input_ids = [bos_id] + all_input_ids
    all_logprobs = [0.0] + all_logprobs
    all_loss_mask = [0] + all_loss_mask

    return {
        "input_ids": torch.tensor([all_input_ids], dtype=torch.int32),
        "loss_mask": torch.tensor([all_loss_mask], dtype=torch.int32),
        "logprobs": torch.tensor([all_logprobs], dtype=torch.float32),
        "versions": torch.full((1, len(all_input_ids)), -1, dtype=torch.int32),
        "attention_mask": torch.ones(1, len(all_input_ids), dtype=torch.bool),
        "rewards": torch.tensor([reward], dtype=torch.float32),
    }


class CUAAgentWorkflow(RolloutWorkflow):
    """Rollout workflow: one task -> genv execution + gbox-mini-agent run -> trajectory for PPO."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        gbox_mini_agent_base_url: str,
        gbox_mini_agent_agent: str,
        gbox_mini_agent_env: dict[str, Any] | None,
        model_base_url: str,
        model_name: str,
        resolved_inference_url: str | None = None,
        max_turns: int = 30,
        max_task_time_seconds: int | None = None,
        max_turn_time_seconds: int | None = None,
        standard_action_space: str | None = "mobile",
        rollout_recorder: Any = None,
        rollout_recorder_factory: Any = None,
        monitor_ingest_config: dict[str, Any] | None = None,
        step_id: int | None = None,
        global_step: int | None = None,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        gym_task_list: list[dict] | None = None,
    ):
        self.tokenizer = tokenizer
        self.gbox_mini_agent_base_url = gbox_mini_agent_base_url
        self.gbox_mini_agent_agent = gbox_mini_agent_agent
        self.gbox_mini_agent_env = gbox_mini_agent_env
        self.model_base_url = model_base_url
        self.model_name = model_name
        self.resolved_inference_url = resolved_inference_url
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.max_turns = max_turns
        self.max_task_time_seconds = max_task_time_seconds
        self.max_turn_time_seconds = max_turn_time_seconds
        self.standard_action_space = standard_action_space
        self.rollout_recorder = rollout_recorder
        self.rollout_recorder_factory = rollout_recorder_factory
        self.monitor_ingest_config = monitor_ingest_config or {}
        self.step_id = step_id
        self.global_step = global_step
        self.gym_task_list = gym_task_list

    def _resolve_model_base_url(self) -> str:
        """Return model base URL. Prefer resolved_inference_url (set on main process) so workers get correct vLLM URL; else name_resolve when experiment/trial set; else model_base_url."""
        if getattr(self, "resolved_inference_url", None):
            return self.resolved_inference_url.rstrip("/")
        if self.experiment_name and self.trial_name:
            try:
                name = names.gen_servers(self.experiment_name, self.trial_name)
                addrs = name_resolve.get_subtree(name)
                if addrs:
                    addr = addrs[0]
                    base = f"http://{addr}".rstrip("/")
                    url = f"{base}/v1" if "/v1" not in base else base
                    logger.info("CUA resolved AReaL vLLM base URL (for mini-agent): %s", url)
                    return url
            except Exception as e:
                logger.warning("CUA could not resolve AReaL vLLM URL from name_resolve: %s", e)
        if self.model_base_url:
            return self.model_base_url.rstrip("/")
        return ""

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        task_id = data.get("task_id") or data.get("id", "")
        description = data.get("description", "")
        name = data.get("name", "")
        gym_base_url = data.get("gym_base_url", "")
        gym_id = data.get("gym_id", "")

        if not task_id or not gym_id:
            logger.warning("CUA episode skipped: missing task_id or gym_id")
            return None

        env = _make_env_with_task(task_id)
        rollout_id = uuid.uuid4().hex
        rollout_logger = RolloutLogger(rollout_id)
        rollout_recorder = self.rollout_recorder
        if self.rollout_recorder_factory is not None:
            rollout_recorder = self.rollout_recorder_factory(rollout_id)
        elif self.monitor_ingest_config.get("base_url") and self.monitor_ingest_config.get("project_token"):
            try:
                from cua_rl.database.ingest_client import IngestClient
                from cua_rl.database.http_rollout_recorder import HttpRolloutRecorder
                resolved_inference_url = self._resolve_model_base_url()
                model_path_for_monitor = self.model_name or resolved_inference_url or self.model_base_url or ""
                base_url = self.monitor_ingest_config["base_url"]
                project_token = self.monitor_ingest_config["project_token"]
                training_id = self.monitor_ingest_config.get("training_id")
                ingest_client = IngestClient(base_url, project_token)
                rollout_recorder = HttpRolloutRecorder(
                    ingest_client,
                    rollout_id,
                    training_id,
                    "step",  # Monitor only accepts baseline/eval/step
                    model_path=model_path_for_monitor,
                    step_id=self.step_id,
                    eval_id=None,
                    baseline_id=None,
                )
            except Exception as e:
                logger.warning("CUA monitor rollout recorder on worker failed: %s", e)
                rollout_recorder = None

        try:
            ctx = await start_genv_execution(
                env=env,
                gym_base_url=gym_base_url,
                gym_id=gym_id,
                preloaded_tasks=getattr(self, "gym_task_list", None),
            )
        except Exception as e:
            logger.warning("CUA start_genv_execution failed: %s", e)
            return None

        gbox_env = ctx.agent_env_payload()
        resolved_base_url = self._resolve_model_base_url()
        model_payload: dict[str, Any] = {}
        # Always send baseUrl when we have one so gbox-mini-agent uses AReaL vLLM, not (default)
        base_url = resolved_base_url or (self.model_base_url.rstrip("/") if self.model_base_url else "")
        if base_url:
            model_payload["baseUrl"] = base_url
        # Use model id from vLLM GET /v1/models so request matches server (avoids "invalid model ID")
        model_name_for_payload = self.model_name
        if base_url:
            resolved_id = await _resolve_model_id_from_api(base_url)
            if resolved_id:
                model_name_for_payload = resolved_id
        if model_name_for_payload:
            model_payload["name"] = model_name_for_payload
        model_payload["provider"] = "openai"

        if rollout_recorder is not None:
            resolved_base_url = self._resolve_model_base_url()
            model_path_str = self.model_name or resolved_base_url or self.model_base_url or ""
            if not rollout_recorder.start_rollout(
                task_id_str=task_id,
                task_description=description or name,
                model_path=model_path_str,
                env_type="genv",
                source_type="step",  # Monitor only accepts baseline/eval/step for rollout filter
                step_id=rollout_recorder.step_id,
                eval_id=rollout_recorder.eval_id,
                baseline_id=rollout_recorder.baseline_id,
                max_turns=self.max_turns,
            ):
                logger.warning("CUA rollout_recorder.start_rollout failed, continuing without monitor")

        try:
            run_result = await run_mini_agent_rollout(
                task=description,
                rollout_logger=rollout_logger,
                rollout_recorder=rollout_recorder,
                max_turns=self.max_turns,
                gbox_mini_agent_base_url=self.gbox_mini_agent_base_url,
                gbox_mini_agent_agent=self.gbox_mini_agent_agent,
                gbox_mini_agent_model=model_payload or None,
                gbox_mini_agent_env=gbox_env,
                gbox_mini_agent_standard_action_space=self.standard_action_space,
                max_task_time_seconds=self.max_task_time_seconds,
                max_turn_time_seconds=self.max_turn_time_seconds,
            )
        except Exception as e:
            logger.warning("CUA run_mini_agent_rollout failed: %s", e)
            try:
                await terminate_execution(
                    base_url=gym_base_url,
                    gym_id=gym_id,
                    execution_id=ctx.execution_id,
                )
            except Exception:
                pass
            return None

        events = rollout_logger.trajectory_data.get("events") or []
        execution_data = run_result if isinstance(run_result, dict) else {}

        try:
            eval_result = await evaluate_execution(
                base_url=gym_base_url,
                gym_id=gym_id,
                execution_id=ctx.execution_id,
                execution_data=execution_data,
            )
        except Exception as e:
            logger.warning("CUA evaluate_execution failed: %s", e)
            eval_result = {}

        reward = 0.0
        if isinstance(eval_result, dict):
            r = eval_result.get("reward") or eval_result.get("score")
            if isinstance(r, (int, float)):
                reward = float(r)
            success = eval_result.get("success")
            if success is True and reward == 0.0:
                reward = 1.0
            elif success is False and reward == 0.0:
                reward = 0.0

        try:
            await terminate_execution(
                base_url=gym_base_url,
                gym_id=gym_id,
                execution_id=ctx.execution_id,
            )
        except Exception as e:
            logger.debug("CUA terminate_execution: %s", e)

        traj = _events_to_trajectory_tensors(events, self.tokenizer, reward)
        return traj
