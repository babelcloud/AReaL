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
        batch: int | None = None,
        group_size: int | None = None,
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
        self.batch = batch
        self.group_size = group_size or 1
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
        instruction = data.get("instruction") or description
        name = data.get("name", "")
        task_number = data.get("task_number", "")
        tags = data.get("tags", [])
        difficulty = data.get("difficulty", "")
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

        # Log task details for debugging (use print for visibility in worker processes)
        tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)
        print(
            f"\n{'='*60}\n"
            f"CUA Task Started:\n"
            f"  task_number: {task_number or 'N/A'}\n"
            f"  difficulty:  {difficulty or 'N/A'}\n"
            f"  tags:        [{tags_str}]\n"
            f"  instruction: {instruction[:200] + '...' if len(instruction) > 200 else instruction}\n"
            f"  description: {description[:100] + '...' if len(description) > 100 else description}\n"
            f"{'='*60}",
            flush=True,
        )

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
            item_idx = data.get("item_idx", 0) if isinstance(data, dict) else 0
            group_size = getattr(self, "group_size", None) or 1
            group_num = item_idx  # Each batch item is a task, so item_idx is the group number
            env_index_in_group = data.get("env_index_in_group", 0) if isinstance(data, dict) else 0
            env_index = env_index_in_group if group_size > 1 else 0
            batch = getattr(self, "batch", 0)
            logger.info(
                "Rollout starting (task_id=%s, group_num=%s, env_index=%s)",
                task_id, group_num, env_index,
            )
            if not rollout_recorder.start_rollout(
                task_id_str=task_id,
                task_description=instruction or description or name,
                model_path=model_path_str,
                env_type="genv",
                source_type="step",  # Monitor only accepts baseline/eval/step for rollout filter
                step_id=rollout_recorder.step_id,
                eval_id=rollout_recorder.eval_id,
                baseline_id=rollout_recorder.baseline_id,
                batch=batch,
                group_num=group_num,
                env_index=env_index,
                max_turns=self.max_turns,
            ):
                logger.warning("CUA rollout_recorder.start_rollout failed, continuing without monitor")

        try:
            run_result = await run_mini_agent_rollout(
                task=instruction,
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
                step=(self.global_step + 1) if self.global_step is not None else None,
                group_num=group_num,
                task_id=task_id,
                execution_id=ctx.execution_id,
                task_number=getattr(ctx, "task_number", None),
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

        # #region agent log - Hypothesis G: Fix reward calculation based on gym API structure
        # Gym API returns: { success, result: { score, success, validators: [...] } }
        # Each validator has: { id, pass, score, weight, ... }
        # Total reward should be calculated from result.score or weighted sum of validators
        reward = 0.0
        if isinstance(eval_result, dict):
            result_data = eval_result.get("result", {})
            
            # Method 1: Use result.score if available (this is the total weighted score from gym)
            if isinstance(result_data, dict):
                total_score = result_data.get("score")
                if isinstance(total_score, (int, float)):
                    reward = float(total_score)
            
            # Method 2: If no result.score, calculate from validators
            if reward == 0.0 and isinstance(result_data, dict):
                validators = result_data.get("validators", [])
                if isinstance(validators, list) and validators:
                    weighted_sum = 0.0
                    total_weight = 0.0
                    for v in validators:
                        if not isinstance(v, dict):
                            continue
                        v_score = v.get("score", 0.0)
                        v_weight = v.get("weight", 0.0)
                        if isinstance(v_score, (int, float)) and isinstance(v_weight, (int, float)):
                            weighted_sum += float(v_score) * float(v_weight)
                            total_weight += float(v_weight)
                    if total_weight > 0:
                        reward = weighted_sum / total_weight
            
            # Method 3: Fallback to top-level reward/score (legacy)
            if reward == 0.0:
                r = eval_result.get("reward") or eval_result.get("score")
                if isinstance(r, (int, float)):
                    reward = float(r)
            
            # Method 4: Binary reward based on success if still 0
            if reward == 0.0:
                success = eval_result.get("success")
                if success is True:
                    reward = 1.0
        
        with open("/home/zhenwei/.cursor/debug-719a0d.log", "a") as _f:
            import json as _json_debug
            result_data = eval_result.get("result", {}) if isinstance(eval_result, dict) else {}
            validators = result_data.get("validators", []) if isinstance(result_data, dict) else []
            _f.write(_json_debug.dumps({"sessionId":"719a0d","hypothesisId":"G","location":"cua_workflow.py:reward_calc","message":"reward_calculated","data":{"reward":reward,"result_score":result_data.get("score") if isinstance(result_data, dict) else None,"num_validators":len(validators),"validators_scores":[{"id":v.get("id"),"score":v.get("score"),"weight":v.get("weight")} for v in validators[:5]] if validators else []},"timestamp":__import__("time").time()}) + "\n")
        # #endregion

        # #region agent log
        import json as _json_debug
        with open("/home/zhenwei/.cursor/debug-719a0d.log", "a") as _f:
            _f.write(_json_debug.dumps({"sessionId":"719a0d","hypothesisId":"A,B,C,D,E","location":"cua_workflow.py:after_eval","message":"eval_result_and_reward","data":{"eval_result":str(eval_result)[:500],"reward":reward,"rollout_recorder_exists":rollout_recorder is not None,"task_id":task_id,"execution_id":ctx.execution_id},"timestamp":__import__("time").time()}) + "\n")
        # #endregion

        # #region agent log - Record validation with correct format for Monitor
        # Monitor expects details_json to have: genv_evaluation (full gym result) for validators display
        # and evaluation_output/gym_evaluation/genv_evaluation for "evaluate output" button
        if rollout_recorder is not None:
            try:
                overall_success = eval_result.get("success") if isinstance(eval_result, dict) else None
                result_data = eval_result.get("result", {}) if isinstance(eval_result, dict) else {}
                validators = result_data.get("validators", []) if isinstance(result_data, dict) else []
                
                # Build details_json in the format Monitor expects:
                # - genv_evaluation: full gym evaluate result (for validators extraction)
                # - evaluation_output: same as genv_evaluation (for "evaluate output" button)
                details_json = {
                    "genv_evaluation": eval_result,  # Full gym evaluate result
                    "evaluation_output": eval_result,  # For "evaluate output" button
                    "reward": reward,
                    "task_id": task_id,
                    "execution_id": ctx.execution_id,
                }
                
                rollout_recorder.record_validation(
                    success=overall_success,
                    actual_result=f"reward={reward}, validators={len(validators)}",
                    details_json=details_json,
                )
                
                with open("/home/zhenwei/.cursor/debug-719a0d.log", "a") as _f:
                    _f.write(_json_debug.dumps({"sessionId":"719a0d","hypothesisId":"I","location":"cua_workflow.py:record_validation","message":"validation_recorded_correct_format","data":{"success":overall_success,"reward":reward,"num_validators":len(validators),"details_json_keys":list(details_json.keys())},"timestamp":__import__("time").time()}) + "\n")
            except Exception as _e:
                with open("/home/zhenwei/.cursor/debug-719a0d.log", "a") as _f:
                    _f.write(_json_debug.dumps({"sessionId":"719a0d","hypothesisId":"I","location":"cua_workflow.py:record_validation","message":"record_validation_failed","data":{"error":str(_e)},"timestamp":__import__("time").time()}) + "\n")
        # #endregion

        try:
            await terminate_execution(
                base_url=gym_base_url,
                gym_id=gym_id,
                execution_id=ctx.execution_id,
            )
        except Exception as e:
            logger.debug("CUA terminate_execution: %s", e)

        # #region agent log - Hypothesis H: complete_rollout needs summary_json for evaluate output
        if rollout_recorder is not None:
            try:
                import json as _json_summary
                # Build summary_json with full evaluate result for Monitor's "evaluate output" display
                summary_data = {
                    "evaluate_output": eval_result if isinstance(eval_result, dict) else None,
                    "reward": reward,
                    "task_success": eval_result.get("success") if isinstance(eval_result, dict) else None,
                    "num_turns": run_result.get("num_turns") if isinstance(run_result, dict) else None,
                    "execution_id": ctx.execution_id,
                    "task_id": task_id,
                }
                rollout_recorder.complete_rollout(
                    reward=reward,
                    task_success=eval_result.get("success") if isinstance(eval_result, dict) else None,
                    validation_passed=eval_result.get("success") if isinstance(eval_result, dict) else None,
                    num_turns=run_result.get("num_turns") if isinstance(run_result, dict) else None,
                    summary_json=_json_summary.dumps(summary_data, default=str),
                )
                with open("/home/zhenwei/.cursor/debug-719a0d.log", "a") as _f:
                    _f.write(_json_debug.dumps({"sessionId":"719a0d","hypothesisId":"H","location":"cua_workflow.py:complete_rollout","message":"complete_rollout_with_summary","data":{"reward":reward,"has_summary":True,"eval_result_keys":list(eval_result.keys()) if isinstance(eval_result, dict) else None},"timestamp":__import__("time").time()}) + "\n")
            except Exception as _e:
                with open("/home/zhenwei/.cursor/debug-719a0d.log", "a") as _f:
                    _f.write(_json_debug.dumps({"sessionId":"719a0d","hypothesisId":"H","location":"cua_workflow.py:complete_rollout","message":"complete_rollout_failed","data":{"error":str(_e)},"timestamp":__import__("time").time()}) + "\n")
        # #endregion

        traj = _events_to_trajectory_tensors(events, self.tokenizer, reward)
        return traj
