"""
CUA Agent PPO training entry point.

- Loads CUAConfig (gym, gbox-mini-agent, monitor).
- Builds task dataset from gym; PPOTrainer + CUAAgentWorkflow.
- Optional Training Monitor lifecycle (post_training, post_step).
- vLLM/proxy URL passed to workflow so gbox uses AReaL-managed inference.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys

# Add examples/cua_agent so that "cua_rl" and "examples.cua_agent" resolve
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from configs import CUAConfig  # noqa: E402

from areal import PPOTrainer  # noqa: E402
from areal.api.cli_args import load_expr_config  # noqa: E402
from areal.utils.hf_utils import load_hf_tokenizer  # noqa: E402

from dataset import get_cua_train_and_valid_datasets  # noqa: E402

logger = logging.getLogger(__name__)


def main(args: list[str] | None = None) -> None:
    argv = args if args is not None else sys.argv[1:]
    config, _ = load_expr_config(argv, CUAConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Optional: init Training Monitor
    monitor_base_url = config.monitor_base_url or os.environ.get("MONITOR_BASE_URL", "")
    project_token = config.project_token or os.environ.get("MONITOR_PROJECT_TOKEN", "")
    training_id = None
    if monitor_base_url and project_token:
        try:
            from cua_rl.database.ingest_client import IngestClient
            from cua_rl.database.monitor_context import set_ingest_client, set_training_id

            ingest = IngestClient(monitor_base_url, project_token)
            set_ingest_client(ingest)
            exp = getattr(config, "experiment_name", "cua_agent")
            trial = getattr(config, "trial_name", "run")
            training_id = ingest.post_training({
                "run_name": f"{exp}_{trial}",
                "status": "running",
                "current_phase": "rollout",
                "experiment_name": exp,
                "trial_name": trial,
            })
            if training_id is not None:
                set_training_id(training_id)
                logger.info("Training Monitor: training_id=%s", training_id)
        except Exception as e:
            logger.warning("Training Monitor init failed: %s", e)

    # Dataset from gym tasks (no HF path)
    train_dataset, valid_dataset = get_cua_train_and_valid_datasets(
        gym_base_url=config.gym_base_url,
        gym_id=config.gym_id,
        train_ratio=config.gym_train_ratio,
        limit=config.gym_limit,
        seed=config.gym_seed,
        eval_number_range=getattr(config, "gym_eval_number_range", None),
    )

    # Load full gym task list once so rollout uses in-memory list instead of calling list_tasks every time
    gym_task_list: list[dict] = []
    try:
        from cua_rl.gym_task_source import list_genv_gym_tasks
        gym_task_list = list_genv_gym_tasks(
            gym_id=config.gym_id or "umetrip",
            gym_base_url=config.gym_base_url or None,
        ) or []
        logger.info("Loaded %d gym tasks once for rollout (no per-episode list_tasks).", len(gym_task_list))
    except Exception as e:
        logger.warning("Could not preload gym task list; rollout will call list_tasks per episode: %s", e)

    # Optional: post all loaded tasks to monitor so "Load tasks" shows in panel
    if monitor_base_url and project_token and training_id is not None:
        try:
            from cua_rl.database.monitor_context import get_ingest_client

            ingest_for_tasks = get_ingest_client()
            if ingest_for_tasks:
                seen = set()
                for ds in (train_dataset, valid_dataset):
                    for row in ds:
                        tid = row.get("task_id") or row.get("id")
                        if tid and tid not in seen:
                            seen.add(tid)
                            payload = {
                                "task_id": str(tid),
                                "name": row.get("name") or str(tid),
                                "description": row.get("description") or "",
                                "source_type": "gym",
                            }
                            ingest_for_tasks.post_task(payload, str(tid))
                logger.info("Training Monitor: posted %d tasks for Load tasks", len(seen))
        except Exception as e:
            logger.warning("Training Monitor post tasks failed: %s", e)

    model_base_url = config.model_base_url or os.environ.get("MODEL_BASE_URL", "")
    model_name = config.model_name or os.environ.get("MODEL_NAME", "")

    # Resolve AReaL vLLM URL once on main process so workers get correct baseUrl (workers may not have name_resolve).
    resolved_inference_url = ""
    exp = getattr(config, "experiment_name", None)
    trial = getattr(config, "trial_name", None)
    if exp and trial:
        try:
            from areal.utils import name_resolve, names
            name = names.gen_servers(exp, trial)
            addrs = name_resolve.get_subtree(name)
            if addrs:
                base = f"http://{addrs[0]}".rstrip("/")
                resolved_inference_url = f"{base}/v1" if "/v1" not in base else base
                logger.info("Resolved inference URL for workers: %s", resolved_inference_url)
        except Exception as e:
            logger.warning("Could not resolve inference URL (workers may use model_base_url): %s", e)

    workflow_kwargs = dict(
        tokenizer=tokenizer,
        gbox_mini_agent_base_url=config.gbox_mini_agent_base_url,
        gbox_mini_agent_agent=config.gbox_mini_agent_agent,
        gbox_mini_agent_env=None,  # Filled per episode from genv ctx
        model_base_url=model_base_url,
        model_name=model_name,
        resolved_inference_url=resolved_inference_url or None,
        experiment_name=exp,
        trial_name=trial,
        max_turns=config.max_turns,
        max_task_time_seconds=config.max_task_time_seconds,
        max_turn_time_seconds=config.max_turn_time_seconds,
        standard_action_space=config.gbox_mini_agent_standard_action_space,
        rollout_recorder=None,
        # Serializable only: no functions (workers get workflow_kwargs via RPC).
        monitor_ingest_config=None,  # Set below when monitor enabled: {base_url, project_token, training_id}
        gym_task_list=gym_task_list,  # Preloaded once; rollout uses this instead of list_tasks per episode
    )

    if monitor_base_url and project_token and training_id is not None:
        try:
            workflow_kwargs["monitor_ingest_config"] = {
                "base_url": monitor_base_url,
                "project_token": project_token,
                "training_id": training_id,
            }
        except Exception as e:
            logger.warning("Training Monitor ingest config failed: %s", e)

    eval_workflow_kwargs = workflow_kwargs.copy()

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        if monitor_base_url and project_token and training_id is not None:
            from cua_rl.database.monitor_training_hooks import record_step_before_rollout
            _rank = int(os.environ.get("RANK", "0"))
            if _rank == 0:
                trainer.on_before_step = record_step_before_rollout
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )

    if monitor_base_url and project_token and training_id is not None:
        try:
            from cua_rl.database.ingest_client import IngestClient
            from cua_rl.database.monitor_context import clear_monitor_context, get_ingest_client

            ingest = get_ingest_client()
            if ingest:
                ingest.post_training({"status": "completed", "training_id": training_id})
            clear_monitor_context()
        except Exception as e:
            logger.warning("Training Monitor finalize failed: %s", e)


if __name__ == "__main__":
    main()
