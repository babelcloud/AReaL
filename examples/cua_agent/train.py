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
            training_id = ingest.post_training({
                "status": "running",
                "current_phase": "rollout",
                "experiment_name": getattr(config, "experiment_name", "cua_agent"),
                "trial_name": getattr(config, "trial_name", "run"),
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
    )

    model_base_url = config.model_base_url or os.environ.get("MODEL_BASE_URL", "")
    model_name = config.model_name or os.environ.get("MODEL_NAME", "")

    workflow_kwargs = dict(
        tokenizer=tokenizer,
        gbox_mini_agent_base_url=config.gbox_mini_agent_base_url,
        gbox_mini_agent_agent=config.gbox_mini_agent_agent,
        gbox_mini_agent_env=None,  # Filled per episode from genv ctx
        model_base_url=model_base_url,
        model_name=model_name,
        max_turns=config.max_turns,
        max_task_time_seconds=config.max_task_time_seconds,
        max_turn_time_seconds=config.max_turn_time_seconds,
        standard_action_space=config.gbox_mini_agent_standard_action_space,
        rollout_recorder=None,  # Optional: HttpRolloutRecorder when monitor enabled
    )

    eval_workflow_kwargs = workflow_kwargs.copy()

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
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
