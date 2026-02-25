"""
Training hooks for monitor (HTTP ingest).

Records steps, baselines, and evals via IngestClient. No database.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from cua_rl.database.monitor_context import (
    get_ingest_client,
    get_training_id,
    set_baseline_id,
    set_eval_id,
    set_step_id,
)

logger = logging.getLogger(__name__)


def _serialize_dt(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def record_step_before_rollout(step: int, batch: Optional[int] = None) -> Optional[int]:
    """
    Post step to monitor before rollout.
    Returns step_id from monitor, or None if ingest not available.
    """
    ingest = get_ingest_client()
    training_id = get_training_id()
    if not ingest or not training_id:
        logger.warning(
            "Monitor: step not posted (ingest or training_id missing). "
            "Timeline will not show this step; check Monitor is enabled and post_training returned training_id."
        )
        return None
    try:
        client_id = f"step-{training_id}-{step}-{batch or 0}"
        payload = {
            "training_id": training_id,
            "step": step,
            "batch": batch,
            "status": "running",
            "current_phase": "rollout",
            "start_time": datetime.utcnow().isoformat(),
        }
        logger.info(
            "Monitor: posting step %s for training_id=%s (open this training in Monitor Timeline to see steps/rollouts)",
            step,
            training_id,
        )
        step_id = ingest.post_step(payload, client_id)
        if step_id is None:
            logger.warning(
                "Monitor: post_step returned no step_id; check ingest API, training_id, and network. "
                "Rollouts may be dropped or not linked to this step."
            )
            return None
        set_step_id(step_id)
        logger.info("Monitor: step %s posted, step_id=%s", step, step_id)
        return step_id
    except Exception as e:
        logger.warning(f"Failed to record step before rollout to monitor: {e}")
        return None


def record_step_after_rollout(step: int, model_path: Optional[str] = None) -> Optional[int]:
    """Step is already created; no additional monitor update needed for post-rollout."""
    return None


def record_step_before_training(step: int) -> Optional[int]:
    """No separate hook for monitor."""
    return None


def record_step_after_training(
    step: int,
    model_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Optionally update step completion via ingest. For now, no-op."""
    return None


def create_baseline_via_monitor(model_path: str) -> Optional[int]:
    """Create baseline record in monitor. Returns baseline_id."""
    ingest = get_ingest_client()
    training_id = get_training_id()
    if not ingest or not training_id:
        return None
    try:
        client_id = f"baseline-{training_id}-{datetime.utcnow().isoformat()}"
        payload = {
            "training_id": training_id,
            "model_path": model_path,
            "status": "running",
            "start_time": datetime.utcnow().isoformat(),
        }
        baseline_id = ingest.post_baseline(payload, client_id)
        if baseline_id is not None:
            set_baseline_id(baseline_id)
        return baseline_id
    except Exception as e:
        logger.warning(f"Failed to create baseline via monitor: {e}")
        return None


def create_eval_via_monitor(step: int, model_path: str) -> Optional[int]:
    """Create eval record in monitor. Returns eval_id."""
    ingest = get_ingest_client()
    training_id = get_training_id()
    if not ingest or not training_id:
        return None
    try:
        client_id = f"eval-{training_id}-{step}"
        payload = {
            "training_id": training_id,
            "step": step,
            "model_path": model_path,
            "status": "running",
            "start_time": datetime.utcnow().isoformat(),
        }
        eval_id = ingest.post_eval(payload, client_id)
        if eval_id is not None:
            set_eval_id(eval_id)
        return eval_id
    except Exception as e:
        logger.warning(f"Failed to create eval via monitor: {e}")
        return None
