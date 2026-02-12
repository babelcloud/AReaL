"""
Global monitor context for training.

Provides training_id, baseline_id, eval_id, and ingest_client for recording
to the training monitor via HTTP. No database/sqlalchemy.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global context (monitor-only)
_training_id: Optional[int] = None
_baseline_id: Optional[int] = None
_eval_id: Optional[int] = None
_step_id: Optional[int] = None
_ingest_client: Any = None


def set_baseline_id(baseline_id: Optional[int]) -> None:
    """Set the current baseline ID for baseline evaluation."""
    global _baseline_id
    _baseline_id = baseline_id
    logger.debug(f"Baseline ID set: {baseline_id}")


def set_eval_id(eval_id: Optional[int]) -> None:
    """Set the current eval ID for evaluation."""
    global _eval_id
    _eval_id = eval_id
    logger.debug(f"Eval ID set: {eval_id}")


def set_step_id(step_id: Optional[int]) -> None:
    """Set the current step ID for training rollouts."""
    global _step_id
    _step_id = step_id
    logger.debug(f"Step ID set: {step_id}")


def get_training_id() -> Optional[int]:
    """Get the global training ID (from monitor)."""
    return _training_id


def set_training_id(training_id: Optional[int]) -> None:
    """Set the global training ID."""
    global _training_id
    _training_id = training_id
    logger.debug(f"Training ID set: {training_id}")


def get_baseline_id() -> Optional[int]:
    """Get the current baseline ID."""
    return _baseline_id


def get_eval_id() -> Optional[int]:
    """Get the current eval ID."""
    return _eval_id


def get_step_id() -> Optional[int]:
    """Get the current step ID."""
    return _step_id


def set_ingest_client(client: Any) -> None:
    """Set the ingest client for HTTP recording."""
    global _ingest_client
    _ingest_client = client


def get_ingest_client() -> Any:
    """Get the ingest client if configured."""
    return _ingest_client


def clear_monitor_context() -> None:
    """Clear the global monitor context."""
    global _training_id, _baseline_id, _eval_id, _step_id, _ingest_client
    _training_id = None
    _baseline_id = None
    _eval_id = None
    _step_id = None
    _ingest_client = None
