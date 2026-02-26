"""CUA Agent training config: extends PPOConfig with gym, gbox-mini-agent, and monitor."""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class CUAConfig(PPOConfig):
    """Configuration for CUA Agent PPO training (genv + gbox-mini-agent + Training Monitor)."""

    workflow: str = field(
        default="cua_workflow.CUAAgentWorkflow",
        metadata={"help": "Path to CUAAgentWorkflow class (run from examples/cua_agent with path in sys.path)."},
    )
    eval_workflow: str = field(
        default="cua_workflow.CUAAgentWorkflow",
        metadata={"help": "Path to workflow class for evaluation."},
    )
    max_turns: int = field(
        default=30,
        metadata={"help": "Maximum agent turns per task."},
    )
    max_task_time_seconds: int | None = field(
        default=600,
        metadata={"help": "Max wall time per run (seconds)."},
    )
    max_turn_time_seconds: int | None = field(
        default=120,
        metadata={"help": "Max time per turn (seconds)."},
    )

    # Gym
    gym_base_url: str = field(
        default="http://localhost:5010",
        metadata={"help": "Gym server base URL."},
    )
    gym_id: str = field(
        default="",
        metadata={"help": "Gym ID for task listing and execution."},
    )
    gym_train_ratio: float = field(default=0.8, metadata={"help": "Train ratio for train/eval split."})
    gym_split_type: str | None = field(
        default="train",
        metadata={"help": "Split: 'train' | 'eval' | None (all)."},
    )
    gym_limit: int | None = field(default=None, metadata={"help": "Limit number of tasks."})
    gym_eval_number_range: str | None = field(default=None, metadata={"help": "Eval set by task number range, e.g. 001-032; rest = train. Overrides train_ratio."})
    gym_seed: int = field(default=42, metadata={"help": "Seed for task sampling/split."})
    gym_tags: str | None = field(
        default=None,
        metadata={"help": "Comma-separated tags to filter tasks by difficulty (e.g. 'easy', 'easy,normal', 'hard'). Tasks with ANY of these tags are included."},
    )

    # gbox-mini-agent
    gbox_mini_agent_base_url: str = field(
        default="http://localhost:3000",
        metadata={"help": "gbox-mini-agent HTTP base URL."},
    )
    gbox_mini_agent_agent: str = field(
        default="mai-ui",
        metadata={"help": "Agent name (e.g. mai-ui, qwen3-vl)."},
    )
    gbox_mini_agent_standard_action_space: str | None = field(
        default="mobile",
        metadata={"help": "Action space: 'mobile' | 'pc'."},
    )
    model_base_url: str = field(
        default="",
        metadata={"help": "OpenAI-compatible API URL for inference (vLLM/proxy). Used as model.baseUrl for gbox."},
    )
    model_name: str = field(
        default="",
        metadata={"help": "Model name for gbox run (e.g. model name on vLLM)."},
    )

    # Training Monitor
    monitor_base_url: str = field(
        default="",
        metadata={"help": "Training Monitor ingest base URL. Empty = disable."},
    )
    project_id: str = field(
        default="",
        metadata={"help": "Training Monitor project ID (or set PROJECT_ID)."},
    )
    project_token: str = field(
        default="",
        metadata={"help": "Project token for monitor (or set MONITOR_PROJECT_TOKEN)."},
    )
