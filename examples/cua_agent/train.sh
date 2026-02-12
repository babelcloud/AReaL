#!/bin/bash
# CUA Agent PPO training (AReaL): genv + gbox-mini-agent + Training Monitor.
#
# Usage examples:
#   # Basic: gym + gbox-mini-agent + inference URL
#   ./train.sh --gym http://localhost:5010 --gym-id my_gym --model-base-url http://localhost:8000/v1
#
#   # With Training Monitor
#   ./train.sh --gym http://localhost:5010 --gym-id my_gym --model-base-url http://localhost:8000/v1 \
#     --monitor-base-url https://monitor.example.com --project-id PROJECT_ID --project-token YOUR_TOKEN
#
#   # Full options
#   ./train.sh --gym http://gym:5010 --gym-id gym1 --gbox-mini-url http://gbox:3000 \
#     --model-base-url http://vllm:8000/v1 --model-name Tongyi-MAI/MAI-UI-8B \
#     --monitor-base-url https://monitor.example.com --project-id PROJECT_ID --project-token YOUR_TOKEN \
#     --max-concurrent-rollouts 8 --batch-size 4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$AREAL_ROOT"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading .env from $SCRIPT_DIR/.env"
    set -a
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.env"
    set +a
fi
if [ -f "$AREAL_ROOT/.env" ]; then
    echo "Loading .env from $AREAL_ROOT/.env"
    set -a
    # shellcheck source=/dev/null
    source "$AREAL_ROOT/.env"
    set +a
fi

# Defaults
GYM_BASE_URL="${GYM_BASE_URL:-}"
GYM_ID="${GYM_ID:-}"
GBOX_MINI_AGENT_BASE_URL="${GBOX_MINI_AGENT_BASE_URL:-http://localhost:3000}"
MODEL_BASE_URL="${MODEL_BASE_URL:-}"
MODEL_NAME="${MODEL_NAME:-}"
MONITOR_BASE_URL="${MONITOR_BASE_URL:-}"
PROJECT_ID="${PROJECT_ID:-}"
PROJECT_TOKEN="${PROJECT_TOKEN:-}"
MAX_CONCURRENT_ROLLOUTS="${MAX_CONCURRENT_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_TASK_TIME="${MAX_TASK_TIME:-600}"
MAX_STEP_TIME="${MAX_STEP_TIME:-120}"
GBOX_AGENT="${GBOX_AGENT:-mai-ui}"
STANDARD_ACTION_SPACE="${STANDARD_ACTION_SPACE:-mobile}"
GYM_TRAIN_RATIO="${GYM_TRAIN_RATIO:-0.8}"
GYM_SEED="${GYM_SEED:-42}"
GYM_LIMIT="${GYM_LIMIT:-}"

# Parse CLI
while [[ $# -gt 0 ]]; do
    case $1 in
        --gym)
            GYM_BASE_URL="$2"
            shift 2
            ;;
        --gym-id)
            GYM_ID="$2"
            shift 2
            ;;
        --gbox-mini-url|--gbox-mini-agent-base-url)
            GBOX_MINI_AGENT_BASE_URL="$2"
            shift 2
            ;;
        --gbox-agent)
            GBOX_AGENT="$2"
            shift 2
            ;;
        --model-base-url|--inference-base-url)
            MODEL_BASE_URL="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --monitor-base-url)
            MONITOR_BASE_URL="$2"
            shift 2
            ;;
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --project-token)
            PROJECT_TOKEN="$2"
            shift 2
            ;;
        --max-concurrent-rollouts)
            MAX_CONCURRENT_ROLLOUTS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --max-task-time)
            MAX_TASK_TIME="$2"
            shift 2
            ;;
        --max-step-time)
            MAX_STEP_TIME="$2"
            shift 2
            ;;
        --standard-action-space)
            STANDARD_ACTION_SPACE="$2"
            shift 2
            ;;
        --gym-train-ratio)
            GYM_TRAIN_RATIO="$2"
            shift 2
            ;;
        --gym-seed)
            GYM_SEED="$2"
            shift 2
            ;;
        --gym-limit)
            GYM_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "CUA Agent PPO training (AReaL). Tasks from gym; rollout via gbox-mini-agent; optional Training Monitor."
            echo ""
            echo "Gym (task source):"
            echo "  --gym URL                    Gym server base URL (e.g. http://localhost:5010)"
            echo "  --gym-id ID                  Gym ID (required if gym requires it)"
            echo "  --gym-train-ratio RATIO     Train/eval split ratio (default: 0.8)"
            echo "  --gym-seed SEED              Random seed (default: 42)"
            echo "  --gym-limit N                Limit number of tasks (optional)"
            echo ""
            echo "gbox-mini-agent (rollout):"
            echo "  --gbox-mini-url URL         gbox-mini-agent base URL (default: http://localhost:3000)"
            echo "  --gbox-agent NAME           Agent name, e.g. mai-ui, qwen3-vl (default: mai-ui)"
            echo "  --standard-action-space SPACE  mobile|pc (default: mobile)"
            echo ""
            echo "Inference (vLLM / OpenAI-compatible API for gbox model.baseUrl):"
            echo "  --model-base-url URL        API base URL (e.g. http://localhost:8000/v1)"
            echo "  --model-name NAME           Model name (optional)"
            echo ""
            echo "Training Monitor (optional):"
            echo "  --monitor-base-url URL      Monitor ingest base URL"
            echo "  --project-id ID              Project ID"
            echo "  --project-token TOKEN        Project token"
            echo "  (Or set MONITOR_BASE_URL, PROJECT_ID, PROJECT_TOKEN in env or .env)"
            echo ""
            echo "Training:"
            echo "  --max-concurrent-rollouts N  Max concurrent rollouts (default: 8)"
            echo "  --batch-size N               Consumer batch size (default: 4)"
            echo "  --max-turns N                Max turns per task (default: 30)"
            echo "  --max-task-time SECS         Max run time per task in seconds (default: 600)"
            echo "  --max-step-time SECS         Max time per step in seconds (default: 120)"
            echo ""
            echo "Examples:"
            echo "  $0 --gym http://localhost:5010 --gym-id my_gym --model-base-url http://localhost:8000/v1"
            echo "  $0 --gym http://gym:5010 --gym-id gym1 --gbox-mini-url http://gbox:3000 --model-base-url http://vllm:8000/v1 \\"
            echo "     --monitor-base-url https://monitor.example.com --project-id PROJECT_ID --project-token YOUR_TOKEN"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage." >&2
            exit 1
            ;;
    esac
done

# Hydra overrides (key=value)
OVERRIDES=()

if [ -n "$GYM_BASE_URL" ]; then
    OVERRIDES+=( "gym_base_url=$GYM_BASE_URL" )
fi
if [ -n "$GYM_ID" ]; then
    OVERRIDES+=( "gym_id=$GYM_ID" )
fi
OVERRIDES+=( "gbox_mini_agent_base_url=$GBOX_MINI_AGENT_BASE_URL" )
OVERRIDES+=( "gbox_mini_agent_agent=$GBOX_AGENT" )
OVERRIDES+=( "gbox_mini_agent_standard_action_space=$STANDARD_ACTION_SPACE" )
OVERRIDES+=( "gym_train_ratio=$GYM_TRAIN_RATIO" )
OVERRIDES+=( "gym_seed=$GYM_SEED" )
OVERRIDES+=( "max_turns=$MAX_TURNS" )
OVERRIDES+=( "max_task_time_seconds=$MAX_TASK_TIME" )
OVERRIDES+=( "max_turn_time_seconds=$MAX_STEP_TIME" )
OVERRIDES+=( "rollout.max_concurrent_rollouts=$MAX_CONCURRENT_ROLLOUTS" )
OVERRIDES+=( "train_dataset.batch_size=$BATCH_SIZE" )
OVERRIDES+=( "valid_dataset.batch_size=$BATCH_SIZE" )

if [ -n "$MODEL_BASE_URL" ]; then
    OVERRIDES+=( "model_base_url=$MODEL_BASE_URL" )
fi
if [ -n "$MODEL_NAME" ]; then
    OVERRIDES+=( "model_name=$MODEL_NAME" )
fi
if [ -n "$MONITOR_BASE_URL" ]; then
    OVERRIDES+=( "monitor_base_url=$MONITOR_BASE_URL" )
fi
if [ -n "$PROJECT_ID" ]; then
    OVERRIDES+=( "project_id=$PROJECT_ID" )
fi
if [ -n "$PROJECT_TOKEN" ]; then
    OVERRIDES+=( "project_token=$PROJECT_TOKEN" )
fi
if [ -n "$GYM_LIMIT" ]; then
    OVERRIDES+=( "gym_limit=$GYM_LIMIT" )
fi

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "============================================"
echo "CUA Agent PPO (AReaL)"
echo "============================================"
echo "  gym_base_url:              $GYM_BASE_URL"
echo "  gym_id:                     $GYM_ID"
echo "  gbox_mini_agent_base_url:   $GBOX_MINI_AGENT_BASE_URL"
echo "  gbox_mini_agent_agent:      $GBOX_AGENT"
echo "  model_base_url:             $MODEL_BASE_URL"
echo "  model_name:                 $MODEL_NAME"
echo "  monitor_base_url:           $MONITOR_BASE_URL"
echo "  project_id:                  $PROJECT_ID"
echo "  project_token:               ${PROJECT_TOKEN:+***${PROJECT_TOKEN: -4}}"
echo "  max_concurrent_rollouts:    $MAX_CONCURRENT_ROLLOUTS"
echo "  batch_size:                 $BATCH_SIZE"
echo "============================================"
echo ""

CMD=( uv run python "$SCRIPT_DIR/train.py" --config "$SCRIPT_DIR/config.yaml" "${OVERRIDES[@]}" )
echo "Running: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
