# CUA Agent PPO Training

CUA (Computer Use Agent) training on AReaL: one episode = **genv** execution + **gbox-mini-agent** run + trajectory conversion for PPO.

- **Path A**: AReaL manages vLLM/proxy; gbox-mini-agent's `model.baseUrl` points to it. No tinker-cookbook dependency.
- **Tasks**: Loaded from gym server (`gym_base_url`, `gym_id`).
- **Monitor**: Optional Training Monitor (IngestClient) via `monitor_base_url` and `project_token`.

## Prerequisites

- **gym server** and **gbox-mini-agent** running.
- **vLLM** (or OpenAI-compatible proxy) for inference; set `model_base_url` / `MODEL_BASE_URL` so gbox uses it.
- Python deps: `httpx`, `datasets` (AReaL env or install separately for `cua_rl`).

## Config

- `config.yaml`: experiment, gym, gbox, rollout, actor, dataset placeholders.
- Override: `gym_id=your_gym`, `model_base_url=http://vllm:8000/v1`, `rollout.max_concurrent_rollouts=8`.

## Run

### Using `train.sh` (recommended)

From AReaL repo root. Script accepts CLI args and passes them as Hydra overrides; optional `.env` in `examples/cua_agent/` or repo root is loaded.

```bash
cd /path/to/AReaL
./examples/cua_agent/train.sh --gym http://localhost:5010 --gym-id my_gym \
  --model-base-url http://localhost:8000/v1
```

With Training Monitor:

```bash
./examples/cua_agent/train.sh --gym http://localhost:5010 --gym-id my_gym \
  --gbox-mini-url http://localhost:3000 --model-base-url http://localhost:8000/v1 \
  --monitor-base-url https://monitor.example.com --project-id YOUR_PROJECT_ID --project-token YOUR_TOKEN \
  --max-concurrent-rollouts 8 --batch-size 4
```

All options: `./examples/cua_agent/train.sh --help`

### Manual Python invocation

From AReaL repo root, with `examples/cua_agent` on `PYTHONPATH`:

```bash
cd /path/to/AReaL
export PYTHONPATH=examples/cua_agent:$PYTHONPATH
uv run python examples/cua_agent/train.py --config examples/cua_agent/config.yaml \
  gym_id=your_gym_id \
  model_base_url=http://localhost:8000/v1
```

Or from `examples/cua_agent`:

```bash
cd examples/cua_agent
uv run python train.py --config config.yaml gym_id=your_gym_id model_base_url=http://localhost:8000/v1
```

## vLLM (CUA)

For CUA, vLLM should be started with:

- `--max-model-len 262144`
- `--limit-mm-per-prompt '{"image":64}'`
- `--allowed-media-domains` for TOS/screenshot domains

## Files

| Path | Description |
|------|-------------|
| `train.py` | Entry: config, Monitor, dataset from gym, PPOTrainer, workflow_kwargs |
| `configs.py` | CUAConfig (gym, gbox, monitor, model_base_url) |
| `config.yaml` | Default YAML (path/type for dataset are placeholders; tasks from gym) |
| `dataset.py` | get_cua_task_dataset / get_cua_train_and_valid_datasets from gym |
| `cua_workflow.py` | CUAAgentWorkflow: genv + gbox run + trajectory tensors |
| `cua_rl/` | Copied from tinker-cookbook (genv, gbox client, task_loader, database, trajectory_utils); no tinker SDK |
