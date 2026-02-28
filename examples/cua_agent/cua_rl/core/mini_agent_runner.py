from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict

from cua_rl.gbox_mini_agent_client import GboxMiniAgentClient
from cua_rl.core.rollout_logger import RolloutLogger

MAX_GBOX_API_RETRIES = 5
GBOX_API_RETRY_DELAY_SECONDS = 0.5


def _event_key(event: dict) -> str:
    """Create a stable key for event de-duplication."""
    event_type = event.get("type", "unknown")
    ts = event.get("ts", "")
    # gbox-mini-agent migrated from turn_* events to step_* events.
    # Keep backward compatibility with older servers that still send "turn".
    turn = event.get("step", event.get("turn"))
    action_index = event.get("actionIndex")
    return f"{event_type}:{ts}:{turn}:{action_index}"


def _extract_text_from_model_output(model_output: Any) -> str:
    if model_output is None:
        return ""
    if isinstance(model_output, str):
        return model_output
    if isinstance(model_output, dict):
        choices = model_output.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    try:
                        return json.dumps(tool_calls, ensure_ascii=False, default=str)
                    except Exception:
                        return str(tool_calls)
        for key in ("text", "content", "message"):
            value = model_output.get(key)
            if isinstance(value, str) and value:
                return value
    try:
        return json.dumps(model_output, default=str)
    except Exception:
        return str(model_output)


def _action_to_log_fields(action: dict) -> dict:
    """Convert gbox-mini-agent action to RolloutLogger.log_action fields."""
    action_type = action.get("name") or action.get("action_type") or "unknown"
    args = action.get("args") or {}
    payload: dict = {"action_type": action_type}

    if "point" in args and isinstance(args["point"], dict):
        payload["coordinates"] = args["point"]
        payload["coordinates_normalized"] = True
    if "startPoint" in args and "endPoint" in args:
        payload["coordinates"] = {
            "start": args.get("startPoint"),
            "end": args.get("endPoint"),
        }
        payload["coordinates_normalized"] = True
    if "content" in args:
        payload["text"] = args.get("content")
    if "key" in args:
        payload["key"] = args.get("key")
    if "direction" in args:
        payload["direction"] = args.get("direction")
    if "durationMs" in args:
        payload["duration"] = args.get("durationMs")
    if "target" in args:
        payload["target_desc"] = args.get("target")
    if "destination" in args:
        payload["end_target"] = args.get("destination")
    if "rawFeedback" in action:
        payload["raw_feedback"] = action.get("rawFeedback")
    if "rawResult" in action:
        payload["raw_result"] = action.get("rawResult")

    return payload


async def _retry_fetch(description: str, fetch_fn, *, max_attempts: int = MAX_GBOX_API_RETRIES,
                       delay_seconds: float = GBOX_API_RETRY_DELAY_SECONDS) -> tuple[Any | None, Exception | None]:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fetch_fn(), None
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay_seconds)
    return None, last_exc


def _steps_to_turns(steps: list[dict]) -> list[dict]:
    """Convert gbox-mini-agent steps to rollout_logger turn format."""
    turns: list[dict] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        turn_num = idx + 1
        before_screenshot = _extract_screenshot_from_messages(step.get("messages"))
        after_screenshot = _extract_screenshot_from_actions(step.get("actions"))
        action_results: list[dict] = []
        actions = step.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                action_results.append(_sanitize_json_payload(_action_to_log_fields(action)))

        model_response = _step_to_model_response(step, action_results)

        turns.append({
            "turn_num": turn_num,
            "action_results": action_results,
            "model_response": model_response,
            "screenshots": {
                "before": before_screenshot,
                "after": after_screenshot,
            },
            "parse_success": True,
            "parse_error": None,
        })
    return turns


def _extract_screenshot_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None

    def _extract_from_message(msg: dict) -> str | None:
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        for part in reversed(content):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "image":
                url = part.get("url") or part.get("image")
                if isinstance(url, str) and url:
                    return url
            if part_type == "image_url":
                image_url = part.get("image_url") or {}
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str) and url:
                        return url
        return None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        if msg.get("callId") == "__step_message_image__":
            candidate = _extract_from_message(msg)
            if candidate:
                return candidate

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        candidate = _extract_from_message(msg)
        if candidate:
            return candidate

    return None


def _extract_screenshot_from_actions(actions: Any) -> str | None:
    """Extract screenshot_after from action observations.
    
    The screenshots array from genv contains: [screenshot_before, screenshot_after]
    We need to find the one named 'screenshot_after', not just take the first element.
    """
    if not isinstance(actions, list):
        return None
    for action in reversed(actions):
        if not isinstance(action, dict):
            continue
        obs = action.get("obs") or {}
        screenshots = obs.get("screenshots") or []
        if isinstance(screenshots, list) and screenshots:
            # First, try to find screenshot_after by name
            for shot in screenshots:
                if isinstance(shot, dict) and shot.get("name") == "screenshot_after":
                    url = shot.get("url")
                    if isinstance(url, str) and url:
                        return url
            # Fallback: if no screenshot_after found, use the last screenshot
            last = screenshots[-1]
            if isinstance(last, dict):
                url = last.get("url")
                if isinstance(url, str) and url:
                    return url
    return None


def _log_step_screenshot_debug(
    rollout_logger: RolloutLogger, turn: int, kind: str, step: dict
) -> None:
    """Log step structure when screenshot extraction returns None (for debugging format mismatches)."""
    keys = list(step.keys()) if isinstance(step, dict) else []
    messages = step.get("messages") if isinstance(step, dict) else None
    actions = step.get("actions") if isinstance(step, dict) else None
    msg_len = len(messages) if isinstance(messages, list) else 0
    act_len = len(actions) if isinstance(actions, list) else 0
    first_msg_keys = list(messages[0].keys()) if isinstance(messages, list) and messages and isinstance(messages[0], dict) else []
    first_act_keys = list(actions[0].keys()) if isinstance(actions, list) and actions and isinstance(actions[0], dict) else []
    rollout_logger.log(
        f"[DEBUG] Turn {turn} {kind} not found: step_keys={keys}, messages_len={msg_len}, first_msg_keys={first_msg_keys}, actions_len={act_len}, first_act_keys={first_act_keys}",
        color="YELLOW",
    )


def _parse_iso_ts(ts: Any) -> float | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _event_step(event: dict) -> int:
    """Get step index from event (gbox-mini-agent uses step, 1-based)."""
    return int(event.get("step", event.get("turn", 0)))


def _step_to_turn_num(step: int) -> int:
    # gbox-mini-agent step is already 1-based, so return as-is
    """Return step as turn_num (gbox-mini-agent step is already 1-based)."""
    return step


def _events_to_turn_timings(events: list[dict]) -> dict[int, dict]:
    """Derive per-turn stage timings from gbox-mini-agent events.

    gbox-mini-agent uses 0-based `step` in events. Returns dict keyed by 0-based step;
    convert to turn_num via _step_to_turn_num(step) when recording to ingest.
    """
    per_turn: dict[int, dict] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        turn = event.get("step", event.get("turn"))
        if not isinstance(turn, int):
            continue
        entry = per_turn.setdefault(turn, {})
        event_type = event.get("type")
        ts = _parse_iso_ts(event.get("ts"))
        if event_type in ("step_start", "turn_start"):
            entry["turn_start_ts"] = ts
        elif event_type == "model_call":
            entry["model_call_ts"] = ts
        elif event_type == "model_output":
            entry["model_output_ts"] = ts
        elif event_type in ("step_end", "turn_end"):
            entry["turn_end_ts"] = ts
        elif event_type == "action_executed":
            start_ts = _parse_iso_ts(event.get("startTs"))
            end_ts = _parse_iso_ts(event.get("endTs"))
            duration_ms = event.get("durationMs")
            if isinstance(duration_ms, (int, float)):
                entry["action_exec_ms"] = entry.get("action_exec_ms", 0.0) + float(duration_ms)
            if start_ts is not None:
                entry["first_action_start_ts"] = min(entry.get("first_action_start_ts", start_ts), start_ts)
            if end_ts is not None:
                entry["last_action_end_ts"] = max(entry.get("last_action_end_ts", end_ts), end_ts)

    timings: dict[int, dict] = {}
    for turn, entry in per_turn.items():
        stage_timings: dict[str, float] = {}
        if entry.get("turn_start_ts") is not None and entry.get("model_call_ts") is not None:
            stage_timings["screenshot_before"] = max(0.0, entry["model_call_ts"] - entry["turn_start_ts"])
        if entry.get("model_call_ts") is not None and entry.get("model_output_ts") is not None:
            stage_timings["model_inference"] = max(0.0, entry["model_output_ts"] - entry["model_call_ts"])
        if entry.get("action_exec_ms") is not None:
            stage_timings["action_exec"] = max(0.0, entry["action_exec_ms"] / 1000.0)
        if entry.get("first_action_start_ts") is not None and entry.get("last_action_end_ts") is not None:
            stage_timings["action_total"] = max(0.0, entry["last_action_end_ts"] - entry["first_action_start_ts"])

        turn_time = None
        if entry.get("turn_start_ts") is not None and entry.get("turn_end_ts") is not None:
            turn_time = max(0.0, entry["turn_end_ts"] - entry["turn_start_ts"])

        timings[turn] = {"stage_timings": stage_timings, "turn_time": turn_time}
    return timings


def _sanitize_json_payload(payload: Any) -> Any:
    """Ensure payload is JSON-serializable (best-effort)."""
    try:
        sanitized = json.loads(json.dumps(payload, default=str))
        if isinstance(sanitized, dict):
            sanitized.pop("logs", None)
        return sanitized
    except Exception:
        return str(payload)


def _step_to_model_response(step: dict, action_results: list[dict]) -> str:
    """Build structured model response from gbox-mini-agent step."""
    thinking = step.get("thinking")
    text = step.get("text")
    actions = action_results if action_results else None
    payload = {
        "thinking": thinking if thinking is not None else None,
        "text": text if text is not None else None,
        "action": actions,
    }
    return json.dumps(_sanitize_json_payload(payload), ensure_ascii=False, default=str)


def _extract_model_input_screenshot(model_input: Any, turn: int | None = None) -> tuple[str | None, Any]:
    """Extract the most recent screenshot URL from model input."""
    if not isinstance(model_input, dict):
        return None, model_input

    messages = model_input.get("messages")
    if not isinstance(messages, list):
        return None, model_input

    screenshot_uri = None
    readable_messages = []

    def _scan_message_for_image(message: dict) -> str | None:
        content = message.get("content")
        if not isinstance(content, list):
            return None
        for part in reversed(content):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "image":
                url = part.get("url") or part.get("image")
                if isinstance(url, str) and url:
                    return url
            if part_type == "image_url":
                image_url = part.get("image_url") or {}
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str) and url:
                        return url
        return None

    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            candidate = _scan_message_for_image(msg)
            if candidate:
                screenshot_uri = candidate
                break

    if screenshot_uri is None:
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            candidate = _scan_message_for_image(msg)
            if candidate:
                screenshot_uri = candidate
                break

    for msg in messages:
        readable_messages.append(msg)

    readable = {
        "messages": readable_messages,
        "image_placeholders": {"__SCREENSHOT_BEFORE__": "screenshot_before"},
        "meta": {"turn": turn},
    }
    return screenshot_uri, readable


async def run_mini_agent_rollout(
    *,
    task: str,
    rollout_logger: RolloutLogger,
    rollout_recorder,
    max_turns: int,
    gbox_mini_agent_base_url: str,
    gbox_mini_agent_agent: str,
    gbox_mini_agent_model: dict | None,
    gbox_mini_agent_env: dict | None,
    gbox_mini_agent_standard_action_space: str | None,
    max_task_time_seconds: int | None = None,
    max_turn_time_seconds: int | None = None,
    step: int | None = None,
    group_num: int | None = None,
    task_id: str | None = None,
    execution_id: str | None = None,
    task_number: str | None = None,
) -> Dict[str, Any]:
    payload: dict = {
        "task": task,
        "agent": gbox_mini_agent_agent,
        "maxSteps": max_turns,
    }
    if max_task_time_seconds is not None and max_task_time_seconds > 0:
        payload["maxRunTimeSec"] = max_task_time_seconds
    if max_turn_time_seconds is not None and max_turn_time_seconds > 0:
        payload["maxStepTimeSec"] = max_turn_time_seconds

    if gbox_mini_agent_model is not None:
        if not isinstance(gbox_mini_agent_model, dict):
            raise ValueError("gbox_mini_agent_model must be a dict when provided")
        payload["model"] = gbox_mini_agent_model

    if gbox_mini_agent_env is not None:
        if not isinstance(gbox_mini_agent_env, dict):
            raise ValueError("gbox_mini_agent_env must be a dict when provided")
        payload["env"] = gbox_mini_agent_env

    if gbox_mini_agent_standard_action_space is not None:
        if gbox_mini_agent_standard_action_space not in {"mobile", "pc"}:
            raise ValueError("gbox_mini_agent_standard_action_space must be 'mobile' or 'pc'")
        payload["standardActionSpace"] = gbox_mini_agent_standard_action_space

    rollout_logger.log(
        f"[gbox-mini-agent] Starting run: agent={gbox_mini_agent_agent} "
        f"max_turns={max_turns} base_url={gbox_mini_agent_base_url}"
    )

    rollout_id_short = (rollout_logger.rollout_id or "")[:8]
    log_prefix_parts = []
    if step is not None:
        log_prefix_parts.append("step=%s" % step)
    if group_num is not None:
        log_prefix_parts.append("group=%s" % group_num)
    if rollout_id_short:
        log_prefix_parts.append("rollout=%s" % rollout_id_short)
    log_prefix = " [%s]" % " ".join(log_prefix_parts) if log_prefix_parts else ""

    events_seen: set[str] = set()
    model_input_recorded_turns: set[int] = set()
    steps_recorded_turns: set[int] = set()
    screenshot_recorded_turns: set[int] = set()
    turn_model_response: dict[int, str] = {}
    turn_model_specific_input: dict[int, dict] = {}  # Track modelSpecificInput per turn
    turn_model_input: dict[int, dict] = {}  # Track modelInput per turn (to be recorded at step_end)
    turn_model_output: dict[int, dict] = {}  # Track modelOutput per turn (to be recorded at step_end)
    turn_actions: dict[int, list[dict]] = {}  # Track action payloads per turn from events
    recorded_action_call_ids: set[str] = set()
    max_turn_seen = 0
    errors: list[str] = []
    run_ok: bool | None = None
    run_reason: str | None = None
    last_event_ts: str | None = None  # Track last event timestamp for incremental polling
    answer: str | None = None  # Track answer from finished action

    if "events" not in rollout_logger.trajectory_data:
        rollout_logger.trajectory_data["events"] = []

    def _persist_execution_details():
        if rollout_recorder is None:
            return
        try:
            payload = {
                "training_data": [],
                "execution_details": _sanitize_json_payload(rollout_logger.trajectory_data),
            }
            rollout_recorder.update(trajectory_data_json=json.dumps(payload, default=str))
        except Exception as exc:
            rollout_logger.log(f"[gbox-mini-agent] Failed to persist execution details: {exc}", color="YELLOW")

    async with GboxMiniAgentClient(base_url=gbox_mini_agent_base_url) as client:
        run_id = await client.start_run(payload)
        rollout_logger.log(f"[gbox-mini-agent] Run started: run_id={run_id}")

        if rollout_recorder is not None:
            rollout_recorder.update_environment(gbox_id=run_id, status="running")
            rollout_recorder.update_status(
                status="running",
                current_phase="task_execution",
                status_message=f"gbox-mini-agent run_id={run_id}",
                gbox_run_id=run_id,
                gbox_base_url=gbox_mini_agent_base_url,
            )

        while True:
            # Poll events incrementally using timestamp filtering (every 3 seconds)
            events, events_error = await _retry_fetch(
                "events",
                lambda: client.get_events(run_id, from_ts=last_event_ts),
            )
            if events_error is not None:
                rollout_logger.log(
                    f"[gbox-mini-agent] Events fetch failed after retries: {events_error}",
                    color="YELLOW",
                )
                events = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                key = _event_key(event)
                if key in events_seen:
                    continue
                events_seen.add(key)
                rollout_logger.trajectory_data["events"].append(event)
                
                # Update last_event_ts for next incremental poll
                event_ts = event.get("ts")
                if isinstance(event_ts, str) and event_ts:
                    last_event_ts = event_ts

                event_type = event.get("type")
                # Newer gbox-mini-agent uses step_* events with `step` (1-based).
                # Keep backward compatibility with turn_* and `turn`.
                if event_type in ("step_start", "turn_start"):
                    step = _event_step(event)
                    turn_num = _step_to_turn_num(step)
                    max_turn_seen = max(max_turn_seen, turn_num)
                    rollout_logger.start_turn(turn_num, max_turns)
                    rollout_logger.log(
                        "Rollout%s turn %s/%s" % (log_prefix, turn_num, max_turns),
                        flush_immediately=True,
                    )
                    if rollout_recorder is not None:
                        rollout_recorder.start_turn(turn_num)
                        rollout_recorder.update_status(
                            status="running",
                            current_phase="task_execution",
                            current_turn=turn_num,
                        )
                elif event_type == "model_call":
                    step = _event_step(event)
                    turn_num = _step_to_turn_num(step)
                    max_turn_seen = max(max_turn_seen, turn_num)
                    # Save modelSpecificInput for this turn
                    model_specific_input = event.get("modelSpecificInput")
                    if model_specific_input is not None:
                        turn_model_specific_input[turn_num] = model_specific_input
                    # Store model input for later recording at step_end (instead of recording now)
                    model_input = (
                        model_specific_input
                        or event.get("modelInput")
                        or ({"messages": event.get("messages")} if isinstance(event.get("messages"), list) else None)
                    )
                    if model_input is not None:
                        turn_model_input[turn_num] = model_input
                elif event_type == "model_output":
                    step = _event_step(event)
                    turn_num = _step_to_turn_num(step)
                    max_turn_seen = max(max_turn_seen, turn_num)
                    # Store model output for later recording at step_end (instead of recording now)
                    model_output = event.get("modelOutput")
                    if model_output is not None:
                        turn_model_output[turn_num] = model_output
                    # Extract response text for logging
                    response_text = _extract_text_from_model_output(model_output)
                    turn_model_response[turn_num] = response_text
                    rollout_logger.log_model_inference(
                        turn_num=turn_num,
                        response_text=response_text,
                        parse_success=True,
                        inference_time=0.0,
                    )
                elif event_type == "action_executed":
                    step = _event_step(event)
                    turn_num = _step_to_turn_num(step)
                    max_turn_seen = max(max_turn_seen, turn_num)
                    action = event.get("action") or {}
                    action_fields = _action_to_log_fields(action)
                    duration_ms = event.get("durationMs")
                    exec_time = duration_ms / 1000.0 if isinstance(duration_ms, (int, float)) else None
                    rollout_logger.log_action(
                        coord_time=None,
                        exec_time=exec_time,
                        total_time=exec_time,
                        **action_fields,
                    )
                    action_payload = {
                        "name": action.get("name"),
                        "args": action.get("args") or {},
                        "callId": action.get("callId") or action.get("call_id"),
                    }
                    turn_actions.setdefault(turn_num, []).append(action_payload)
                    # Record action immediately from event stream to avoid relying on steps fetch.
                    if rollout_recorder is not None:
                        call_id = action_payload.get("callId")
                        dedupe_key = call_id or f"{turn_num}:{len(turn_actions.get(turn_num, []))}"
                        if dedupe_key not in recorded_action_call_ids:
                            action_id = rollout_recorder.record_action(
                                turn_num=turn_num,
                                action=action_payload,
                            )
                            if action_id is None:
                                rollout_logger.log(
                                    f"[WARNING] Failed to record action for turn {turn_num} (event)",
                                    color="YELLOW",
                                )
                            recorded_action_call_ids.add(dedupe_key)
                    
                    # Extract answer from finished action
                    if action.get("name") == "finished":
                        args = action.get("args") or {}
                        content = args.get("content")
                        if isinstance(content, str) and content:
                            answer = content

                elif event_type in ("step_end", "turn_end"):
                    step = _event_step(event)
                    turn_num = _step_to_turn_num(step)
                    max_turn_seen = max(max_turn_seen, turn_num)
                    rollout_logger.end_turn(turn_num)

                    # Compute per-turn timing (best-effort) so the monitor can render stage timeline
                    # without a second end_turn() call that would overwrite screenshots/actions.
                    turn_time = None
                    metrics = None
                    try:
                        turn_timings_now = _events_to_turn_timings(
                            rollout_logger.trajectory_data.get("events", [])
                        )
                        timing = turn_timings_now.get(step)  # step is 0-based
                        if isinstance(timing, dict):
                            turn_time = timing.get("turn_time")
                            metrics = {"stage_timings": timing.get("stage_timings", {})}
                    except Exception:
                        # Never fail the rollout on timing extraction
                        pass
                    model_response = ""
                    step_payload: dict | None = None
                    step_error: Exception | None = None
                    step_index = step - 1  # Convert 1-based step to 0-based index for get_steps
                    step_fetch_ok = False
                    fetched_step: dict | None = None
                    # Fetch the new step by index (0-based)
                    try:
                        if step_index >= 0:
                            for attempt in range(3):
                                await asyncio.sleep(0.5 * (attempt + 1))
                                fetched_step, step_error = await _retry_fetch(
                                    "steps",
                                    lambda: client.get_steps(run_id, index=step_index),
                                )
                                if step_error is None and isinstance(fetched_step, dict):
                                    step_fetch_ok = True
                                    # Unwrap if API returns {"step": {...}}
                                    if "step" in fetched_step and isinstance(fetched_step.get("step"), dict):
                                        fetched_step = fetched_step["step"]
                                    break
                                fetched_step = None
                            if not step_fetch_ok and step_error is not None:
                                rollout_logger.log(
                                    f"[gbox-mini-agent] Step fetch failed after retries: {step_error}",
                                    color="YELLOW",
                                )
                            if isinstance(fetched_step, dict) and rollout_recorder is not None:
                                step_turn = step  # step is already 1-based turn_num
                                if step_turn not in steps_recorded_turns:
                                    # Record model_input from step messages (mini-agent stores TOS URLs, not base64)
                                    step_messages = fetched_step.get("messages") if isinstance(fetched_step.get("messages"), list) else None
                                    if step_messages:
                                        model_input_payload = {"messages": step_messages, "meta": {"turn": step_turn}}
                                        obs_id = rollout_recorder.record_observation(
                                            turn_num=step_turn,
                                            obs_type="model_input",
                                            model_input=_sanitize_json_payload(model_input_payload),
                                        )
                                        if obs_id is not None:
                                            rollout_logger.log(f"[DEBUG] Recorded model_input for turn {step_turn}, obs_id={obs_id}")
                                        else:
                                            rollout_logger.log(f"[WARNING] Failed to record model_input for turn {step_turn}", color="YELLOW")

                                    before_screenshot = _extract_screenshot_from_messages(fetched_step.get("messages"))
                                    if not before_screenshot and (fetched_step.get("messages") or fetched_step.get("actions")):
                                        _log_step_screenshot_debug(rollout_logger, step_turn, "screenshot_before", fetched_step)
                                    if before_screenshot:
                                        rollout_recorder.record_observation(
                                            turn_num=step_turn,
                                            obs_type="screenshot_before",
                                            screenshot_uri=before_screenshot,
                                        )
                                        screenshot_recorded_turns.add(step_turn)
                                    actions = fetched_step.get("actions") if isinstance(fetched_step.get("actions"), list) else []
                                    step_action_results: list[dict] = []
                                    actions_ok = True
                                    for action in actions:
                                        if not isinstance(action, dict):
                                            continue
                                        step_action_results.append(_sanitize_json_payload(_action_to_log_fields(action)))
                                        action_payload = {
                                            "name": action.get("name"),
                                            "args": action.get("args") or {},
                                            "callId": action.get("callId") or action.get("call_id"),
                                        }
                                        call_id = action_payload.get("callId")
                                        dedupe_key = call_id or f"{step_turn}:{len(turn_actions.get(step_turn, []))}"
                                        if dedupe_key not in recorded_action_call_ids:
                                            action_id = rollout_recorder.record_action(
                                                turn_num=step_turn,
                                                action=action_payload,
                                            )
                                            if action_id is None:
                                                actions_ok = False
                                                rollout_logger.log(
                                                    f"[WARNING] Failed to record action for turn {step_turn}",
                                                    color="YELLOW",
                                                )
                                            recorded_action_call_ids.add(dedupe_key)
                                    model_response = _step_to_model_response(fetched_step, step_action_results)
                                    turn_model_response[step_turn] = model_response
                                    after_screenshot = _extract_screenshot_from_actions(actions)
                                    if not after_screenshot and actions:
                                        _log_step_screenshot_debug(rollout_logger, step_turn, "screenshot_after", fetched_step)
                                    if after_screenshot:
                                        rollout_recorder.record_observation(
                                            turn_num=step_turn,
                                            obs_type="screenshot_after",
                                            screenshot_uri=after_screenshot,
                                        )
                                        screenshot_recorded_turns.add(step_turn)
                                    if actions_ok:
                                        steps_recorded_turns.add(step_turn)
                            elif rollout_recorder is not None:
                                # Fall back to event-captured data if step payload is missing.
                                step_turn = step  # step is already 1-based turn_num
                                if step_turn not in steps_recorded_turns:
                                    # Skip model_input - no step data, event may have base64
                                    actions_ok = True
                                    for action_payload in turn_actions.get(step_turn, []):
                                        call_id = action_payload.get("callId")
                                        dedupe_key = call_id or f"{step_turn}:{len(turn_actions.get(step_turn, []))}"
                                        if dedupe_key in recorded_action_call_ids:
                                            continue
                                        action_id = rollout_recorder.record_action(
                                            turn_num=step_turn,
                                            action=action_payload,
                                        )
                                        if action_id is None:
                                            actions_ok = False
                                            rollout_logger.log(
                                                f"[WARNING] Failed to record action for turn {step_turn} (event fallback)",
                                                color="YELLOW",
                                            )
                                        recorded_action_call_ids.add(dedupe_key)
                                    if (step_turn in turn_model_input or turn_actions.get(step_turn)) and actions_ok:
                                        steps_recorded_turns.add(step_turn)
                            # Update trajectory_data with this step
                            if isinstance(fetched_step, dict):
                                if "turns" not in rollout_logger.trajectory_data:
                                    rollout_logger.trajectory_data["turns"] = []
                                turns_data = rollout_logger.trajectory_data["turns"]
                                # Ensure we have enough slots (turn_num is 1-based)
                                while len(turns_data) < turn_num:
                                    turns_data.append({})
                                # Convert this step to turn format
                                turn_data_list = _steps_to_turns([fetched_step])
                                if turn_data_list:
                                    if len(turns_data) >= turn_num:
                                        turns_data[turn_num - 1] = turn_data_list[0]
                                    else:
                                        turns_data.append(turn_data_list[0])
                        _persist_execution_details()
                    except Exception as exc:
                        rollout_logger.log(
                            f"[gbox-mini-agent] Step fetch failed after step_end/turn_end: {exc}",
                            color="YELLOW",
                        )
                    _persist_execution_details()
                    if rollout_recorder is not None:
                        # Prepare turn data from step (first end_turn call - timing added later in timing loop)
                        screen_before = None
                        screen_after = None
                        thinking = None
                        text = None
                        first_action = None
                        model_specific_input = turn_model_specific_input.get(turn_num)
                        if isinstance(fetched_step, dict):
                            before_screenshot = _extract_screenshot_from_messages(fetched_step.get("messages"))
                            if before_screenshot:
                                screen_before = before_screenshot
                            actions = fetched_step.get("actions") if isinstance(fetched_step.get("actions"), list) else []
                            if actions and isinstance(actions[0], dict):
                                first_action = {
                                    "name": actions[0].get("name"),
                                    "args": actions[0].get("args") or {},
                                    "callId": actions[0].get("callId") or actions[0].get("call_id"),
                                }
                            after_screenshot = _extract_screenshot_from_actions(actions)
                            if after_screenshot:
                                screen_after = after_screenshot
                            thinking = fetched_step.get("thinking")
                            text = fetched_step.get("text")
                        else:
                            model_response = turn_model_response.get(turn_num, "")

                        # Ensure screenshot obs are recorded whenever we have URIs (e.g. when step was already in steps_recorded_turns)
                        if rollout_recorder is not None and turn_num not in screenshot_recorded_turns and (screen_before or screen_after):
                            if screen_before:
                                rollout_recorder.record_observation(
                                    turn_num=turn_num,
                                    obs_type="screenshot_before",
                                    screenshot_uri=screen_before,
                                )
                            if screen_after:
                                rollout_recorder.record_observation(
                                    turn_num=turn_num,
                                    obs_type="screenshot_after",
                                    screenshot_uri=screen_after,
                                )
                            screenshot_recorded_turns.add(turn_num)

                        if not model_response and not isinstance(fetched_step, dict):
                            rollout_logger.log(
                                f"[gbox-mini-agent] Step fetch failed for turn {turn_num}, recording turn with event data only",
                                color="YELLOW",
                            )
                            # Still call end_turn so the turn appears in the monitor (with event-captured data)
                            rollout_recorder.end_turn(
                                turn_num=turn_num,
                                model_response=turn_model_response.get(turn_num, ""),
                                reward=0.0,
                                episode_done=False,
                                screen_before=screen_before,
                                screen_after=screen_after,
                                thinking=thinking,
                                text=text,
                                action_json=first_action,
                                model_specific_input=model_specific_input,
                                turn_time=turn_time,
                                metrics=metrics,
                            )
                        else:
                            rollout_recorder.end_turn(
                                turn_num=turn_num,
                                model_response=model_response,
                                reward=0.0,
                                episode_done=False,
                                screen_before=screen_before,
                                screen_after=screen_after,
                                thinking=thinking,
                                text=text,
                                action_json=first_action,
                                model_specific_input=model_specific_input,
                                turn_time=turn_time,
                                metrics=metrics,
                            )
                elif event_type == "run_end":
                    run_ok = bool(event.get("ok", False))
                    run_reason = event.get("reason")
                    break
                elif event_type == "error":
                    message = str(event.get("message", "unknown error"))
                    errors.append(message)
                    rollout_logger.log(f"[gbox-mini-agent] Error: {message}", color="RED")

            if run_ok is not None:
                break

            status_payload = await client.get_status(run_id)
            status_value = status_payload.get("status")
            if status_value in ("completed", "failed"):
                run_ok = status_value == "completed"
                run_reason = status_payload.get("reason") or status_value
                break

            await asyncio.sleep(3.0)  # Poll every 3 seconds

        rollout_logger.trajectory_data["run_status"] = {
            "run_id": run_id,
            "status": "completed" if run_ok is True else "failed" if run_ok is False else "unknown",
            "reason": run_reason,
        }

        if rollout_recorder is not None:
            rollout_recorder.update_environment(
                status="terminated",
                termination_time=datetime.utcnow().isoformat(),
                status_message=run_reason,
            )

        try:
            turn_timings = _events_to_turn_timings(rollout_logger.trajectory_data.get("events", []))
            rollout_logger.trajectory_data["turn_timings"] = turn_timings
        except Exception as exc:
            rollout_logger.log(f"[gbox-mini-agent] Timing extraction failed: {exc}", color="YELLOW")

        step_texts: dict[int, str] = {}
        steps: list[dict] = []
        steps_error: Exception | None = None
        tokens: list[dict] = []
        steps_result, steps_error = await _retry_fetch(
            "steps",
            lambda: client.get_steps(run_id),
        )
        if isinstance(steps_result, list):
            steps = steps_result
        if steps:
            rollout_logger.trajectory_data["turns"] = _steps_to_turns(steps)
            if rollout_recorder is not None:
                for idx, step in enumerate(steps):
                    if not isinstance(step, dict):
                        continue
                    turn = idx + 1
                    if turn in steps_recorded_turns:
                        continue
                    step_texts[turn] = step.get("text") or ""
                    
                    # Record model_input from step messages (mini-agent stores TOS URLs, not base64)
                    step_messages = step.get("messages") if isinstance(step.get("messages"), list) else None
                    if step_messages:
                        model_input_payload = {"messages": step_messages, "meta": {"turn": turn}}
                        obs_id = rollout_recorder.record_observation(
                            turn_num=turn,
                            obs_type="model_input",
                            model_input=_sanitize_json_payload(model_input_payload),
                        )
                        if obs_id is not None:
                            rollout_logger.log(f"[DEBUG] Recorded model_input for turn {turn} (post-run), obs_id={obs_id}")
                        else:
                            rollout_logger.log(f"[WARNING] Failed to record model_input for turn {turn} (post-run)", color="YELLOW")
                    
                    before_screenshot = _extract_screenshot_from_messages(step.get("messages"))
                    if before_screenshot:
                        rollout_recorder.record_observation(
                            turn_num=turn,
                            obs_type="screenshot_before",
                            screenshot_uri=before_screenshot,
                        )
                    actions = step.get("actions") if isinstance(step.get("actions"), list) else []
                    for action in actions:
                        if not isinstance(action, dict):
                            continue
                        action_payload = {
                            "name": action.get("name"),
                            "args": action.get("args") or {},
                            "callId": action.get("callId") or action.get("call_id"),
                        }
                        rollout_recorder.record_action(
                            turn_num=turn,
                            action=action_payload,
                        )
                    after_screenshot = _extract_screenshot_from_actions(actions)
                    if after_screenshot:
                        rollout_recorder.record_observation(
                            turn_num=turn,
                            obs_type="screenshot_after",
                            screenshot_uri=after_screenshot,
                        )
            _persist_execution_details()
        else:
            rollout_logger.log(
                f"[gbox-mini-agent] Steps fetch failed after retries: {steps_error}",
                color="YELLOW",
            )

        if rollout_recorder is not None:
            # Persist final execution_details (env_build, turn_timings, etc.) to monitor
            _persist_execution_details()

    if max_turn_seen and rollout_logger.current_turn is not None:
        rollout_logger.end_turn(max_turn_seen)  # max_turn_seen is already 1-based

    if rollout_recorder is not None:
        rollout_recorder.update_status(
            status="completed" if run_ok else "failed",
            current_phase="cleanup",
            status_message=run_reason,
        )

    finished_extra = []
    if task_number:
        finished_extra.append("task-%s" % task_number)
    if task_id:
        finished_extra.append("task_id=%s" % (task_id[:12] + "..." if len(task_id) > 12 else task_id))
    if execution_id:
        finished_extra.append("execution_id=%s" % execution_id)
    extra_str = ", " + ", ".join(finished_extra) if finished_extra else ""
    rollout_logger.log(
        "Rollout%s finished: status=%s, turns=%s/%s, task_success=%s%s"
        % (
            log_prefix,
            "completed" if run_ok else "failed",
            max_turn_seen,
            max_turns,
            run_ok,
            extra_str,
        ),
        flush_immediately=True,
    )
    return {
        "task_success": bool(run_ok),
        "task_completed": bool(run_ok),
        "num_turns": max_turn_seen,
        "max_turns": max_turns,
        "errors": errors,
        "result_message": run_reason or "",
        "answer": answer,  # Return answer from finished action
    }
