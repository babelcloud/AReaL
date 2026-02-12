"""Extract token_ids and logprobs from gbox-mini-agent events (no tinker dependency)."""

from __future__ import annotations

from typing import Any


def _event_turn(event: dict) -> int:
    """Turn index (1-based) from event. gbox uses step (0-based) or turn."""
    turn = event.get("turn")
    if isinstance(turn, int):
        return turn
    step = event.get("step", 0)
    return int(step) + 1


def extract_model_inputs_from_events(events: list[dict]) -> dict[int, dict]:
    """Extract model_call modelSpecificInput by turn (1-based)."""
    inputs: dict[int, dict] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "model_call":
            continue
        turn = _event_turn(event)
        model_input = event.get("modelSpecificInput") or event.get("modelInput")
        if isinstance(model_input, dict):
            inputs[turn] = model_input
    return inputs


def extract_model_outputs_from_events(events: list[dict]) -> dict[int, dict]:
    """Extract model_output events by turn (1-based)."""
    outputs: dict[int, dict] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "model_output":
            continue
        turn = _event_turn(event)
        model_output = event.get("modelOutput")
        if isinstance(model_output, dict):
            outputs[turn] = model_output
    return outputs


def extract_tokens_and_logprobs(model_output: dict) -> tuple[list[int], list[float]] | None:
    """Extract token_ids and logprobs from OpenAI-compatible model_output.

    Expects choices[0].token_ids and choices[0].logprobs.content[].logprob.
    """
    try:
        choices = model_output.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list):
            return None
        logprob_entries = (choice.get("logprobs") or {}).get("content", [])
        if not isinstance(logprob_entries, list):
            return None
        logprobs: list[float] = []
        for entry in logprob_entries:
            if isinstance(entry, dict) and "logprob" in entry:
                logprobs.append(float(entry["logprob"]))
        if len(logprobs) != len(token_ids):
            return None
        return [int(t) for t in token_ids], logprobs
    except Exception:
        return None
