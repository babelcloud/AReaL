"""
Minimal client for gbox-mini-agent HTTP API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class GboxMiniAgentClient:
    """Async client wrapper for gbox-mini-agent HTTP server."""

    def __init__(self, base_url: str = "http://localhost:3000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def start_run(self, payload: Dict[str, Any]) -> str:
        """Start a new run and return runId."""
        try:
            response = await self._client.post("/runs", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error(f"[gbox-mini-agent] Failed to start run: {exc}")
            raise RuntimeError(f"Failed to start gbox-mini-agent run: {exc}") from exc

        run_id = data.get("runId")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError(f"Invalid runId from gbox-mini-agent: {data}")
        return run_id

    async def get_status(self, run_id: str) -> Dict[str, Any]:
        """Fetch run status."""
        try:
            response = await self._client.get(f"/runs/{run_id}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(f"[gbox-mini-agent] Failed to get status for {run_id}: {exc}")
            raise RuntimeError(f"Failed to get gbox-mini-agent status: {exc}") from exc

    async def get_events(
        self, run_id: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch events as a list, optionally filtered by timestamp.
        
        Args:
            run_id: The run ID
            from_ts: Optional ISO-8601 timestamp or epoch-ms to filter events from (inclusive)
            to_ts: Optional ISO-8601 timestamp or epoch-ms to filter events to (exclusive)
        """
        try:
            params = {}
            if from_ts is not None:
                params["from"] = from_ts
            if to_ts is not None:
                params["to"] = to_ts
            
            response = await self._client.get(f"/runs/{run_id}/events", params=params)
            if response.status_code == 404:
                return []
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error(f"[gbox-mini-agent] Failed to get events for {run_id}: {exc}")
            raise RuntimeError(f"Failed to get gbox-mini-agent events: {exc}") from exc

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "events" in data and isinstance(data["events"], list):
            return data["events"]
        raise RuntimeError(f"Invalid events payload from gbox-mini-agent: {data}")

    async def get_trajectory(self, run_id: str) -> List[Dict[str, Any]]:
        """Fetch trajectory as a JSON array."""
        try:
            response = await self._client.get(f"/runs/{run_id}/trajectory")
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error(f"[gbox-mini-agent] Failed to get trajectory for {run_id}: {exc}")
            raise RuntimeError(f"Failed to get gbox-mini-agent trajectory: {exc}") from exc

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "trajectory" in data and isinstance(data["trajectory"], list):
            return data["trajectory"]
        raise RuntimeError(f"Invalid trajectory payload from gbox-mini-agent: {data}")

    async def get_tokens(self, run_id: str) -> List[Dict[str, Any]]:
        """Fetch tokens.jsonl as a JSON array."""
        try:
            response = await self._client.get(f"/runs/{run_id}/tokens")
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error(f"[gbox-mini-agent] Failed to get tokens for {run_id}: {exc}")
            raise RuntimeError(f"Failed to get gbox-mini-agent tokens: {exc}") from exc

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "tokens" in data and isinstance(data["tokens"], list):
            return data["tokens"]
        raise RuntimeError(f"Invalid tokens payload from gbox-mini-agent: {data}")

    async def get_steps(
        self, run_id: str, index: Optional[int] = None
    ) -> List[Dict[str, Any]] | Dict[str, Any] | None:
        """
        Fetch steps snapshot as a JSON array, or a single step by index.
        
        Args:
            run_id: The run ID
            index: Optional step index (0-based). If provided, returns a single step dict.
                   If not provided, returns all steps as a list.
        """
        try:
            if index is not None:
                # Fetch single step by index
                response = await self._client.get(f"/runs/{run_id}/steps/{index}")
                response.raise_for_status()
                return response.json()
            # Fetch all steps
            response = await self._client.get(f"/runs/{run_id}/steps")
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                return None if index is not None else []
            logger.error(f"[gbox-mini-agent] Failed to get steps for {run_id}: {exc}")
            raise RuntimeError(f"Failed to get gbox-mini-agent steps: {exc}") from exc
        except Exception as exc:
            logger.error(f"[gbox-mini-agent] Failed to get steps for {run_id}: {exc}")
            raise RuntimeError(f"Failed to get gbox-mini-agent steps: {exc}") from exc

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], list):
            return data["steps"]
        raise RuntimeError(f"Invalid steps payload from gbox-mini-agent: {data}")

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GboxMiniAgentClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
