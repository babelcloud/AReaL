from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _format_http_error(error: httpx.HTTPStatusError) -> str:
    response = error.response
    if response is None:
        return str(error)
    try:
        detail = response.json()
    except Exception:
        detail = response.text
    return f"{error} | response={detail}"


class GenvHttpClient:
    """Synchronous HTTP client for the genv high-level API."""

    def __init__(self, *, gym_id: str, base_url: Optional[str] = None):
        self.gym_id = str(gym_id)
        self.base_url = _normalize_base_url(base_url or "http://localhost:5010")

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=600.0,
                write=30.0,
                pool=30.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
            ),
        )

    def list_tasks(self) -> list[dict[str, Any]]:
        response = self.client.get(
            f"/api/v1/{self.gym_id}/tasks",
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        tasks = payload.get("tasks") if isinstance(payload, dict) else payload
        return tasks if isinstance(tasks, list) else []

    def get_task(self, task_id: str) -> dict[str, Any]:
        response = self.client.get(
            f"/api/v1/{self.gym_id}/tasks/{task_id}",
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def create_execution(
        self, *, task_identifier: str, options: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {}
        if options is not None:
            request_body["options"] = options
        response = self.client.post(
            f"/api/v1/{self.gym_id}/tasks/{task_identifier}",
            json=request_body,
            timeout=120.0,
        )
        if not response.is_success:
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", "Unknown error")
                logger.error(
                    "genv create_execution failed for task '%s': %s (status: %s)",
                    task_identifier,
                    error_detail,
                    response.status_code,
                )
            except Exception:
                logger.error(
                    "genv create_execution failed for task '%s': %s (status: %s)",
                    task_identifier,
                    response.text,
                    response.status_code,
                )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def evaluate_execution(
        self,
        *,
        execution_id: str,
        execution_data: Optional[dict[str, Any]] = None,
        answer_text: Optional[str] = None,
        context_text: Optional[str] = None,
        facts: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {}
        if execution_data is not None:
            request_body["execution_data"] = execution_data
        if answer_text is not None:
            request_body["answer_text"] = answer_text
        if context_text is not None:
            request_body["context_text"] = context_text
        if facts is not None:
            request_body["facts"] = facts

        response = self.client.post(
            f"/api/v1/{self.gym_id}/executions/{execution_id}/evaluate",
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def terminate_execution(self, *, execution_id: str) -> dict[str, Any]:
        response = self.client.post(
            f"/api/v1/{self.gym_id}/executions/{execution_id}/terminate"
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def close(self) -> None:
        self.client.close()


class AsyncGenvHttpClient:
    """Async HTTP client for the genv high-level API."""

    def __init__(self, *, gym_id: str, base_url: Optional[str] = None):
        self.gym_id = str(gym_id)
        self.base_url = _normalize_base_url(base_url or "http://localhost:5010")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=600.0,
                write=30.0,
                pool=30.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
            ),
        )

    async def list_tasks(self) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/api/v1/{self.gym_id}/tasks",
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        tasks = payload.get("tasks") if isinstance(payload, dict) else payload
        return tasks if isinstance(tasks, list) else []

    async def create_execution(
        self, *, task_identifier: str, options: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {}
        if options is not None:
            request_body["options"] = options
        response = await self.client.post(
            f"/api/v1/{self.gym_id}/tasks/{task_identifier}",
            json=request_body,
            timeout=300.0,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                _format_http_error(exc),
                request=exc.request,
                response=exc.response,
            ) from exc
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def evaluate_execution(
        self,
        *,
        execution_id: str,
        execution_data: Optional[dict[str, Any]] = None,
        answer_text: Optional[str] = None,
        context_text: Optional[str] = None,
        facts: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {}
        if execution_data is not None:
            request_body["execution_data"] = execution_data
        if answer_text is not None:
            request_body["answer_text"] = answer_text
        if context_text is not None:
            request_body["context_text"] = context_text
        if facts is not None:
            request_body["facts"] = facts

        response = await self.client.post(
            f"/api/v1/{self.gym_id}/executions/{execution_id}/evaluate",
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def terminate_execution(self, *, execution_id: str) -> dict[str, Any]:
        response = await self.client.post(
            f"/api/v1/{self.gym_id}/executions/{execution_id}/terminate"
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def close(self) -> None:
        await self.client.aclose()
