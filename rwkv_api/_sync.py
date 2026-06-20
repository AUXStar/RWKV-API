"""Client —— RWKV-Server Task API 的同步客户端。

完全基于同步 httpx.Client，无需 asyncio。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from ._task import Task
from .exceptions import (
    ConnectionError as RWKVConnectionError,
    raise_for_status,
)
from .models import TaskInfo, TaskResponseModel

# API 基础路径
_TASKS_PREFIX = "/v1/tasks"

# 生成参数字段名集合
_GEN_PARAMS = frozenset({
    "max_tokens", "temperature", "top_k", "top_p",
    "presence_penalty", "repetition_penalty", "penalty_decay", "seed",
})


class Client:
    """RWKV-Server Task API 同步客户端。

    Args:
        base_url: 服务地址，如 ``http://localhost:8000``。
        timeout: HTTP 请求超时（秒）。
        headers: 额外的 HTTP 请求头。
        httpx_client: 自定义的 httpx.Client 实例（高级用法）。

    Usage::

        with Client("http://localhost:8000") as client:
            task = client.create("Hello, world!", max_tokens=50)
            result = task.wait()
            print(result.result)

        # 实时流式
        with Client("http://localhost:8000") as client:
            for chunk in client.create("Hello", stream=True):
                print(chunk, end="")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
        httpx_client: httpx.Client | None = None,
    ) -> None:
        if httpx_client is not None:
            self._client = httpx_client
        else:
            self._client = httpx.Client(
                base_url=base_url.rstrip("/"),
                timeout=httpx.Timeout(timeout),
                headers=headers,
            )

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """关闭底层 HTTP 连接池。"""
        self._client.close()

    # ===================================================================
    # 内部 HTTP 方法
    # ===================================================================

    def _post_json(
        self, path: str, body: dict[str, Any], *, response_model: type[Any] | None = None,
    ) -> Any:
        """发送 POST 请求，解析 JSON 响应。"""
        try:
            resp = self._client.post(path, json=body)
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e
        raise_for_status(resp.status_code, resp.text)
        if response_model is not None:
            return response_model.model_validate(resp.json())
        return resp.json()

    def _get_json(
        self, path: str, *, response_model: type[Any] | None = None,
    ) -> Any:
        """发送 GET 请求，解析 JSON 响应。"""
        try:
            resp = self._client.get(path)
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e
        raise_for_status(resp.status_code, resp.text)
        if response_model is not None:
            return response_model.model_validate(resp.json())
        return resp.json()

    def _post_no_content(
        self, path: str, body: dict[str, Any], *, params: dict[str, str] | None = None,
    ) -> None:
        """发送 POST 请求，不解析响应体。"""
        try:
            resp = self._client.post(path, json=body, params=params)
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e
        raise_for_status(resp.status_code, resp.text)

    def _stream_sse_sync(self, method: str, endpoint: str, body: dict[str, Any]) -> Iterator[str]:
        """同步 SSE 流式请求，逐 chunk yield。"""
        try:
            with self._client.stream(method, endpoint, json=body) as resp:
                raise_for_status(resp.status_code, "")
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if "prefill_time" in parsed and "data" not in parsed:
                        continue
                    if "data" in parsed:
                        yield parsed["data"]
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e

    # ===================================================================
    # 流式 API
    # ===================================================================

    def create_stream(
        self,
        prompt: str | list[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        penalty_decay: float | None = None,
        seed: int | None = None,
        persistent: bool = False,
    ) -> Iterator[str]:
        """创建任务并实时流式返回生成内容（同步）。"""
        body: dict[str, Any] = {"prompt": prompt, "stream": True}
        for key in _GEN_PARAMS:
            value = locals()[key]
            if value is not None:
                body[key] = value
        endpoint = f"{_TASKS_PREFIX}/create" if persistent else f"{_TASKS_PREFIX}/tmp"
        yield from self._stream_sse_sync("POST", endpoint, body)

    def fim_stream(
        self,
        prefix: str,
        suffix: str = "",
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        penalty_decay: float | None = None,
        seed: int | None = None,
    ) -> Iterator[str]:
        """FIM 实时流式返回生成内容（同步）。"""
        body: dict[str, Any] = {"prefix": prefix, "suffix": suffix, "stream": True}
        for key in _GEN_PARAMS:
            value = locals()[key]
            if value is not None:
                body[key] = value
        yield from self._stream_sse_sync("POST", f"{_TASKS_PREFIX}/fim", body)

    # ===================================================================
    # Task API（stream=True 时自动分发到流式）
    # ===================================================================

    def create(
        self,
        prompt: str | list[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        penalty_decay: float | None = None,
        stream: bool = False,
        seed: int | None = None,
        persistent: bool = False,
    ) -> Task | Iterator[str]:
        """创建任务（同步版本）。

        stream=True 时返回 Iterator[str]（实时流式）；
        stream=False 时返回 Task 对象。
        """
        if stream is True:
            return self.create_stream(
                prompt, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                seed=seed, persistent=persistent,
            )
        body: dict[str, Any] = {"prompt": prompt, "stream": False}
        for key in _GEN_PARAMS:
            value = locals()[key]
            if value is not None:
                body[key] = value
        endpoint = f"{_TASKS_PREFIX}/create" if persistent else f"{_TASKS_PREFIX}/tmp"
        resp_model = self._post_json(endpoint, body, response_model=TaskResponseModel)
        return Task(self, resp_model.task_id, response=resp_model)

    def create_tmp(
        self, prompt: str | list[int] | None = None, /, **kwargs: Any
    ) -> Task | Iterator[str]:
        """创建临时任务。"""
        if prompt is not None:
            kwargs["prompt"] = prompt
        return self.create(**kwargs, persistent=False)

    def create_persistent(
        self, prompt: str | list[int] | None = None, /, **kwargs: Any
    ) -> Task | Iterator[str]:
        """创建持久化任务。"""
        if prompt is not None:
            kwargs["prompt"] = prompt
        return self.create(**kwargs, persistent=True)

    def fim(
        self,
        prefix: str,
        suffix: str = "",
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        penalty_decay: float | None = None,
        stream: bool = False,
        seed: int | None = None,
    ) -> Task | Iterator[str]:
        """Fill In Middle（同步版本）。

        stream=True 时返回 Iterator[str]（实时流式）；
        stream=False 时返回 Task 对象。
        """
        if stream is True:
            return self.fim_stream(
                prefix, suffix, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                seed=seed,
            )
        body: dict[str, Any] = {"prefix": prefix, "suffix": suffix, "stream": False}
        for key in _GEN_PARAMS:
            value = locals()[key]
            if value is not None:
                body[key] = value
        resp_model = self._post_json(f"{_TASKS_PREFIX}/fim", body, response_model=TaskResponseModel)
        return Task(self, resp_model.task_id, response=resp_model)

    # ===================================================================
    # 任务管理
    # ===================================================================

    def list_tasks(self) -> dict[str, Any]:
        """列出所有任务。"""
        return self._get_json(f"{_TASKS_PREFIX}/list")

    def get_task_result(self, task_id: str) -> TaskResponseModel:
        """获取任务结果。"""
        return self._get_json(f"{_TASKS_PREFIX}/{task_id}/get_result", response_model=TaskResponseModel)

    def get_task_status(self, task_id: str) -> TaskInfo:
        """获取任务状态。"""
        return self._get_json(f"{_TASKS_PREFIX}/{task_id}/status", response_model=TaskInfo)

    def stop_task(self, task_id: str) -> None:
        """停止任务。"""
        self._post_no_content(f"{_TASKS_PREFIX}/{task_id}/stop", {})

    def delete_task(self, task_id: str, *, force: bool = False) -> None:
        """删除任务。"""
        self._post_no_content(
            f"{_TASKS_PREFIX}/{task_id}/delete", {},
            params={"force": str(force).lower()},
        )

    def fork_task(self, task_id: str, **kwargs: Any) -> Task:
        """Fork 任务。"""
        body = self._build_update_body(kwargs)
        resp_model = self._post_json(
            f"{_TASKS_PREFIX}/{task_id}/fork", body, response_model=TaskResponseModel,
        )
        return Task(self, resp_model.task_id, response=resp_model)

    def continue_task(self, task_id: str, **kwargs: Any) -> Task:
        """继续任务。"""
        body = self._build_update_body(kwargs)
        resp_model = self._post_json(
            f"{_TASKS_PREFIX}/{task_id}/continue", body, response_model=TaskResponseModel,
        )
        return Task(self, resp_model.task_id, response=resp_model)

    def as_template(self, task_id: str) -> Task:
        """转为模板。"""
        resp_model = self._post_json(
            f"{_TASKS_PREFIX}/{task_id}/as_template", {}, response_model=TaskResponseModel,
        )
        return Task(self, resp_model.task_id, response=resp_model)

    def stream_task(self, task_id: str) -> Iterator[str]:
        """订阅已创建任务的实时流式输出。

        如果任务已完成，一次性返回完整结果。
        如果任务还在运行，从当前位置开始流式。
        """
        yield from self._stream_sse_sync("GET", f"{_TASKS_PREFIX}/{task_id}/stream", {})

    @staticmethod
    def _build_update_body(overrides: dict[str, Any]) -> dict[str, Any]:
        """从关键字参数构建更新请求体。"""
        body: dict[str, Any] = {"stream": False}
        for key in ("prompt", *_GEN_PARAMS, "stream"):
            if key in overrides:
                body[key] = overrides[key]
        return body
