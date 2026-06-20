"""AsyncClient —— RWKV-Server Task API 的异步客户端

基于 httpx.AsyncHTTPClient 实现所有 API 调用，
支持流式 SSE 响应和完整的 Task 对象生命周期管理
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from . import _task
from .exceptions import (
    ConnectionError as RWKVConnectionError,
    RWKVError,
    raise_for_status,
)
from .models import (
    TaskInfo,
    TaskResponseModel,
)

# API 基础路径
_TASKS_PREFIX = "/v1/tasks"

# 生成参数字段名集合，用于构建请求体
_GEN_PARAMS = frozenset({
    "max_tokens", "temperature", "top_k", "top_p",
    "presence_penalty", "repetition_penalty", "penalty_decay", "seed",
})


class AsyncClient:
    """RWKV-Server Task API 异步客户端

    Args:
        base_url: 服务地址，如 ``http://localhost:8000``
        timeout: HTTP 请求超时（秒）
        headers: 额外的 HTTP 请求头
        httpx_client: 自定义的 httpx.AsyncClient 实例（高级用法）

    Usage::

        async with AsyncClient("http://localhost:8000") as client:
            task = await client.create("Hello, world!", max_tokens=50)
            result = await task.wait()
            print(result.result)

        # 实时流式
        async with AsyncClient("http://localhost:8000") as client:
            async for chunk in await client.create("Hello", stream=True):
                print(chunk, end="")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        if httpx_client is not None:
            self._client = httpx_client
        else:
            self._client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"),
                timeout=httpx.Timeout(timeout),
                headers=headers,
            )

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """关闭底层 HTTP 连接池"""
        await self._client.aclose()

    # ===================================================================
    # 流式 API
    # ===================================================================

    async def _stream_sse(
        self,
        method: str,
        endpoint: str,
        body: dict[str, Any],
    ) -> AsyncIterator[str]:
        """通用 SSE 流式请求，逐 chunk yield"""
        try:
            async with self._client.stream(method, endpoint, json=body) as resp:
                raise_for_status(resp.status_code, "")
                async for line in resp.aiter_lines():
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
                    # 跳过首事件（仅含 prefill_time，无 data）
                    if "prefill_time" in parsed and "data" not in parsed:
                        continue
                    if "data" in parsed:
                        yield parsed["data"]
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e

    async def create_stream(
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
    ) -> AsyncIterator[str]:
        """创建任务并实时流式返回生成内容"""
        body: dict[str, Any] = {"prompt": prompt, "stream": True}
        self._inject_gen_params(body, max_tokens=max_tokens, temperature=temperature,
                                 top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                                 repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                                 seed=seed)
        endpoint = f"{_TASKS_PREFIX}/create" if persistent else f"{_TASKS_PREFIX}/tmp"
        async for chunk in self._stream_sse("POST", endpoint, body):
            yield chunk

    async def fim_stream(
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
    ) -> AsyncIterator[str]:
        """FIM 实时流式返回生成内容"""
        body: dict[str, Any] = {"prefix": prefix, "suffix": suffix, "stream": True}
        self._inject_gen_params(body, max_tokens=max_tokens, temperature=temperature,
                                 top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                                 repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                                 seed=seed)
        async for chunk in self._stream_sse("POST", f"{_TASKS_PREFIX}/fim", body):
            yield chunk

    # ===================================================================
    # Task API
    # ===================================================================

    async def create(
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
    ) -> _task.AsyncTask | AsyncIterator[str]:
        """创建任务

        Args:
            prompt: 提示文本或 token id 列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样参数
            top_p: Top-P 采样参数
            presence_penalty: 存在惩罚
            repetition_penalty: 重复惩罚
            penalty_decay: 惩罚衰减
            stream: 是否使用流式响应True 时返回异步生成器
            seed: 随机种子
            persistent: 是否创建持久化任务（True 使用 /create，False 使用 /tmp）

        Returns:
            stream=True 时返回 AsyncIterator[str]；
            stream=False 时返回 AsyncTask 对象
        """
        if stream is True:
            return await self.create_stream(
                prompt, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                seed=seed, persistent=persistent,
            )

        body: dict[str, Any] = {"prompt": prompt, "stream": False}
        self._inject_gen_params(body, max_tokens=max_tokens, temperature=temperature,
                                 top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                                 repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                                 seed=seed)
        endpoint = f"{_TASKS_PREFIX}/create" if persistent else f"{_TASKS_PREFIX}/tmp"
        resp_model = await self._post_json(endpoint, body, response_model=TaskResponseModel)
        return _task.AsyncTask(self, resp_model.task_id, response=resp_model)

    async def create_tmp(
        self, prompt: str | list[int] | None = None, /, **kwargs: Any
    ) -> _task.AsyncTask | AsyncIterator[str]:
        """创建临时任务（等同于 create(..., persistent=False)）"""
        if prompt is not None:
            kwargs["prompt"] = prompt
        return await self.create(**kwargs, persistent=False)

    async def create_persistent(
        self, prompt: str | list[int] | None = None, /, **kwargs: Any
    ) -> _task.AsyncTask | AsyncIterator[str]:
        """创建持久化任务（等同于 create(..., persistent=True)）"""
        if prompt is not None:
            kwargs["prompt"] = prompt
        return await self.create(**kwargs, persistent=True)

    async def fim(
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
    ) -> _task.AsyncTask | AsyncIterator[str]:
        """Fill In Middle —— 在 prefix 和 suffix 之间生成文本

        Args:
            prefix: 前缀文本
            suffix: 后缀文本
            stream: 是否使用流式响应True 时返回异步生成器
            其余参数同 create()

        Returns:
            stream=True 时返回 AsyncIterator[str]；
            stream=False 时返回 AsyncTask 对象
        """
        if stream is True:
            return await self.fim_stream(
                prefix, suffix, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                seed=seed,
            )

        body: dict[str, Any] = {"prefix": prefix, "suffix": suffix, "stream": False}
        self._inject_gen_params(body, max_tokens=max_tokens, temperature=temperature,
                                 top_k=top_k, top_p=top_p, presence_penalty=presence_penalty,
                                 repetition_penalty=repetition_penalty, penalty_decay=penalty_decay,
                                 seed=seed)
        resp_model = await self._post_json(f"{_TASKS_PREFIX}/fim", body, response_model=TaskResponseModel)
        return _task.AsyncTask(self, resp_model.task_id, response=resp_model)

    async def get_task_result(self, task_id: str) -> TaskResponseModel:
        """获取任务结果"""
        return await self._get_json(f"{_TASKS_PREFIX}/{task_id}/get_result", response_model=TaskResponseModel)

    async def get_task_status(self, task_id: str) -> TaskInfo:
        """获取任务状态"""
        return await self._get_json(f"{_TASKS_PREFIX}/{task_id}/status", response_model=TaskInfo)

    async def fork_task(self, task_id: str, **overrides: Any) -> _task.AsyncTask:
        """Fork 任务"""
        body = self._build_update_body(overrides)
        resp_model = await self._post_json(
            f"{_TASKS_PREFIX}/{task_id}/fork", body, response_model=TaskResponseModel,
        )
        return _task.AsyncTask(self, resp_model.task_id, response=resp_model)

    async def continue_task(self, task_id: str, **overrides: Any) -> _task.AsyncTask:
        """继续生成"""
        body = self._build_update_body(overrides)
        resp_model = await self._post_json(
            f"{_TASKS_PREFIX}/{task_id}/continue", body, response_model=TaskResponseModel,
        )
        return _task.AsyncTask(self, resp_model.task_id, response=resp_model)

    async def stop_task(self, task_id: str) -> None:
        """停止任务"""
        await self._post_no_content(f"{_TASKS_PREFIX}/{task_id}/stop", {})

    async def as_template(self, task_id: str) -> _task.AsyncTask:
        """将任务转为模板"""
        resp_model = await self._post_json(
            f"{_TASKS_PREFIX}/{task_id}/as_template", {}, response_model=TaskResponseModel,
        )
        return _task.AsyncTask(self, resp_model.task_id, response=resp_model)

    async def stream_task(self, task_id: str) -> AsyncIterator[str]:
        """订阅已创建任务的实时流式输出"""
        async for chunk in self._stream_sse("GET", f"{_TASKS_PREFIX}/{task_id}/stream", {}):
            yield chunk

    async def delete_task(self, task_id: str, *, force: bool = False) -> None:
        """删除任务"""
        await self._post_no_content(
            f"{_TASKS_PREFIX}/{task_id}/delete", {},
            params={"force": str(force).lower()},
        )

    async def list_tasks(self) -> dict[str, Any]:
        """列出所有任务"""
        return await self._get_json(f"{_TASKS_PREFIX}/list")

    # ===================================================================
    # 内部 HTTP 方法
    # ===================================================================

    async def _post_json(
        self, path: str, body: dict[str, Any], *, response_model: type[Any] | None = None,
    ) -> Any:
        """发送 POST 请求，解析 JSON 响应"""
        try:
            resp = await self._client.post(path, json=body)
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e
        raise_for_status(resp.status_code, resp.text)
        if response_model is not None:
            return response_model.model_validate(resp.json())
        return resp.json()

    async def _get_json(
        self, path: str, *, response_model: type[Any] | None = None,
    ) -> Any:
        """发送 GET 请求，解析 JSON 响应"""
        try:
            resp = await self._client.get(path)
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e
        raise_for_status(resp.status_code, resp.text)
        if response_model is not None:
            return response_model.model_validate(resp.json())
        return resp.json()

    async def _post_no_content(
        self, path: str, body: dict[str, Any], *, params: dict[str, str] | None = None,
    ) -> None:
        """发送 POST 请求，不解析响应体"""
        try:
            resp = await self._client.post(path, json=body, params=params)
        except httpx.ConnectError as e:
            raise RWKVConnectionError(f"无法连接到服务器: {e}") from e
        except httpx.TimeoutException as e:
            from .exceptions import TimeoutError as _Timeout
            raise _Timeout(f"请求超时: {e}") from e
        raise_for_status(resp.status_code, resp.text)

    # ===================================================================
    # 内部辅助
    # ===================================================================

    @staticmethod
    def _inject_gen_params(body: dict[str, Any], **params: Any) -> None:
        """将非 None 的生成参数注入请求体"""
        for key, value in params.items():
            if value is not None:
                body[key] = value

    @staticmethod
    def _build_update_body(overrides: dict[str, Any]) -> dict[str, Any]:
        """从关键字参数构建更新请求体"""
        body: dict[str, Any] = {"stream": False}
        for key in ("prompt", *_GEN_PARAMS, "stream"):
            if key in overrides:
                body[key] = overrides[key]
        return body
