"""RWKV-API Task 对象"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from .models import Status, TaskInfo, TaskResponseModel

if TYPE_CHECKING:
    from ._client import AsyncClient
    from ._sync import Client


class AsyncTask:
    """异步任务对象，封装任务状态和操作"""

    def __init__(
        self,
        client: AsyncClient,
        task_id: str,
        *,
        response: TaskResponseModel | None = None,
    ) -> None:
        self._client = client
        self.task_id = task_id

        self.result: str = ""
        self.prefill_time: float = 0.0
        self.gen_time: float = 0.0
        self.speed: float = 0.0
        self.finished: bool = False

        if response is not None:
            self.result = response.result
            self.prefill_time = response.prefill_time
            self.gen_time = response.gen_time
            self.speed = response.speed
            self.finished = response.finished

    def _update_from_response(self, resp: TaskResponseModel) -> None:
        """从响应更新任务状态"""
        self.result = resp.result
        self.prefill_time = resp.prefill_time
        self.gen_time = resp.gen_time
        self.speed = resp.speed
        self.finished = resp.finished

    @property
    def status(self) -> Status:
        """快捷获取任务状态枚举（同步属性，内部发异步请求）"""
        # 由子类 Task 覆盖
        raise NotImplementedError("AsyncTask.status 需要通过 async_status 获取")

    async def async_status(self) -> TaskInfo:
        """获取任务状态信息"""
        return await self._client.get_task_status(self.task_id)

    async def async_get_result(self) -> TaskResponseModel:
        """获取最新结果并更新属性"""
        resp = await self._client.get_task_result(self.task_id)
        self._update_from_response(resp)
        return resp

    async def wait(
        self,
        *,
        poll_interval: float = 0.5,
        timeout: float | None = None,
    ) -> AsyncTask:
        """轮询等待任务完成"""
        import asyncio
        from .exceptions import TimeoutError as _Timeout

        elapsed = 0.0
        while True:
            resp = await self._client.get_task_result(self.task_id)
            self._update_from_response(resp)
            if self.finished:
                return self
            if timeout is not None and elapsed >= timeout:
                raise _Timeout(f"等待任务 {self.task_id} 超时（{timeout}s）")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    async def stop(self) -> None:
        """停止任务"""
        await self._client.stop_task(self.task_id)

    async def delete(self, *, force: bool = False) -> None:
        """删除任务"""
        await self._client.delete_task(self.task_id, force=force)

    async def fork(self, **overrides: Any) -> AsyncTask:
        """Fork 当前任务"""
        return await self._client.fork_task(self.task_id, **overrides)

    async def continue_(self, **overrides: Any) -> AsyncTask:
        """继续生成"""
        return await self._client.continue_task(self.task_id, **overrides)

    async def as_template(self) -> AsyncTask:
        """转为模板"""
        return await self._client.as_template(self.task_id)

    def stream(self, pos: int = 0) -> AsyncIterator[str]:
        """订阅此任务的实时流式输出

        Args:
            pos: 从第几个 token 开始（默认 0）
        """
        return self._client.stream_task(self.task_id, pos=pos)


class Task:
    """同步任务对象，封装任务状态和操作"""

    def __init__(
        self,
        client: Client,
        task_id: str,
        *,
        response: TaskResponseModel | None = None,
    ) -> None:
        self._client = client
        self.task_id = task_id

        self.result: str = ""
        self.prefill_time: float = 0.0
        self.gen_time: float = 0.0
        self.speed: float = 0.0
        self.finished: bool = False

        if response is not None:
            self.result = response.result
            self.prefill_time = response.prefill_time
            self.gen_time = response.gen_time
            self.speed = response.speed
            self.finished = response.finished

    def _update_from_response(self, resp: TaskResponseModel) -> None:
        """从响应更新任务状态"""
        self.result = resp.result
        self.prefill_time = resp.prefill_time
        self.gen_time = resp.gen_time
        self.speed = resp.speed
        self.finished = resp.finished

    @property
    def status(self) -> Status:
        """快捷获取任务状态枚举"""
        info = self._client.get_task_status(self.task_id)
        return info.status

    def wait(
        self,
        *,
        poll_interval: float = 0.5,
        timeout: float | None = None,
    ) -> Task:
        """轮询等待任务完成（同步版本）"""
        import time
        from .exceptions import TimeoutError as _Timeout

        elapsed = 0.0
        while True:
            resp = self._client.get_task_result(self.task_id)
            self._update_from_response(resp)
            if self.finished:
                return self
            if timeout is not None and elapsed >= timeout:
                raise _Timeout(f"等待任务 {self.task_id} 超时（{timeout}s）")
            time.sleep(poll_interval)
            elapsed += poll_interval

    def stop(self) -> None:
        """停止任务"""
        self._client.stop_task(self.task_id)

    def delete(self, *, force: bool = False) -> None:
        """删除任务"""
        self._client.delete_task(self.task_id, force=force)

    def fork(self, **overrides: Any) -> Task:
        """Fork 当前任务"""
        return self._client.fork_task(self.task_id, **overrides)

    def continue_(self, **overrides: Any) -> Task:
        """继续生成"""
        return self._client.continue_task(self.task_id, **overrides)

    def as_template(self) -> Task:
        """转为模板"""
        return self._client.as_template(self.task_id)

    def stream(self) -> Iterator[str]:
        """订阅此任务的实时流式输出

        如果任务已完成，一次性返回完整结果
        如果任务还在运行，从当前位置开始流式
        """
        return self._client.stream_task(self.task_id)
