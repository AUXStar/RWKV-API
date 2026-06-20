"""AsyncTask / Task 集成测试 —— 使用真实 RWKV-Server

覆盖 wait、stop、delete、fork、continue_、as_template 等方法
"""
import pytest
from rwkv_api import AsyncClient
from rwkv_api._task import AsyncTask
from rwkv_api.exceptions import TimeoutError as RWKVTimeoutError


class TestAsyncTaskWait:
    @pytest.mark.asyncio
    async def test_wait_finished(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=5)
        result = await task.wait(timeout=30)
        assert result.finished
        assert isinstance(result.result, str)

    @pytest.mark.asyncio
    async def test_wait_timeout(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=500)
        with pytest.raises(RWKVTimeoutError):
            await task.wait(timeout=0.5, poll_interval=0.1)


class TestAsyncTaskOperations:
    @pytest.mark.asyncio
    async def test_stop_and_delete(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=100)
        await task.stop()
        await task.delete()

    @pytest.mark.asyncio
    async def test_fork(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=5)
        await task.wait(timeout=30)
        forked = await task.fork(prompt="World", max_tokens=5)
        assert forked.task_id != task.task_id

    @pytest.mark.asyncio
    async def test_continue(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=5)
        await task.wait(timeout=30)
        continued = await task.continue_(max_tokens=5)
        assert continued.task_id == task.task_id

    @pytest.mark.asyncio
    async def test_as_template(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=5)
        # fork 一个 task 到模板（as_template 要求 TASK_ 开头）
        forked = await task.fork(prompt="Hello", max_tokens=5)
        await forked.wait(timeout=30)
        tpl = await forked.as_template()
        assert tpl.task_id.startswith("_TMPL_")