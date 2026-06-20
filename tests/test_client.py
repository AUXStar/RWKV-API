"""AsyncClient 集成测试 —— 使用真实 RWKV-Server"""
import pytest
from rwkv_api import AsyncClient
from rwkv_api._task import AsyncTask
from rwkv_api.exceptions import RWKVValidationError

BASE_URL = "http://localhost:8000"


class TestCreate:
    async def _create_task(self, client: AsyncClient, **kw) -> AsyncTask:
        task = await client.create("def hello():\n    ", max_tokens=10, **kw)
        assert isinstance(task, AsyncTask)
        assert task.task_id
        return task

    @pytest.mark.asyncio
    async def test_create_tmp(self, server_alive, async_client):
        task = await self._create_task(async_client)
        assert task.task_id.startswith("TMP_")

    @pytest.mark.asyncio
    async def test_create_persistent(self, server_alive, async_client):
        task = await self._create_task(async_client, persistent=True)
        assert task.task_id.startswith("TASK_")

    @pytest.mark.asyncio
    async def test_create_tmp_alias(self, server_alive, async_client):
        task = await async_client.create_tmp("def hello():\n    ", max_tokens=10)
        assert task.task_id.startswith("TMP_")

    @pytest.mark.asyncio
    async def test_create_with_seed(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=5, seed=12345)
        assert task.task_id

    @pytest.mark.asyncio
    async def test_create_validation_error(self, server_alive, async_client):
        with pytest.raises(RWKVValidationError):
            await async_client.create("Hello", max_tokens=-1)


class TestFIM:
    @pytest.mark.asyncio
    async def test_fim_success(self, server_alive, async_client):
        task = await async_client.fim(
            prefix="def add(a, b):\n    return ",
            suffix="\n    # end",
            max_tokens=10,
        )
        assert isinstance(task, AsyncTask)
        assert task.task_id
        # wait and verify result
        result = await task.wait(timeout=30)
        assert result.finished
        assert isinstance(result.result, str)


class TestListTasks:
    @pytest.mark.asyncio
    async def test_list_tasks(self, server_alive, async_client):
        result = await async_client.list_tasks()
        assert "total_count" in result
        assert "cpu_tasks" in result
        assert isinstance(result["total_count"], int)
        assert isinstance(result["cpu_tasks"], list)


class TestTaskOperations:
    @pytest.mark.asyncio
    async def test_get_result(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=10)
        await task.wait(timeout=30)
        assert task.finished
        assert isinstance(task.result, str)
        assert len(task.result) > 0

    @pytest.mark.asyncio
    async def test_get_status(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=10)
        info = await task.async_status()
        assert info.task_id == task.task_id
        assert info.status in (0, 1, 2, 3)

    @pytest.mark.asyncio
    async def test_stop_and_delete(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=100)
        await task.stop()
        await task.delete()

    @pytest.mark.asyncio
    async def test_fork_task(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=5)
        await task.wait(timeout=30)
        forked = await task.fork(prompt="World", max_tokens=5)
        assert isinstance(forked, AsyncTask)
        assert forked.task_id != task.task_id

    @pytest.mark.asyncio
    async def test_continue_task(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=5)
        await task.wait(timeout=30)
        continued = await task.continue_(max_tokens=5)
        assert continued.task_id == task.task_id