"""AsyncClient mock 测试"""
import pytest
import httpx
from rwkv_api import AsyncClient, AsyncTask
from rwkv_api.exceptions import RWKVValidationError, TaskNotFoundError, RWKVServerError


@pytest.fixture
def client():
    return AsyncClient("http://localhost:8000")


class TestCreate:
    """创建任务测试"""

    @pytest.mark.asyncio
    async def test_create_tmp_success(self, client, respx_mock):
        """默认 create 走 /tmp"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_abc123",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.create("Hello", max_tokens=10)
        assert isinstance(task, AsyncTask)
        assert task.task_id == "TASK_abc123"
        assert not task.finished

    @pytest.mark.asyncio
    async def test_create_persistent_success(self, client, respx_mock):
        """persistent=True 走 /create"""
        respx_mock.post("http://localhost:8000/v1/tasks/create").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_persistent",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.create("Hello", max_tokens=10, persistent=True)
        assert task.task_id == "TASK_persistent"

    @pytest.mark.asyncio
    async def test_create_tmp_alias(self, client, respx_mock):
        """create_tmp 是 create(persistent=False) 的别名"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TMP_abc123",
                "result": "",
                "prefill_time": 0.05,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.create_tmp("Hi", max_tokens=5)
        assert task.task_id == "TMP_abc123"

    @pytest.mark.asyncio
    async def test_create_validation_error(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(422, json={"detail": "Invalid max_tokens"})
        )
        with pytest.raises(RWKVValidationError):
            await client.create("Hello", max_tokens=-1)

    @pytest.mark.asyncio
    async def test_create_not_found(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(404)
        )
        with pytest.raises(TaskNotFoundError):
            await client.create("Hello")

    @pytest.mark.asyncio
    async def test_create_server_error(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(500, json={"detail": "Internal error"})
        )
        with pytest.raises(RWKVServerError):
            await client.create("Hello")


class TestFIM:
    """FIM 测试"""

    @pytest.mark.asyncio
    async def test_fim_success(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/fim").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TMP_fim123",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.fim(prefix="def foo():", suffix="\n    return", max_tokens=20)
        assert task.task_id == "TMP_fim123"


class TestListTasks:
    """列出任务测试"""

    @pytest.mark.asyncio
    async def test_list_tasks(self, client, respx_mock):
        respx_mock.get("http://localhost:8000/v1/tasks/list").mock(
            return_value=httpx.Response(200, json={
                "cpu_cache_count": 2,
                "database_count": 5,
                "total_count": 7,
                "cpu_tasks": ["TASK_1", "TASK_2"],
                "db_tasks": ["TASK_3"],
            })
        )
        result = await client.list_tasks()
        assert result["total_count"] == 7
        assert len(result["cpu_tasks"]) == 2


class TestTaskOperations:
    """任务操作测试"""

    @pytest.mark.asyncio
    async def test_get_result(self, client, respx_mock):
        respx_mock.get("http://localhost:8000/v1/tasks/TASK_1/get_result").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_1",
                "result": "Hello world",
                "prefill_time": 0.1,
                "gen_time": 0.5,
                "speed": 50.0,
                "finished": True,
            })
        )
        result = await client.get_task_result("TASK_1")
        assert result.result == "Hello world"
        assert result.finished

    @pytest.mark.asyncio
    async def test_get_status(self, client, respx_mock):
        respx_mock.get("http://localhost:8000/v1/tasks/TASK_1/status").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_1",
                "generated_buf": 42,
                "status": "RUNNING",
            })
        )
        info = await client.get_task_status("TASK_1")
        assert info.task_id == "TASK_1"
        assert info.status == "RUNNING"

    @pytest.mark.asyncio
    async def test_stop_task(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_1/stop").mock(
            return_value=httpx.Response(200)
        )
        await client.stop_task("TASK_1")

    @pytest.mark.asyncio
    async def test_delete_task(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_1/delete?force=false").mock(
            return_value=httpx.Response(200)
        )
        await client.delete_task("TASK_1")

    @pytest.mark.asyncio
    async def test_fork_task(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_1/fork").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_forked",
                "result": "",
                "prefill_time": 0,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.fork_task("TASK_1", max_tokens=10)
        assert task.task_id == "TASK_forked"

    @pytest.mark.asyncio
    async def test_continue_task(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_1/continue").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_1",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.continue_task("TASK_1", max_tokens=10)
        assert task.task_id == "TASK_1"
