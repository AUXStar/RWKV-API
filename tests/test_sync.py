"""同步 Client 测试"""
import pytest
import httpx
from rwkv_api import Client


@pytest.fixture
def client():
    return Client("http://localhost:8000")


class TestSyncCreate:
    """同步创建任务测试"""

    def test_create_tmp(self, client, respx_mock):
        """默认 create 走 /tmp"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_sync1",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = client.create("Hello", max_tokens=10)
        assert task.task_id == "TASK_sync1"

    def test_create_persistent(self, client, respx_mock):
        """persistent=True 走 /create"""
        respx_mock.post("http://localhost:8000/v1/tasks/create").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_sync_p",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = client.create("Hello", max_tokens=10, persistent=True)
        assert task.task_id == "TASK_sync_p"

    def test_create_tmp_alias(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TMP_sync",
                "result": "",
                "prefill_time": 0.05,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = client.create_tmp("Hi", max_tokens=5)
        assert task.task_id.startswith("TMP_")

    def test_fim(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/fim").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TMP_fim",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = client.fim(prefix="def foo():", suffix="", max_tokens=10)
        assert task.task_id == "TMP_fim"

    def test_list_tasks(self, client, respx_mock):
        respx_mock.get("http://localhost:8000/v1/tasks/list").mock(
            return_value=httpx.Response(200, json={
                "cpu_cache_count": 1,
                "database_count": 3,
                "total_count": 4,
                "cpu_tasks": ["TASK_1"],
                "db_tasks": [],
            })
        )
        result = client.list_tasks()
        assert result["total_count"] == 4


class TestSyncTaskOperations:
    """同步 Task 操作测试"""

    def test_wait(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_sw",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.get("http://localhost:8000/v1/tasks/TASK_sw/get_result").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_sw",
                "result": "Sync done",
                "prefill_time": 0.1,
                "gen_time": 0.5,
                "speed": 50.0,
                "finished": True,
            })
        )
        task = client.create("Hello")
        result = task.wait(timeout=5, poll_interval=0.1)
        assert result.result == "Sync done"

    def test_stop(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_ss",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_ss/stop").mock(
            return_value=httpx.Response(200)
        )
        task = client.create("Hello")
        task.stop()

    def test_delete(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_sd",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_sd/delete?force=false").mock(
            return_value=httpx.Response(200)
        )
        task = client.create("Hello")
        task.delete()
