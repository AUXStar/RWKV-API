"""同步 Client 集成测试 —— 使用真实 RWKV-Server"""
import pytest
import time
from rwkv_api import Client, AsyncClient
from rwkv_api._task import Task


class TestSyncCreate:
    def test_create_tmp(self, server_alive, sync_client):
        task = sync_client.create("def hello():\n    ", max_tokens=10)
        assert isinstance(task, Task)
        assert task.task_id.startswith("TMP_")

    def test_create_persistent(self, server_alive, sync_client):
        task = sync_client.create("Hello", max_tokens=10, persistent=True)
        assert task.task_id.startswith("TASK_")

    def test_create_tmp_alias(self, server_alive, sync_client):
        task = sync_client.create_tmp("Hello", max_tokens=10)
        assert task.task_id.startswith("TMP_")

    def test_create_with_seed(self, server_alive, sync_client):
        task = sync_client.create("Hello", max_tokens=5, seed=42)
        assert task.task_id

    def test_fim(self, server_alive, sync_client):
        task = sync_client.fim(prefix="def foo():\n    return ", suffix="\n    # end", max_tokens=10)
        assert task.task_id

    def test_list_tasks(self, server_alive, sync_client):
        result = sync_client.list_tasks()
        assert "total_count" in result
        assert isinstance(result["total_count"], int)


class TestSyncTaskOperations:
    def test_wait(self, server_alive, sync_client):
        task = sync_client.create("Hello", max_tokens=10)
        task.wait(timeout=30)
        assert task.finished
        assert isinstance(task.result, str)

    def test_stop_and_delete(self, server_alive, sync_client):
        task = sync_client.create("Hello", max_tokens=100)
        task.stop()
        task.delete()

    def test_fork(self, server_alive, sync_client):
        task = sync_client.create("Hello", max_tokens=5)
        task.wait(timeout=30)
        forked = task.fork(prompt="World", max_tokens=5)
        assert forked.task_id != task.task_id

    def test_continue(self, server_alive, sync_client):
        task = sync_client.create("Hello", max_tokens=5)
        task.wait(timeout=30)
        continued = task.continue_(max_tokens=5)
        assert continued.task_id == task.task_id