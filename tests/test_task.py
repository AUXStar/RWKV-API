"""Task / AsyncTask 测试"""
import pytest
import httpx
from rwkv_api import AsyncClient, AsyncTask


@pytest.fixture
def client():
    return AsyncClient("http://localhost:8000")


class TestAsyncTaskWait:
    """AsyncTask.wait() 轮询测试"""

    @pytest.mark.asyncio
    async def test_wait_finished(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_wait",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.get("http://localhost:8000/v1/tasks/TASK_wait/get_result").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_wait",
                "result": "Done!",
                "prefill_time": 0.1,
                "gen_time": 0.5,
                "speed": 50.0,
                "finished": True,
            })
        )
        task = await client.create("Hello", max_tokens=10)
        result = await task.wait(timeout=5, poll_interval=0.1)
        assert result.result == "Done!"
        assert result.finished

    @pytest.mark.asyncio
    async def test_wait_timeout(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_timeout",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.get("http://localhost:8000/v1/tasks/TASK_timeout/get_result").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_timeout",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.create("Hello", max_tokens=10)
        from rwkv_api.exceptions import TimeoutError
        with pytest.raises(TimeoutError):
            await task.wait(timeout=0.2, poll_interval=0.1)


class TestAsyncTaskOperations:
    """AsyncTask 操作方法测试"""

    @pytest.mark.asyncio
    async def test_task_stop(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_stop",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_stop/stop").mock(
            return_value=httpx.Response(200)
        )
        task = await client.create("Hello")
        await task.stop()

    @pytest.mark.asyncio
    async def test_task_delete(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_del",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_del/delete?force=false").mock(
            return_value=httpx.Response(200)
        )
        task = await client.create("Hello")
        await task.delete()

    @pytest.mark.asyncio
    async def test_task_fork(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_fork_src",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_fork_src/fork").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_forked",
                "result": "",
                "prefill_time": 0,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.create("Hello")
        forked = await task.fork(max_tokens=10)
        assert forked.task_id == "TASK_forked"

    @pytest.mark.asyncio
    async def test_task_continue(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_cont",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_cont/continue").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_cont",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.create("Hello")
        continued = await task.continue_(max_tokens=10)
        assert continued.task_id == "TASK_cont"

    @pytest.mark.asyncio
    async def test_task_as_template(self, client, respx_mock):
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_tpl",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        respx_mock.post("http://localhost:8000/v1/tasks/TASK_tpl/as_template").mock(
            return_value=httpx.Response(200, json={
                "task_id": "_TMPL_abc",
                "result": "",
                "prefill_time": 0,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        task = await client.create("Hello")
        tpl = await task.as_template()
        assert tpl.task_id.startswith("_TMPL_")
