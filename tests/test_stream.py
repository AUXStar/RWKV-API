"""流式 API 全量测试 —— 覆盖 create_stream、fim_stream、stream_task、边界条件"""
import json
import pytest
import httpx
from rwkv_api import Client, AsyncClient
from rwkv_api.exceptions import ConnectionError, TimeoutError


# ===================================================================
# 同步流式测试
# ===================================================================

class TestSyncStream:
    """同步流式 API 测试"""

    def test_create_stream_basic(self, respx_mock):
        """create_stream 基本流式输出"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_s1"}\n\n',
                b'data: {"data": "Hello", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_s1"}\n\n',
                b'data: {"data": " world", "gen_time": 0.3, "speed": 50.0, "task_id": "TASK_s1"}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(side_effect=sse_response)

        client = Client()
        chunks = list(client.create_stream("Hello", max_tokens=10))
        assert chunks == ["Hello", " world"]

    def test_create_stream_with_gen_params(self, respx_mock):
        """create_stream 传递生成参数"""
        captured = {}
        def sse_response(request):
            captured['body'] = json.loads(request.content)
            return httpx.Response(200, content=b'data: [DONE]\n\n', headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(side_effect=sse_response)

        client = Client()
        list(client.create_stream("Hi", max_tokens=100, temperature=0.5, top_k=10, top_p=0.9,
                                   presence_penalty=1.0, repetition_penalty=1.2, penalty_decay=0.5, seed=42))
        assert captured['body']['stream'] is True
        assert captured['body']['max_tokens'] == 100
        assert captured['body']['temperature'] == 0.5
        assert captured['body']['top_k'] == 10
        assert captured['body']['top_p'] == 0.9
        assert captured['body']['presence_penalty'] == 1.0
        assert captured['body']['repetition_penalty'] == 1.2
        assert captured['body']['penalty_decay'] == 0.5
        assert captured['body']['seed'] == 42

    def test_create_stream_empty_response(self, respx_mock):
        """create_stream 空响应（直接 DONE）"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, content=b'data: [DONE]\n\n',
                                         headers={"content-type": "text/event-stream"})
        )
        client = Client()
        chunks = list(client.create_stream("Hello"))
        assert chunks == []

    def test_create_stream_connection_error(self, respx_mock):
        """create_stream 连接失败"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        client = Client()
        with pytest.raises(ConnectionError):
            list(client.create_stream("Hello"))

    def test_fim_stream_basic(self, respx_mock):
        """fim_stream 基本流式输出"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_fim1"}\n\n',
                b'data: {"data": "return a + b", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_fim1"}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/fim").mock(side_effect=sse_response)

        client = Client()
        chunks = list(client.fim_stream(prefix="def add(a, b):\n    ", suffix="\n    return result"))
        assert chunks == ["return a + b"]

    def test_create_with_stream_true_returns_iterator(self, respx_mock):
        """create(stream=True) 返回 Iterator 而非 Task"""
        def sse_response(request):
            return httpx.Response(200, content=b'data: [DONE]\n\n',
                                  headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(side_effect=sse_response)

        client = Client()
        result = client.create("Hello", stream=True)
        # 应该返回生成器，不是 Task
        assert hasattr(result, '__iter__')
        assert not hasattr(result, 'task_id')

    def test_create_with_stream_false_returns_task(self, respx_mock):
        """create(stream=False) 返回 Task"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_no_stream",
                "result": "Hello",
                "prefill_time": 0.1,
                "gen_time": 0.5,
                "speed": 50.0,
                "finished": True,
            })
        )
        client = Client()
        task = client.create("Hello", stream=False)
        assert hasattr(task, 'task_id')
        assert task.task_id == "TASK_no_stream"


# ===================================================================
# 异步流式测试
# ===================================================================

class TestAsyncStream:
    """异步流式 API 测试"""

    @pytest.mark.asyncio
    async def test_create_stream_async(self, respx_mock):
        """异步 create_stream"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_a1"}\n\n',
                b'data: {"data": "Async", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_a1"}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(side_effect=sse_response)

        client = AsyncClient()
        chunks = []
        async for chunk in client.create_stream("Hello", max_tokens=10):
            chunks.append(chunk)
        assert chunks == ["Async"]

    @pytest.mark.asyncio
    async def test_fim_stream_async(self, respx_mock):
        """异步 fim_stream"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_afim"}\n\n',
                b'data: {"data": "x + y", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_afim"}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/fim").mock(side_effect=sse_response)

        client = AsyncClient()
        chunks = []
        async for chunk in client.fim_stream("def add(x, y):\n    ", "\n    return result"):
            chunks.append(chunk)
        assert chunks == ["x + y"]


# ===================================================================
# stream_task 测试（对已创建任务订阅流式）
# ===================================================================

class TestStreamTask:
    """stream_task —— 对已创建任务订阅流式"""

    def test_stream_task_sync(self, respx_mock):
        """同步 stream_task"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_existing"}\n\n',
                b'data: {"data": "Continued", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_existing"}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.get("http://localhost:8000/v1/tasks/TASK_existing/stream").mock(side_effect=sse_response)

        client = Client()
        chunks = list(client.stream_task("TASK_existing"))
        assert chunks == ["Continued"]

    @pytest.mark.asyncio
    async def test_stream_task_async(self, respx_mock):
        """异步 stream_task"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_existing"}\n\n',
                b'data: {"data": "AsyncContinued", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_existing"}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.get("http://localhost:8000/v1/tasks/TASK_existing/stream").mock(side_effect=sse_response)

        client = AsyncClient()
        chunks = []
        async for chunk in client.stream_task("TASK_existing"):
            chunks.append(chunk)
        assert chunks == ["AsyncContinued"]

    def test_task_stream_method(self, respx_mock):
        """Task.stream() 实例方法"""
        # 先 mock create 返回 Task
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(200, json={
                "task_id": "TASK_method",
                "result": "",
                "prefill_time": 0.1,
                "gen_time": 0,
                "speed": 0,
                "finished": False,
            })
        )
        # 再 mock stream 端点
        def sse_response(request):
            return httpx.Response(200, content=b'data: {"data": "FromTask"}\n\ndata: [DONE]\n\n',
                                  headers={"content-type": "text/event-stream"})

        respx_mock.get("http://localhost:8000/v1/tasks/TASK_method/stream").mock(side_effect=sse_response)

        client = Client()
        task = client.create("Hello")
        chunks = list(task.stream())
        assert chunks == ["FromTask"]


# ===================================================================
# 边界条件测试
# ===================================================================

class TestStreamEdgeCases:
    """流式边界条件测试"""

    def test_stream_malformed_sse(self, respx_mock):
        """处理格式错误的 SSE 行"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_mal"}\n\n',
                b'data: not-json\n\n',  # 格式错误，应跳过
                b'data: {"data": "Valid", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_mal"}\n\n',
                b'not-data: something\n\n',  # 不以 data: 开头，应跳过
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(side_effect=sse_response)

        client = Client()
        chunks = list(client.create_stream("Hello"))
        assert chunks == ["Valid"]

    def test_stream_skip_prefill_event(self, respx_mock):
        """跳过只含 prefill_time 的首事件"""
        def sse_response(request):
            chunks = [
                b'data: {"prefill_time": 0.1, "task_id": "TASK_pf"}\n\n',  # 无 data 字段，应跳过
                b'data: {"data": "AfterPrefill", "gen_time": 0.2, "speed": 50.0, "task_id": "TASK_pf"}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(200, content=b''.join(chunks), headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(side_effect=sse_response)

        client = Client()
        chunks = list(client.create_stream("Hello"))
        assert chunks == ["AfterPrefill"]

    def test_stream_timeout(self, respx_mock):
        """流式请求超时"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            side_effect=httpx.TimeoutException("Read timeout")
        )
        client = Client()
        with pytest.raises(TimeoutError):
            list(client.create_stream("Hello"))

    def test_stream_500_error(self, respx_mock):
        """流式请求服务端 500"""
        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        client = Client()
        with pytest.raises(Exception) as exc:
            list(client.create_stream("Hello"))
        assert "500" in str(exc.value) or "Server" in str(exc.value)

    def test_stream_task_not_found(self, respx_mock):
        """stream_task 404"""
        respx_mock.get("http://localhost:8000/v1/tasks/TASK_missing/stream").mock(
            return_value=httpx.Response(404)
        )
        client = Client()
        with pytest.raises(Exception) as exc:
            list(client.stream_task("TASK_missing"))
        assert "404" in str(exc.value)

    def test_stream_large_chunks(self, respx_mock):
        """流式大 chunk"""
        large_text = "A" * 10000
        def sse_response(request):
            content = f'data: {{"prefill_time": 0.1, "task_id": "TASK_large"}}\n\ndata: {{"data": "{large_text}", "gen_time": 0.5, "speed": 100.0, "task_id": "TASK_large"}}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=content.encode(), headers={"content-type": "text/event-stream"})

        respx_mock.post("http://localhost:8000/v1/tasks/tmp").mock(side_effect=sse_response)

        client = Client()
        chunks = list(client.create_stream("Hello"))
        assert len(chunks) == 1
        assert len(chunks[0]) == 10000
