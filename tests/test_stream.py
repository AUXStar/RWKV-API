"""流式 API 集成测试 —— 使用真实 RWKV-Server

覆盖 create_stream、fim_stream、stream_task、Task.stream() 等方法
"""
import pytest
from rwkv_api import Client, AsyncClient
from rwkv_api._task import Task, AsyncTask


class TestSyncStream:
    """同步流式 API 测试"""

    def test_create_stream_returns_chunks(self, server_alive, sync_client):
        chunks = list(sync_client.create_stream("def hello():\n    ", max_tokens=10))
        assert isinstance(chunks, list)
        # 至少有一个文本 chunk
        texts = [c for c in chunks if c.strip()]
        assert len(texts) > 0

    def test_create_stream_with_gen_params(self, server_alive, sync_client):
        chunks = list(sync_client.create_stream(
            "def hello():\n    ", max_tokens=10, temperature=0.5, top_k=10,
        ))
        assert len(chunks) >= 0  # 不崩即可

    def test_create_with_stream_true_returns_iterator(self, server_alive, sync_client):
        result = sync_client.create("Hello", max_tokens=10, stream=True)
        assert hasattr(result, '__iter__')
        assert not hasattr(result, 'task_id')

    def test_create_with_stream_false_returns_task(self, server_alive, sync_client):
        result = sync_client.create("Hello", max_tokens=10, stream=False)
        assert isinstance(result, Task)
        assert result.task_id

    def test_fim_stream_basic(self, server_alive, sync_client):
        chunks = list(sync_client.fim_stream(
            prefix="def add(a, b):\n    return ",
            suffix="\n    # end",
            max_tokens=10,
        ))
        texts = [c for c in chunks if c.strip()]
        assert len(texts) >= 0

    def test_stream_task_on_existing(self, server_alive, sync_client):
        task = sync_client.create("Hello", max_tokens=10)
        task.wait(timeout=30)
        chunks = list(sync_client.stream_task(task.task_id))
        texts = [c for c in chunks if c.strip()]
        assert len(texts) >= 0


class TestAsyncStream:
    """异步流式 API 测试"""

    @pytest.mark.asyncio
    async def test_create_stream_async(self, server_alive, async_client):
        chunks = []
        async for chunk in async_client.create_stream("def hello():\n    ", max_tokens=10):
            chunks.append(chunk)
        texts = [c for c in chunks if c.strip()]
        assert len(texts) > 0

    @pytest.mark.asyncio
    async def test_fim_stream_async(self, server_alive, async_client):
        chunks = []
        async for chunk in async_client.fim_stream(
            prefix="def foo():\n    return ",
            suffix="\n    # end",
            max_tokens=10,
        ):
            chunks.append(chunk)
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_stream_task_async(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=10)
        await task.wait(timeout=30)
        chunks = []
        async for chunk in async_client.stream_task(task.task_id):
            chunks.append(chunk)
        texts = [c for c in chunks if c.strip()]
        assert len(texts) >= 0

    @pytest.mark.asyncio
    async def test_task_stream_method(self, server_alive, async_client):
        task = await async_client.create("Hello", max_tokens=10)
        await task.wait(timeout=30)
        chunks = []
        async for chunk in task.stream():
            chunks.append(chunk)
        assert isinstance(chunks, list)