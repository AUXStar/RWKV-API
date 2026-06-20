"""rwkv_api —— RWKV-Server Task API 的 Python SDK

提供同步和异步两种接口，支持流式 SSE 响应和完整的任务生命周期管理

快速开始（异步）::

    import asyncio
    from rwkv_api import AsyncClient

    async def main():
        async with AsyncClient("http://localhost:8000") as client:
            task = await client.create("Hello, world!", max_tokens=50)
            await task.wait()
            print(task.result)

    asyncio.run(main())

快速开始（同步）::

    from rwkv_api import Client

    with Client("http://localhost:8000") as client:
        task = client.create("Hello, world!", max_tokens=50)
        task.wait()
        print(task.result)

流式生成（异步）::

    async with AsyncClient("http://localhost:8000") as client:
        task = await client.create("Hello", stream=True)
        async for chunk in task:
            print(chunk, end="")

流式生成（同步）::

    with Client("http://localhost:8000") as client:
        task = client.create("Hello", stream=True)
        for chunk in task:
            print(chunk, end="")
"""

from . import exceptions
from ._client import AsyncClient
from ._sync import Client
from ._task import AsyncTask, Task
from .exceptions import (
    ConnectionError as RWKVConnectionError,
    RWKVError,
    RWKVServerError,
    RWKVValidationError,
    TaskCancelledError,
    TaskNotFoundError,
    TimeoutError as RWKVTimeoutError,
)
from .models import (
    FIMRequest,
    Status,
    TaskCreate,
    TaskInfo,
    TaskResponseModel,
    TaskUpdate,
)

__version__ = "0.1.0"
__all__ = [
    # 客户端
    "AsyncClient",
    "Client",
    # Task 对象
    "Task",
    "AsyncTask",
    # 数据模型
    "TaskCreate",
    "TaskUpdate",
    "TaskResponseModel",
    "TaskInfo",
    "FIMRequest",
    "Status",
    # 异常
    "RWKVError",
    "RWKVValidationError",
    "RWKVServerError",
    "TaskNotFoundError",
    "TaskCancelledError",
    "RWKVTimeoutError",
    "RWKVConnectionError",
    "exceptions",
]


