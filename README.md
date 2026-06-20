# RWKV-API

RWKV-Server 的官方 Python SDK，**以最优雅的方式调用 RWKV 推理**。

> 无需关心批量调度、状态管理、GPU 显存分配——只需创建一个 `Task`，RWKV-Server 会在后台自动完成一切。

## 配套服务端

本 SDK 对应 [RWKV-Server](https://github.com/AUXStar/RWKV-Server) 的 Task API。服务端基于 RWKV7 线性注意力架构，支持动态批量推理、2的幂次自动扩缩容、模板持久化等高级特性，单机可达 **11,081 tok/s** 峰值吞吐。

## 安装

```bash
pip install rwkv-api
```

## 快速开始

### 同步调用

```python
from rwkv_api import Client

client = Client("http://localhost:8000")

# 创建任务并等待结果 —— 服务端自动批量调度
task = client.create("Hello world", max_tokens=50)
result = task.wait()
print(result.result)

# FIM 代码补全
result = client.fim(
    prefix="def add(a, b):\n    ",
    suffix="\n    return result",
    max_tokens=30
)
print(result.result)
```

### 异步调用

```python
import asyncio
from rwkv_api import AsyncClient

async def main():
    client = AsyncClient("http://localhost:8000")
    task = await client.create("Hello world", max_tokens=50)
    result = await task.wait()
    print(result.result)

asyncio.run(main())
```

### 实时流式（创建时）

```python
from rwkv_api import Client

client = Client("http://localhost:8000")

# 方式1：create_stream 显式流式
for chunk in client.create_stream("Tell me a story", max_tokens=200):
    print(chunk, end="", flush=True)

# 方式2：create(stream=True) 自动分发
for chunk in client.create("Tell me a story", max_tokens=200, stream=True):
    print(chunk, end="", flush=True)
```

### 对已创建任务订阅流式

```python
from rwkv_api import Client

client = Client("http://localhost:8000")

# 先创建任务（非流式）
task = client.create("Hello", max_tokens=200)

# 稍后订阅实时流式输出
for chunk in task.stream():
    print(chunk, end="", flush=True)

# 或者直接用 task_id
for chunk in client.stream_task(task.task_id):
    print(chunk, end="", flush=True)
```

### 异步实时流式

```python
import asyncio
from rwkv_api import AsyncClient

async def main():
    client = AsyncClient("http://localhost:8000")

    # 创建时流式
    async for chunk in client.create_stream("Tell me a story", max_tokens=200):
        print(chunk, end="", flush=True)

    # 对已创建任务订阅流式
    task = await client.create("Hello", max_tokens=200)
    async for chunk in task.stream():
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Task 对象 —— 优雅调用 RWKV 推理的核心

SDK 的核心是 `Task` 对象。你只需关注**要生成什么**，所有底层细节（批量调度、状态生命周期、GPU 显存管理）都由服务端自动处理：

```python
task = client.create("Hello", max_tokens=50)

# 属性
print(task.task_id)       # "TASK_xxx"
print(task.finished)      # False
print(task.result)        # 生成结果（完成后可用）
print(task.speed)         # tok/s
print(task.prefill_time)  # prefill 耗时（秒）
print(task.gen_time)      # 生成耗时（秒）

# 操作
result = task.wait()           # 阻塞等待完成，返回 self
task.stop()                    # 停止生成
forked = task.fork()           # Fork 新任务
continued = task.continue_()   # 继续生成
tpl = task.as_template()       # 转为模板
task.delete()                  # 删除任务

# 流式订阅（任务创建后仍可实时获取输出）
for chunk in task.stream():
    print(chunk, end="", flush=True)
```

## 流式说明

SDK 提供两种流式方式：

| 方式 | 方法 | 返回值 | 适用场景 |
|------|------|--------|---------|
| **创建时流式** | `create_stream()` / `create(stream=True)` | `Iterator[str]` / `AsyncIterator[str]` | 创建任务的同时实时输出 |
| **任务订阅流式** | `task.stream()` / `client.stream_task(id)` | `Iterator[str]` / `AsyncIterator[str]` | 对已创建任务实时订阅 |

**注意**：
- `create(stream=True)` 返回**生成器**，不是 `Task` 对象
- `task.stream(pos=N)` 从第 N 个 token 开始（重连时传已读数量避免重复）
- 如果任务已完成，`stream()` 一次性返回完整结果

## 异常处理

```python
from rwkv_api import Client
from rwkv_api.exceptions import (
    RWKVValidationError,   # 422 参数错误
    TaskNotFoundError,      # 404 任务不存在
    RWKVServerError,        # 500 服务端错误
    TimeoutError,           # 等待超时
    ConnectionError,        # 连接失败
)

client = Client("http://localhost:8000")

try:
    task = client.create("Hello", max_tokens=50)
    result = task.wait(timeout=10)
except RWKVValidationError as e:
    print(f"参数错误: {e}")
except TimeoutError:
    print("等待超时")
except RWKVServerError as e:
    print(f"服务端错误: {e}")
```

## API 参考

### Client / AsyncClient

| 方法 | 说明 |
|------|------|
| `create(prompt, ...)` | 创建任务，`stream=True` 时返回生成器 |
| `create_stream(prompt, ...)` | **实时流式**，返回生成器 |
| `create_tmp(prompt, ...)` | 创建临时任务 |
| `create_persistent(prompt, ...)` | 创建持久化任务 |
| `fim(prefix, suffix, ...)` | FIM 代码补全，`stream=True` 时返回生成器 |
| `fim_stream(prefix, suffix, ...)` | **FIM 实时流式** |
| `stream_task(task_id, pos=0)` | **订阅已创建任务的流式输出** |
| `list_tasks()` | 列出所有任务 |
| `get_task_result(task_id)` | 获取任务结果 |
| `get_task_status(task_id)` | 获取任务状态 |
| `stop_task(task_id)` | 停止任务 |
| `delete_task(task_id, force=False)` | 删除任务 |
| `fork_task(task_id, ...)` | Fork 任务 |
| `continue_task(task_id, ...)` | 继续生成 |
| `as_template(task_id)` | 转为模板 |

### Task / AsyncTask

| 方法 | 说明 |
|------|------|
| `task.wait()` | 等待任务完成 |
| `task.stop()` | 停止生成 |
| `task.fork()` | Fork 新任务 |
| `task.continue_()` | 继续生成 |
| `task.as_template()` | 转为模板 |
| `task.delete()` | 删除任务 |
| `task.stream(pos=0)` | **订阅此任务的实时流式输出** |

### 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_tokens` | `int` | 256 | 最大生成 token 数 |
| `temperature` | `float` | 1.0 | 采样温度 (0~2) |
| `top_k` | `int` | 0 | Top-K 采样 (0=禁用) |
| `top_p` | `float` | 0.8 | Top-P 采样 |
| `presence_penalty` | `float` | 0.0 | 存在惩罚 (0~10) |
| `repetition_penalty` | `float` | 0.0 | 重复惩罚 (0~10) |
| `penalty_decay` | `float` | 0.0 | 惩罚衰减 (0~1) |
| `seed` | `int \| None` | None | 随机种子 |
| `stream` | `bool` | False | 流式输出 |

## 开发

```bash
git clone https://github.com/AUXStar/RWKV-API.git
cd RWKV-API
pip install -e ".[dev]"
pytest tests/ -v
```

## 许可证

MIT
