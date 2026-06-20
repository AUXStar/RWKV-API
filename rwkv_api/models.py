"""RWKV API 数据模型（Pydantic v2）。

定义所有请求 / 响应的序列化结构，
与 RWKV-Server Task API 的 JSON Schema 一一对应。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class Status(str, Enum):
    """任务状态枚举。"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"

class TaskCreate(BaseModel):
    """创建任务的请求体。"""
    prompt: str | list[int]
    max_tokens: int = Field(default=4096, ge=0, le=40960)
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_k: int = Field(default=-1, ge=-1, le=100)
    top_p: float = Field(default=1.0, ge=0, le=1)
    presence_penalty: float = Field(default=0.0, ge=0, le=10)
    repetition_penalty: float = Field(default=1.0, ge=0, le=10)
    penalty_decay: float = Field(default=0.0, ge=0, le=1)
    stream: bool = False
    seed: int | None = None


class TaskUpdate(BaseModel):
    """更新 / fork / 继续任务的请求体。

    所有字段均为可选（除 stream 默认 False），
    未提供的字段将沿用原任务的值。
    """
    prompt: str | list[int] | None = None
    max_tokens: int | None = Field(default=None, ge=0, le=40960)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_k: int | None = Field(default=None, ge=-1, le=100)
    top_p: float | None = Field(default=None, ge=0, le=1)
    presence_penalty: float | None = Field(default=None, ge=0, le=10)
    repetition_penalty: float | None = Field(default=None, ge=0, le=10)
    penalty_decay: float | None = Field(default=None, ge=0, le=1)
    stream: bool = False
    seed: int | None = None


class FIMRequest(BaseModel):
    """Fill In Middle 请求体。"""
    prefix: str
    suffix: str = ""
    max_tokens: int = Field(default=4096, ge=0, le=40960)
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_k: int = Field(default=-1, ge=-1, le=100)
    top_p: float = Field(default=1.0, ge=0, le=1)
    presence_penalty: float = Field(default=0.0, ge=0, le=10)
    repetition_penalty: float = Field(default=1.0, ge=0, le=10)
    penalty_decay: float = Field(default=0.0, ge=0, le=1)
    stream: bool = False
    seed: int | None = None


class TaskResponseModel(BaseModel):
    """任务响应模型（非流式）。"""
    task_id: str
    result: str = ""
    prefill_time: float = 0.0
    gen_time: float = 0.0
    speed: float = 0.0
    finished: bool = False


class TaskInfo(BaseModel):
    """任务状态信息。"""
    task_id: str
    generated_buf: int = 0
    status: Status = Status.FINISHED



# SSE 结束标记


# SSE 结束标记

# SSE 结束标记

# SSE 结束标记

# SSE 结束标记

# SSE 结束标记

# SSE 结束标记
 