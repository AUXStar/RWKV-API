"""RWKV API 异常体系。

所有 SDK 抛出的异常都继承自 RWKVError，
方便用户统一捕获或按类型细分处理。
"""


class RWKVError(Exception):
    """RWKV API 基础异常。"""

    def __init__(self, message: str = "", *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class RWKVValidationError(RWKVError):
    """请求参数验证失败（HTTP 422）。"""

    def __init__(self, message: str = "请求参数验证失败", **kwargs) -> None:  # type: ignore[override]
        kwargs.setdefault("status_code", 422)
        super().__init__(message, **kwargs)


class RWKVServerError(RWKVError):
    """服务端内部错误（HTTP 500）。"""

    def __init__(self, message: str = "服务端内部错误", **kwargs) -> None:  # type: ignore[override]
        kwargs.setdefault("status_code", 500)
        super().__init__(message, **kwargs)


class TaskNotFoundError(RWKVError):
    """任务不存在（HTTP 404）。"""

    def __init__(self, message: str = "任务不存在", **kwargs) -> None:  # type: ignore[override]
        kwargs.setdefault("status_code", 404)
        super().__init__(message, **kwargs)


class TaskCancelledError(RWKVError):
    """任务被取消或停止。"""

    def __init__(self, message: str = "任务已被取消") -> None:
        super().__init__(message)


class TimeoutError(RWKVError):
    """请求超时。"""

    def __init__(self, message: str = "请求超时") -> None:
        super().__init__(message)


class ConnectionError(RWKVError):
    """连接失败。"""

    def __init__(self, message: str = "连接失败") -> None:
        super().__init__(message)


# HTTP 状态码到异常类的映射
_STATUS_CODE_MAP: dict[int, type[RWKVError]] = {
    404: TaskNotFoundError,
    422: RWKVValidationError,
    500: RWKVServerError,
}


def raise_for_status(status_code: int, body: str | None = None) -> None:
    """根据 HTTP 状态码抛出对应的异常。

    Args:
        status_code: HTTP 响应状态码。
        body: 可选的响应体文本，用于丰富错误信息。

    Raises:
        RWKVError: 对应状态码的异常。
    """
    if 200 <= status_code < 300:
        return
    exc_cls = _STATUS_CODE_MAP.get(status_code, RWKVError)
    message = f"HTTP {status_code}"
    if body:
        message = f"{message}: {body}"
    raise exc_cls(message, status_code=status_code)
