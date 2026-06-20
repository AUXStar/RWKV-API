"""异常体系测试"""
import pytest
from rwkv_api.exceptions import (
    RWKVError,
    RWKVValidationError,
    RWKVServerError,
    TaskNotFoundError,
    TaskCancelledError,
    TimeoutError,
    ConnectionError,
    raise_for_status,
)


class TestRaiseForStatus:
    """raise_for_status 映射测试"""

    def test_200_no_raise(self):
        raise_for_status(200)

    def test_404_not_found(self):
        with pytest.raises(TaskNotFoundError) as exc:
            raise_for_status(404, "Task not found")
        assert exc.value.status_code == 404

    def test_422_validation(self):
        with pytest.raises(RWKVValidationError) as exc:
            raise_for_status(422, "Invalid input")
        assert exc.value.status_code == 422

    def test_500_server(self):
        with pytest.raises(RWKVServerError) as exc:
            raise_for_status(500, "Server error")
        assert exc.value.status_code == 500

    def test_unknown_status(self):
        with pytest.raises(RWKVError) as exc:
            raise_for_status(418, "I'm a teapot")
        assert exc.value.status_code == 418


class TestExceptionHierarchy:
    """异常继承关系测试"""

    def test_all_inherit_rwkv_error(self):
        assert issubclass(RWKVValidationError, RWKVError)
        assert issubclass(RWKVServerError, RWKVError)
        assert issubclass(TaskNotFoundError, RWKVError)
        assert issubclass(TaskCancelledError, RWKVError)
        assert issubclass(TimeoutError, RWKVError)
        assert issubclass(ConnectionError, RWKVError)

    def test_error_message(self):
        err = RWKVError("test", status_code=500)
        assert "test" in str(err)
        assert err.status_code == 500

    def test_error_message_no_code(self):
        err = RWKVError("test")
        assert str(err) == "test"
