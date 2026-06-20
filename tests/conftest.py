"""pytest 共享 fixture"""
import pytest


@pytest.fixture
def respx_mock():
    """提供 respx mock 支持"""
    try:
        import respx
        with respx.mock:
            yield respx
    except ImportError:
        pytest.skip("respx not installed")
