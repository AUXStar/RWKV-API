"""pytest 共享 fixture —— 使用真实 RWKV-Server"""
import asyncio
import aiohttp
import pytest
import pytest_asyncio
from rwkv_api import AsyncClient, Client

BASE_URL = "http://localhost:8000"


@pytest_asyncio.fixture
async def session():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        yield s


@pytest_asyncio.fixture
async def server_alive(session):
    try:
        resp = await session.get(f"{BASE_URL}/v1/models", timeout=aiohttp.ClientTimeout(total=5))
        if resp.status != 200:
            pytest.skip("Server not ready")
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pytest.skip("Server not reachable")


@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(BASE_URL) as client:
        yield client


@pytest.fixture
def sync_client():
    with Client(BASE_URL) as client:
        yield client