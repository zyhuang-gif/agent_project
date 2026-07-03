import pytest


@pytest.mark.asyncio
async def test_startup_and_shutdown_call_mcp_manager(monkeypatch):
    import main

    calls = []

    class _Manager:
        async def start(self):
            calls.append("start")

        async def shutdown(self):
            calls.append("shutdown")

    async def _noop_async(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "init_db", _noop_async)
    monkeypatch.setattr(main, "init_database_session_manager", _noop_async)
    monkeypatch.setattr(main, "connect_redis", _noop_async)
    monkeypatch.setattr(main, "close_redis", _noop_async)
    monkeypatch.setattr(main, "check_and_download_reranker_model", lambda: None)
    monkeypatch.setattr(main, "start_scheduler", lambda: None)
    monkeypatch.setattr(main, "stop_scheduler", lambda: None)
    monkeypatch.setattr(main, "mcp_manager", _Manager())

    await main.startup_event()
    await main.shutdown_event()

    assert calls == ["start", "shutdown"]
