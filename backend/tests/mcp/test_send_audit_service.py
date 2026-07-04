"""SendAuditService (P3a)：DB 主路径 + JSONL 兼容双写 + DB 失败兜底。"""
import json
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.chat_history import Base
from app.models.send_audit import SendAudit


@pytest_asyncio.fixture
async def audit_service_db(monkeypatch, tmp_path):
    """内存 SQLite + tmp_path JSONL：验证 DB 主路径 + JSONL 双写。"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    import app.services.send_audit_service as mod
    monkeypatch.setattr(mod, "AsyncSessionLocal", Session)
    monkeypatch.setattr(mod, "_audit_path", lambda: tmp_path / "sends.log")
    yield Session, tmp_path
    await engine.dispose()


@pytest.mark.asyncio
async def test_record_send_success_persists_row_to_db(audit_service_db):
    Session, _ = audit_service_db
    from app.services.send_audit_service import send_audit_service

    await send_audit_service.record_send_success(
        channel="gmail",
        recipient_kind="email",
        recipient="a@example.com",
        recipient_name="张三",
        subject="报告",
        message_id="gmail-1",
        tool_name="send_email",
        user_id="u1",
        session_id="s1",
        artifact_name="report.md",
        artifact_path="/tmp/report.md",
        request_payload={"to": ["a@example.com"], "subject": "报告"},
    )

    async with Session() as db:
        rows = (await db.execute(select(SendAudit))).scalars().all()

    assert len(rows) == 1
    row = rows[0]
    assert row.channel == "gmail"
    assert row.recipient == "a@example.com"
    assert row.recipient_name == "张三"
    assert row.subject == "报告"
    assert row.message_id == "gmail-1"
    assert row.tool_name == "send_email"
    assert row.user_id == "u1"
    assert row.session_id == "s1"
    assert row.status == "success"
    assert row.artifact_name == "report.md"
    assert "a@example.com" in (row.request_payload_preview or "")


@pytest.mark.asyncio
async def test_record_send_success_feishu_file_persists_file_key(audit_service_db):
    Session, _ = audit_service_db
    from app.services.send_audit_service import send_audit_service

    await send_audit_service.record_send_success(
        channel="feishu",
        recipient_kind="open_id",
        recipient="ou_ok",
        recipient_name="张三",
        artifact_name="report.md",
        artifact_path="/tmp/report.md",
        message_id="om_file",
        file_key="file_v3_abc",
        tool_name="send_file_to_user",
    )

    async with Session() as db:
        rows = (await db.execute(select(SendAudit))).scalars().all()

    assert len(rows) == 1
    row = rows[0]
    assert row.file_key == "file_v3_abc"
    assert row.channel == "feishu"
    assert row.recipient_kind == "open_id"
    assert row.tool_name == "send_file_to_user"


@pytest.mark.asyncio
async def test_record_send_success_also_writes_jsonl_compat(audit_service_db):
    """DB 成功时依然写 JSONL 兼容记录（旧脚本/离线兜底仍能读到）。"""
    _Session, tmp_path = audit_service_db
    from app.services.send_audit_service import send_audit_service

    await send_audit_service.record_send_success(
        channel="gmail",
        recipient_kind="email",
        recipient="a@example.com",
        message_id="msg-1",
        tool_name="send_email",
    )

    log = tmp_path / "sends.log"
    assert log.exists()
    body = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert body["message_id"] == "msg-1"
    assert body["channel"] == "gmail"
    assert body["db_persisted"] is True


@pytest.mark.asyncio
async def test_record_send_success_swallows_db_error_and_writes_jsonl(monkeypatch, tmp_path):
    """DB 层报错不能 raise 出来；应 warning + 走 JSONL 兜底。"""
    from app.services import send_audit_service as mod

    class _BadSession:
        async def __aenter__(self):
            raise RuntimeError("db down")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(mod, "AsyncSessionLocal", lambda: _BadSession())
    monkeypatch.setattr(mod, "_audit_path", lambda: tmp_path / "sends.log")

    warnings: list[str] = []
    monkeypatch.setattr(mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))

    from app.services.send_audit_service import send_audit_service
    await send_audit_service.record_send_success(
        channel="feishu",
        recipient_kind="open_id",
        recipient="ou_ok",
        message_id="om-1",
        tool_name="send_message_to_user",
    )

    assert warnings, "DB 失败必须记录 warning"
    assert any("send_audit" in w.lower() for w in warnings)

    lines = (tmp_path / "sends.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    body = json.loads(lines[0])
    assert body["message_id"] == "om-1"
    assert body["channel"] == "feishu"
    assert body["db_persisted"] is False


@pytest.mark.asyncio
async def test_record_send_success_does_not_raise_when_jsonl_fails(monkeypatch, tmp_path):
    """极端情况：DB 与 JSONL 都失败，也不能让调用方拿到异常。"""
    from app.services import send_audit_service as mod

    class _BadSession:
        async def __aenter__(self):
            raise RuntimeError("db down")

        async def __aexit__(self, *a):
            return False

    def _bad_path():
        raise OSError("disk down")

    monkeypatch.setattr(mod, "AsyncSessionLocal", lambda: _BadSession())
    monkeypatch.setattr(mod, "_audit_path", _bad_path)

    warnings: list[str] = []
    monkeypatch.setattr(mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))

    from app.services.send_audit_service import send_audit_service
    # 不应 raise
    await send_audit_service.record_send_success(
        channel="gmail",
        recipient_kind="email",
        recipient="a@example.com",
        message_id="msg-x",
        tool_name="send_email",
    )

    # DB warning + JSONL warning 至少各一条
    assert len(warnings) >= 2


# ── JSONL 路径回锁：必须在仓库根/log/sends.log 下 ─────────────────


def test_audit_path_points_to_repo_root_log():
    """P0-P2 的 sends.log 在 <repo>/log/ 下。改造后必须保持不变，不能搬到
    backend/log/ 或其它任何位置——否则运维旧脚本读不到、审计线索错位。"""
    import app.services.send_audit_service as mod

    p = mod._audit_path()
    assert p.name == "sends.log", f"文件名必须是 sends.log，实际={p.name}"
    assert p.parent.name == "log", f"父目录必须叫 log，实际={p.parent.name}"

    # 反锁：绝不允许再次搬到 backend/log
    assert p.parent.parent.name != "backend", (
        f"sends.log 不能落在 backend/log 下，应在仓库根/log；实际路径={p}"
    )

    # 正锁：与 P0-P2 保持一致 —— <repo>/log/sends.log
    # send_audit_service.py 位于 backend/app/services/，parents[3] = <repo>
    repo_root = Path(mod.__file__).resolve().parents[3]
    assert p == repo_root / "log" / "sends.log", (
        f"预期 {repo_root / 'log' / 'sends.log'}，实际 {p}"
    )


def test_audit_path_matches_send_guard_history():
    """更严格：与 P0-P2 时 send_guard.py 里的路径口径完全一致。"""
    import app.mcp.send_guard as sg_mod
    import app.services.send_audit_service as svc_mod

    # send_guard.py 位于 backend/app/mcp/，parents[3] = <repo>
    legacy_repo_root = Path(sg_mod.__file__).resolve().parents[3]
    assert svc_mod._audit_path() == legacy_repo_root / "log" / "sends.log"
