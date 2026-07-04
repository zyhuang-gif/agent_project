"""发送审计 service (P3a)。

- DB 是主路径：send_guard 成功后调用 `send_audit_service.record_send_success(...)`，
  写入 send_audit 表。
- JSONL 兼容双写：既保留 P0-P2 的 log/sends.log 兜底文件（便于人工排查、
  离线工具兼容），也在 DB 写入异常时作为"至少留下一份痕迹"的降级路径。
- DB / JSONL 都失败也绝不 raise 到调用方：外部工具已经真的发出消息，本地
  审计失败不能变成"用户看到失败"。仅 warning。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.logger_handler import logger
from app.db.db_config import AsyncSessionLocal
from app.models.send_audit import SendAudit


def _audit_path() -> Path:
    """log/sends.log 相对 backend/ 的绝对路径。

    __file__ = backend/app/services/send_audit_service.py
    parents[2] = backend/
    """
    return Path(__file__).resolve().parents[2] / "log" / "sends.log"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_preview(payload: Optional[dict], limit: int = 500) -> Optional[str]:
    if not payload:
        return None
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = str(payload)
    return text[:limit]


class SendAuditService:
    """P3a 审计 service：暴露一个入口 `record_send_success`。"""

    async def record_send_success(
        self,
        *,
        channel: str,
        recipient_kind: str,
        recipient: str,
        message_id: str,
        tool_name: str,
        recipient_name: Optional[str] = None,
        subject: Optional[str] = None,
        artifact_name: Optional[str] = None,
        artifact_path: Optional[str] = None,
        file_key: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_payload: Optional[dict] = None,
    ) -> None:
        payload_preview = _payload_preview(request_payload)

        db_ok = await self._write_db(
            channel=channel,
            recipient_kind=recipient_kind,
            recipient=recipient,
            recipient_name=recipient_name,
            subject=subject,
            artifact_name=artifact_name,
            artifact_path=artifact_path,
            message_id=message_id,
            file_key=file_key,
            tool_name=tool_name,
            payload_preview=payload_preview,
            user_id=user_id,
            session_id=session_id,
        )

        # JSONL 双写：即便 DB 成功也写，作为兼容/离线可读的记录。
        try:
            self._write_jsonl(
                channel=channel,
                recipient_kind=recipient_kind,
                recipient=recipient,
                recipient_name=recipient_name,
                subject=subject,
                artifact_name=artifact_name,
                message_id=message_id,
                file_key=file_key,
                tool_name=tool_name,
                user_id=user_id,
                session_id=session_id,
                db_ok=db_ok,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("send_audit JSONL 兜底写入失败：%s", exc)

    async def _write_db(self, **fields) -> bool:
        try:
            async with AsyncSessionLocal() as db:
                db.add(SendAudit(
                    user_id=fields.get("user_id"),
                    session_id=fields.get("session_id"),
                    channel=fields["channel"],
                    recipient_kind=fields["recipient_kind"],
                    recipient=fields["recipient"],
                    recipient_name=fields.get("recipient_name"),
                    subject=fields.get("subject"),
                    artifact_name=fields.get("artifact_name"),
                    artifact_path=fields.get("artifact_path"),
                    message_id=fields["message_id"],
                    file_key=fields.get("file_key"),
                    tool_name=fields["tool_name"],
                    request_payload_preview=fields.get("payload_preview"),
                    status="success",
                ))
                await db.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("send_audit DB 写入失败，降级 JSONL：%s", exc)
            return False

    def _write_jsonl(self, *, db_ok: bool, **fields) -> None:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": _now_iso(),
            "user_id": fields.get("user_id"),
            "session_id": fields.get("session_id"),
            "channel": fields["channel"],
            "recipient_kind": fields["recipient_kind"],
            "recipient": fields["recipient"],
            "recipient_name": fields.get("recipient_name"),
            "subject_or_preview": fields.get("subject") or fields.get("artifact_name"),
            "message_id": fields["message_id"],
            "tool_name": fields["tool_name"],
            "db_persisted": db_ok,
        }
        if fields.get("file_key"):
            record["file_key"] = fields["file_key"]
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


send_audit_service = SendAuditService()
