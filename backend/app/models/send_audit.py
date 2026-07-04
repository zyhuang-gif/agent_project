"""发送审计表 (P3a)：send_guard 成功后落库，只记录"成功"。

设计要点：
- 每次外部工具真实发送成功对应一条记录；混合通道场景（Gmail + Feishu 私聊 +
  Feishu 文件）会产生多条。
- 字段全部为可空的信息型，nullable=False 只落在 send_guard 一定拿得到的字段上：
  channel / recipient_kind / recipient / message_id / tool_name。
- status 只有 "success"。失败不写这张表。
"""
from sqlalchemy import Column, DateTime, Index, Integer, String, Text
from sqlalchemy.sql import func

from app.models.chat_history import Base


class SendAudit(Base):
    __tablename__ = "send_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user_id = Column(String(64), nullable=True)
    session_id = Column(String(64), nullable=True)

    channel = Column(String(16), nullable=False)          # gmail | feishu
    recipient_kind = Column(String(16), nullable=False)   # email | open_id | chat_id
    recipient = Column(String(255), nullable=False)       # 邮箱 / open_id / chat_id
    recipient_name = Column(String(255), nullable=True)   # 显示名，如"张三"、"运营组"

    subject = Column(String(255), nullable=True)
    artifact_name = Column(String(255), nullable=True)
    artifact_path = Column(String(1024), nullable=True)

    message_id = Column(String(128), nullable=False)
    file_key = Column(String(128), nullable=True)         # 仅飞书文件工具会有
    tool_name = Column(String(64), nullable=False)        # 如 send_email / send_message_to_user

    request_payload_preview = Column(Text, nullable=True) # 请求 payload 前 N 字，便于排查
    status = Column(String(16), nullable=False, default="success", server_default="success")

    __table_args__ = (
        Index("ix_send_audit_channel", "channel"),
        Index("ix_send_audit_recipient", "recipient"),
        Index("ix_send_audit_message_id", "message_id"),
        Index("ix_send_audit_created_at", "created_at"),
        Index("ix_send_audit_session_id", "session_id"),
    )
