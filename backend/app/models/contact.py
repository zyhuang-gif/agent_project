from sqlalchemy import Boolean, Column, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from app.models.chat_history import Base


class Contact(Base):
    """企业内可发送对象（P0 邮件；P2 飞书私聊）。"""
    __tablename__ = "contacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, unique=True, index=True)
    alias = Column(String(255), nullable=False, default="")
    email = Column(String(255), nullable=True, index=True)
    feishu_open_id = Column(String(64), nullable=True, index=True)
    feishu_user_id = Column(String(64), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, server_default="1", index=True)
    note = Column(String(255), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class FeishuGroup(Base):
    """飞书群（P2 用）。P0 建表以便早期填充数据。"""
    __tablename__ = "feishu_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, unique=True, index=True)
    chat_id = Column(String(64), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True, server_default="1", index=True)
    note = Column(String(255), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("name", name="uk_feishu_group_name"),
        UniqueConstraint("chat_id", name="uk_feishu_group_chat_id"),
    )
