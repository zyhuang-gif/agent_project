from typing import Optional

from pydantic import BaseModel, Field


class ContactCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    alias: str = ""
    email: Optional[str] = None
    feishu_open_id: Optional[str] = None
    feishu_user_id: Optional[str] = None
    is_active: bool = True
    note: str = ""


class ContactUpdate(BaseModel):
    alias: Optional[str] = None
    email: Optional[str] = None
    feishu_open_id: Optional[str] = None
    feishu_user_id: Optional[str] = None
    is_active: Optional[bool] = None
    note: Optional[str] = None


class ContactOut(BaseModel):
    id: int
    name: str
    alias: str
    email: Optional[str] = None
    feishu_open_id: Optional[str] = None
    feishu_user_id: Optional[str] = None
    is_active: bool
    note: str


class FeishuGroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    chat_id: str = Field(min_length=1, max_length=64)
    is_active: bool = True
    note: str = ""


class FeishuGroupOut(BaseModel):
    id: int
    name: str
    chat_id: str
    is_active: bool
    note: str
