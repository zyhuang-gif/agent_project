from pydantic import BaseModel
from typing import Any, Dict, List, Tuple, Optional


class QueryRequest(BaseModel):
    session_id: Optional[str] = None
    query: str


class RAGRequest(BaseModel):
    query: str


class AgentStep(BaseModel):
    thought: Optional[str] = None
    tool: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
    session_id: str
    steps: Optional[List[AgentStep]] = None


class Citation(BaseModel):
    """单条来源引用"""
    filename: str
    chunk_preview: str
    score: float
    kb_id: Optional[str] = None


class SessionMessage(BaseModel):
    """会话历史的结构化单条消息;assistant 消息会附带 citations / steps / send_report 元数据。"""
    role: str
    content: str
    citations: List[Citation] = []
    steps: List[Dict[str, Any]] = []
    send_report: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    history: List[Tuple[str, str]]
    messages: List[SessionMessage] = []
    title: Optional[str] = None
    archived: bool = False
    archived_at: Optional[str] = None


class RAGResponse(BaseModel):
    response: str
    citations: Optional[List[Citation]] = None


class ReorderRequest(BaseModel):
    query: str
    documents: List[str]


class ReorderResponse(BaseModel):
    documents: List[dict]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_size: int
    chunk_count: int
    kb_id: Optional[str] = None
    upload_time: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


# ── 知识库 ──────────────────────────────────────────────────────────────────

class KBCreateRequest(BaseModel):
    name: str
    description: str = ""
    scope: str = "personal"    # personal / dept / company
    dept_id: Optional[str] = None


class KBInfo(BaseModel):
    kb_id: str
    name: str
    description: str
    owner_id: str
    scope: str
    dept_id: Optional[str] = None
    created_at: Optional[str] = None


class KBListResponse(BaseModel):
    kbs: List[KBInfo]
    total: int


class KBMemberRequest(BaseModel):
    principal_id: str
    principal_type: str = "user"   # user / dept
    role: str = "viewer"           # viewer / editor / admin


class KBQueryRequest(BaseModel):
    query: str


class KBQueryResponse(BaseModel):
    summary: str
    citations: List[Citation]
    documents: List[str]


class KBUpdateRequest(BaseModel):
    name: str
    description: Optional[str] = None
