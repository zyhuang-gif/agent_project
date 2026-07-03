import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from app.db.db_config import AsyncSessionLocal
from app.models.chat_history import ChatSession, ChatMessage
from app.core.logger_handler import logger


class DatabaseSessionManager:
    """基于数据库的会话管理器"""

    def __init__(self):
        """初始化会话管理器"""
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls) -> "DatabaseSessionManager":
        """
        异步创建并初始化 DatabaseSessionManager
        :return: 初始化完成的 DatabaseSessionManager 实例
        """
        instance = cls()
        logger.info("【数据库会话管理】初始化完成")
        return instance

    async def get_session(self, session_id: str, user_id: str) -> Dict:
        """获取会话"""
        async with AsyncSessionLocal() as db:
            # 尝试查找会话，验证属于该用户
            result = await db.run_sync(
                lambda session: session.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
            )

            if result:
                # 获取会话历史
                messages = await db.run_sync(
                    lambda session: session.query(ChatMessage).filter(ChatMessage.session_id == result.id).order_by(ChatMessage.created_at).all()
                )
                # 转换为 (user_message, assistant_message) 格式
                history = []
                i = 0
                while i < len(messages):
                    if messages[i].role == "user" and i + 1 < len(messages) and messages[i+1].role == "assistant":
                        history.append((messages[i].content, messages[i+1].content))
                        i += 2
                    else:
                        i += 1
                # 构建结构化消息列表(供前端切回会话时还原 citations / steps / send_report)
                structured_messages = []
                for m in messages:
                    item = {"role": m.role, "content": m.content}
                    if m.role == "assistant":
                        meta = m.metadata_ or {}
                        item["citations"] = list(meta.get("citations") or [])
                        item["steps"] = list(meta.get("steps") or [])
                        item["send_report"] = meta.get("send_report")
                    structured_messages.append(item)
                return {
                    "history": history,
                    "messages": structured_messages,
                    "title": result.title,
                    "archived": result.archived,
                    "archived_at": result.archived_at.isoformat() if result.archived_at else None
                }
            else:
                # 检查会话id是否存在
                existing_session = await db.run_sync(
                    lambda session: session.query(ChatSession).filter(ChatSession.id == session_id).first()
                )
                
                if existing_session:
                    # 会话存在但不属于当前用户
                    logger.warning(f"【数据库会话管理】会话 {session_id} 不属于用户 {user_id}")
                    from fastapi import HTTPException, status
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="当前会话不属于你"
                    )
                else:
                    # 会话不存在，创建一个新的
                    new_session = ChatSession(
                        id=session_id,
                        user_id=user_id,
                        title="新的对话"
                    )
                    db.add(new_session)
                    await db.commit()
                    await db.refresh(new_session)
                    logger.info(f"【数据库会话管理】创建新会话: {session_id} 属于用户: {user_id}")
                    return {
                        "history": [],
                        "messages": [],
                        "title": new_session.title,
                        "archived": False,
                        "archived_at": None
                    }

    async def ensure_session_writable(self, session_id: str, user_id: str):
        """确认会话可继续写入；归档会话只能查看。"""
        if not session_id:
            return

        async with AsyncSessionLocal() as db:
            session = await db.run_sync(
                lambda session: session.query(ChatSession).filter(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user_id
                ).first()
            )

            if session and session.archived:
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="归档会话不能继续对话，请先取消归档"
                )

    async def add_message(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        citations: Optional[List[Dict]] = None,
        steps: Optional[List[Dict]] = None,
        send_report: Optional[str] = None,
    ):
        """添加消息并保存到数据库。

        :param citations: 检索引用列表(可选)。提供时与 steps/send_report 一起写入 assistant metadata_。
        :param steps: agent 执行步骤列表(可选,前端 SSE schema 格式 {id, level, status, title, detail})。
        :param send_report: send 节点的可见发送结果(可选)。P0 Gmail 发送路径会填充。
        """
        async with AsyncSessionLocal() as db:
            # 检查会话id是否存在
            existing_session = await db.run_sync(
                lambda session: session.query(ChatSession).filter(ChatSession.id == session_id).first()
            )

            if existing_session:
                # 检查会话是否属于当前用户
                if existing_session.user_id != user_id:
                    # 会话存在但不属于当前用户，不添加消息
                    logger.warning(f"【数据库会话管理】会话 {session_id} 不属于用户 {user_id}，无法添加消息")
                    from fastapi import HTTPException, status
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="当前会话不属于你，无法添加消息"
                    )
                if existing_session.archived:
                    logger.warning(f"【数据库会话管理】会话 {session_id} 已归档，拒绝追加消息")
                    from fastapi import HTTPException, status
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="归档会话不能继续对话，请先取消归档"
                    )
                session = existing_session
            else:
                # 会话不存在，创建一个新的
                session = ChatSession(
                    id=session_id,
                    user_id=user_id,
                    title="新的对话"
                )
                db.add(session)
                await db.commit()
                await db.refresh(session)

            # 检查是否是新会话且标题为默认值，如果是则更新为用户的第一个问题
            if session.title == "新的对话":
                # 生成用户问题的摘要作为标题（截取前30个字符）
                title_summary = user_message[:30].strip()
                if len(user_message) > 30:
                    title_summary += "..."
                session.title = title_summary

            # 添加用户消息
            user_msg = ChatMessage(
                session_id=session.id,
                role="user",
                content=user_message
            )
            db.add(user_msg)

            # 添加助手消息;若调用方提供了 citations / steps / send_report,则一并写入 metadata_
            assistant_metadata = None
            if citations is not None or steps is not None or send_report is not None:
                assistant_metadata = {
                    "citations": list(citations or []),
                    "steps": list(steps or []),
                    "send_report": send_report,
                }
            assistant_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=assistant_message,
                metadata_=assistant_metadata,
            )
            db.add(assistant_msg)

            await db.commit()
            logger.info(f"【数据库会话管理】添加消息到会话: {session_id} 属于用户: {user_id}")

    async def get_history(self, session_id: str, user_id: str) -> List[Tuple[str, str]]:
        """获取会话历史"""
        session_data = await self.get_session(session_id, user_id)
        return session_data.get("history", [])

    async def clear_session(self, session_id: str, user_id: str):
        """清除会话"""
        async with AsyncSessionLocal() as db:
            # 查找会话，验证属于该用户
            session = await db.run_sync(
                lambda session: session.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
            )

            if session:
                # 删除会话（级联删除消息）
                await db.delete(session)
                await db.commit()
                logger.info(f"【数据库会话管理】会话 {session_id} 已清除，属于用户: {user_id}")

    async def set_session_archived(self, session_id: str, user_id: str, archived: bool) -> Dict:
        """设置会话归档状态"""
        async with AsyncSessionLocal() as db:
            session = await db.run_sync(
                lambda session: session.query(ChatSession).filter(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user_id
                ).first()
            )

            if not session:
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="会话不存在"
                )

            session.archived = archived
            session.archived_at = datetime.utcnow() if archived else None
            await db.commit()
            await db.refresh(session)
            logger.info(
                f"【数据库会话管理】会话 {session_id} 归档状态更新为 {archived}，属于用户: {user_id}"
            )
            return {
                "id": session.id,
                "title": session.title,
                "archived": session.archived,
                "archived_at": session.archived_at.isoformat() if session.archived_at else None,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            }

    async def get_all_session_ids(self, user_id: Optional[str] = None) -> List[str]:
        """获取所有会话 ID，如果提供了 user_id，则只返回该用户的会话"""
        async with AsyncSessionLocal() as db:
            if user_id:
                sessions = await db.run_sync(
                    lambda session: session.query(ChatSession).filter(ChatSession.user_id == user_id).all()
                )
            else:
                sessions = await db.run_sync(
                    lambda session: session.query(ChatSession).all()
                )
            return [session.id for session in sessions]

    async def get_user_sessions(self, user_id: str, archived: bool = False) -> List[Dict]:
        """获取用户所有会话详细信息"""
        async with AsyncSessionLocal() as db:
            sessions = await db.run_sync(
                lambda session: session.query(ChatSession).filter(
                    ChatSession.user_id == user_id,
                    ChatSession.archived == archived
                ).order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc()).all()
            )
            return [
                {
                    "id": session.id,
                    "title": session.title,
                    "archived": session.archived,
                    "archived_at": session.archived_at.isoformat() if session.archived_at else None,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "updated_at": session.updated_at.isoformat() if session.updated_at else None
                }
                for session in sessions
            ]


# 全局数据库会话管理器实例
database_session_manager = None

# 初始化数据库会话管理器
async def init_database_session_manager():
    """
    初始化数据库会话管理器
    :return: 初始化完成的 DatabaseSessionManager 实例
    """
    global database_session_manager
    database_session_manager = await DatabaseSessionManager.create()
    return database_session_manager
