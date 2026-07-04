import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text
from app.models.chat_history import Base

# 加载环境变量
load_dotenv()

# 数据库URL
ASYNC_DATABSE_URL = f"mysql+aiomysql://{os.getenv('MYSQL_USER', 'root')}:{os.getenv('MYSQL_PASSWORD', '')}@{os.getenv('MYSQL_HOST', 'localhost')}:{os.getenv('MYSQL_PORT', '3306')}/{os.getenv('MYSQL_DATABASE', 'chat_history')}?charset=utf8mb4"

# 创建异步引擎
async_engine = create_async_engine(
    ASYNC_DATABSE_URL,
    pool_size=10, # 连接池中保持的持久连接数
    max_overflow=20, # 连接池中允许创建的额外连接数
    pool_pre_ping=True, # 取连接前先探活，避免拿到被 MySQL wait_timeout 回收的死连接
    pool_recycle=3600, # 连接最长存活 1 小时后主动回收，规避空闲断连
    echo=os.getenv("DB_ECHO", "false").lower() == "true", # SQL 日志默认关闭，DB_ECHO=true 开启
)

# 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 初始化数据库，创建所有表
async def init_db():
    from app.models import contact  # noqa: F401 —— 让 Base.metadata 注册 contact/FeishuGroup

    async with async_engine.begin() as conn:
        # 先删除旧表，然后创建新表
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_chat_session_archive_columns(conn)


async def _ensure_chat_session_archive_columns(conn):
    """为已有 MySQL 表补齐会话归档字段。"""
    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_NAME = 'chat_sessions' "
            "AND COLUMN_NAME IN ('archived', 'archived_at')"
        )
    )
    existing_columns = {row[0] for row in result.fetchall()}

    if "archived" not in existing_columns:
        await conn.execute(
            text("ALTER TABLE chat_sessions ADD COLUMN archived BOOLEAN NOT NULL DEFAULT FALSE")
        )
        await conn.execute(text("CREATE INDEX ix_chat_sessions_archived ON chat_sessions (archived)"))

    if "archived_at" not in existing_columns:
        await conn.execute(
            text("ALTER TABLE chat_sessions ADD COLUMN archived_at DATETIME NULL")
        )

# 依赖项
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()

        except Exception:
            await session.rollback()
            raise

        finally:
            await session.close()




async def check_mysql_connection() -> bool:
    """检查MySQL连接"""
    try:
        async with async_engine.connect() as conn:
            # 执行简单查询
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"MySQL连接失败: {e}")
        return False
