import os
import time
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from app.db.db_config import init_db
from app.db.redis_config import connect_redis, close_redis
from app.mcp.client import mcp_manager
from app.router.chat import chat_router
from app.router.contacts import contacts_router
from app.router.health import health_router
from app.router.user import user_router

from app.services.database_session_manager import init_database_session_manager

from app.core.failed_response_register import register_exception_handlers
from app.core.rate_limit import RateLimitMiddleware
from app.core.logger_handler import logger

from app.rag.reorder_service import check_and_download_reranker_model
from app.scheduler.scheduler import start_scheduler, stop_scheduler

# 加载环境变量
load_dotenv()

app = FastAPI()

# 集成限流中间件
app.add_middleware(RateLimitMiddleware, limit=100, window=60) # 每分钟100个请求

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response

# 集成API路由
app.include_router(chat_router)
app.include_router(contacts_router)
app.include_router(health_router)
app.include_router(user_router)




# CORS 白名单：默认本地前端，生产可用 env CORS_ALLOW_ORIGINS（逗号分隔）覆盖
_default_origins = [
    "http://localhost:3000", "http://localhost:3001",
    "http://127.0.0.1:3000", "http://127.0.0.1:3001",
]
_cors_env = os.getenv("CORS_ALLOW_ORIGINS", "")
cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] or _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins, # 允许访问的源（白名单，不再用 *）
    allow_credentials=True, # 允许携带cookie
    allow_methods=["*"], # 允许的请求方法
    allow_headers=["*"], # 允许的请求头
)

# 注册异常处理函数
register_exception_handlers(app)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化会话管理器"""
    # 初始化数据库表结构
    await init_db()
    logger.info("数据库表结构初始化完成")

    # 使用数据库版本的会话管理器
    await init_database_session_manager()
    logger.info("数据库会话管理器初始化完成")

    # 初始化 MCP 客户端（读 yaml、拉工具、注册 send_guard）
    await mcp_manager.start()
    logger.info("MCP 工具管理器初始化完成")

    # 连接Redis
    await connect_redis()
    logger.info("Redis连接初始化完成")

    # 检查并重排序模型
    check_and_download_reranker_model()
    logger.info("重排序模型检查完成")

    # 启动持续更新调度器（SCHEDULER_ENABLED=true 时生效）
    start_scheduler()

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时关闭Redis连接"""
    stop_scheduler()
    await mcp_manager.shutdown()
    logger.info("MCP 工具管理器已关闭")
    await close_redis()
    logger.info("Redis连接已关闭")