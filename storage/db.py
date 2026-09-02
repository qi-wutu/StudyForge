"""SQLAlchemy 数据库操作

三步走：
  1. 配置连接 → engine（引擎）
  2. 创建会话类 → SessionLocal（类似连接池）
  3. scoped_session → db（线程安全的全局会话）
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from storage.schemas import Base


# === 1. engine（引擎） ===
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_size=5, pool_pre_ping=True)


# === 2. SessionLocal（会话工厂） ===
SessionLocal = sessionmaker(bind=engine)


# === 3. db（线程安全的全局会话） ===
# scoped_session 给每个线程分配独立的 session
# 并发请求不会共用一个数据库连接，避免 pymysql 包序错乱
db = scoped_session(SessionLocal)


def init_db():
    """自动建表 + 增量迁移字段"""
    Base.metadata.create_all(engine)

    # 增量迁移：新增字段
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE knowledge_points ADD COLUMN embedding JSON NULL"
            ))
            conn.commit()
        print("  [迁移] 添加 knowledge_points.embedding 字段")
    except Exception:
        pass  # 字段已存在或表不存在，忽略
