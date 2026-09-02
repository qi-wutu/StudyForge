"""会话管理 — 创建、查询、解析

会话（Session）是 StudyForge 的隔离单元：
  每个会话对应一个学习主题（如"Go 八股"、"Redis 八股"），
  有自己的资料、知识点、答题记录，互不干扰。

当前会话 ID 由前端 localStorage 持有，每次请求带上。
后端不再维护"当前会话"状态，真正做到无状态。
"""

from storage.db import db
from storage.schemas import Session


def resolve_session_id(session_id: int | None = None) -> int:
    """解析会话 ID

    前端传了就用传的，没传就 fallback 到 default 会话。
    """
    if session_id is not None:
        s = db.query(Session).filter_by(id=session_id).first()
        if s:
            return s.id

    # fallback：default 会话
    s = db.query(Session).filter_by(name="default").first()
    if not s:
        s = Session(name="default")
        db.add(s)
        db.commit()
    return s.id


def list_sessions() -> list[dict]:
    """列出所有会话（后端不标记当前，由前端比对）"""
    sessions = db.query(Session).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "created_at": str(s.created_at),
        }
        for s in sessions
    ]


def create_session(name: str) -> dict:
    """创建新会话

    如果重名则抛 ValueError，由 router 层转成 HTTP 400。
    前端拿到返回的 id 后自己存 localStorage。
    """
    existing = db.query(Session).filter_by(name=name).first()
    if existing:
        raise ValueError(f"会话「{name}」已存在")
    s = Session(name=name)
    db.add(s)
    db.commit()
    return {"id": s.id, "name": s.name, "created_at": str(s.created_at)}


def get_session(session_id: int) -> dict:
    """按 ID 查询会话信息"""
    s = db.query(Session).filter_by(id=session_id).first()
    if not s:
        raise ValueError("会话不存在")
    return {"id": s.id, "name": s.name}
