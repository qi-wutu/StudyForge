"""会话检索器 — 每个 session 的混合检索器（BM25 + 向量），带缓存

从 V1 的 graph/node.py 拆出（原 `_retriever_caches` / `_get_retriever`）。
多个子 Agent 共用：
  - review_agent.question_gen  → 扩散相关知识点 / 出题参考
  - qa_agent.answer           → 检索资料回答
  - import_agent              → 知识点变更后 invalidate，下次重建
"""

from storage.db import db
from storage.schemas import KnowledgePoint
from rag.hybrid import HybridRetriever

# 每个 session 的 HybridRetriever，按 session_id 缓存；知识点变更时 invalidate
_retriever_caches: dict[int, HybridRetriever] = {}


def get_session_retriever(session_id: int) -> HybridRetriever:
    """获取当前会话的混合检索器

    第一次调用时从数据库加载知识点并构建 BM25 + 向量索引。
    """
    if session_id not in _retriever_caches:
        kps = db.query(KnowledgePoint).filter_by(session_id=session_id).all()
        retriever = HybridRetriever()
        if kps:
            docs = [kp.content for kp in kps]
            embeddings = [kp.embedding for kp in kps]
            has_embeddings = all(e is not None for e in embeddings)
            retriever.build(docs, embeddings if has_embeddings else None)
        _retriever_caches[session_id] = retriever
    return _retriever_caches[session_id]


def invalidate_retriever(session_id: int):
    """知识点变更后调用，让该 session 的检索器下次重建"""
    _retriever_caches.pop(session_id, None)
