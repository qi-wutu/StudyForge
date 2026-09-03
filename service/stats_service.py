"""统计与分析 — 知识点查询、Dashboard 数据、薄弱分析

只做读操作（SELECT + 聚合），不涉及 LangGraph。

三个功能：
  1. list_knowledge_points  → 知识点列表 + 每道题的平均分/答题次数
  2. get_dashboard_stats    → 首页概览计数（总知识点数、答题数、平均分……）
  3. analyze                → 薄弱分析（数据聚合 + 可选的 LLM 分析报告）

缓存策略：
  analyze 和 get_dashboard_stats 的结果会缓存 60 秒。
  如果期间有新的答题记录（review_records 的最大 ID 变了），缓存立即失效。
"""

import time

from sqlalchemy import func

from agent.analyzer import analyze as _run_analyze
from storage.db import db
from storage.schemas import Document, KnowledgePoint, ReviewRecord, Session


# ===== 分析结果缓存 =====
# {session_id: {"data": ..., "max_record_id": int, "cached_at": float}}

_analyze_cache: dict[int, dict] = {}
_dashboard_cache: dict[int, dict] = {}
_CACHE_TTL = 900  # 秒（15 分钟）


def _get_max_record_id(session_id: int) -> int:
    """获取当前会话最新的答题记录 ID，用来判断数据是否变了"""
    row = (
        db.query(func.max(ReviewRecord.id))
        .filter_by(session_id=session_id)
        .first()
    )
    return row[0] or 0


def _is_cache_valid(entry: dict | None, session_id: int) -> bool:
    """判断缓存是否有效：TTL 内 + 没有新增答题记录"""
    if entry is None:
        return False
    if time.time() - entry["cached_at"] > _CACHE_TTL:
        return False
    if _get_max_record_id(session_id) != entry["max_record_id"]:
        return False
    return True


def list_knowledge_points(session_id: int) -> list[dict]:
    """列出当前会话的所有知识点（含历史答题统计）

    两张表 JOIN：
      knowledge_points  ← 知识点列表
      review_records    ← 聚合出 avg_score + review_count

    没有答题记录的知识点也会返回（avg_score=null, review_count=0）。
    """
    kps = db.query(KnowledgePoint).filter_by(session_id=session_id).all()

    # 一次性查出所有知识点的平均分和答题次数
    # 避免逐条查的 N+1 问题
    stats_rows = (
        db.query(
            ReviewRecord.kp_id,
            func.avg(ReviewRecord.ai_score),
            func.count(ReviewRecord.id),
        )
        .filter_by(session_id=session_id)
        .group_by(ReviewRecord.kp_id)
        .all()
    )
    # 转成 {kp_id: (avg_score, count)} 的字典方便查找
    stats = {row[0]: (row[1], row[2]) for row in stats_rows}

    return [
        {
            "id": kp.id,
            "title": kp.title,
            "content": kp.content,
            "avg_score": round(stats.get(kp.id, (None,))[0], 1) if kp.id in stats else None,
            "review_count": stats.get(kp.id, (0,))[1] if kp.id in stats else 0,
        }
        for kp in kps
    ]


def get_dashboard_stats(session_id: int) -> dict:
    """Dashboard 概览统计

    一次查 5 张表/聚合，拼成一个 JSON 对象返回。
    前端首页的 5 个统计卡片就用这个数据。

    缓存 60 秒，有新的答题记录时自动失效。
    """
    global _dashboard_cache
    cached = _dashboard_cache.get(session_id)
    if _is_cache_valid(cached, session_id):
        return cached["data"]
    session = db.query(Session).filter_by(id=session_id).first()

    kp_count = db.query(KnowledgePoint).filter_by(session_id=session_id).count()
    review_count = db.query(ReviewRecord).filter_by(session_id=session_id).count()

    avg_row = db.query(func.avg(ReviewRecord.ai_score)).filter_by(session_id=session_id).first()
    avg_score = round(avg_row[0], 1) if avg_row and avg_row[0] else None
    doc_count = db.query(Document).filter_by(session_id=session_id).count()

    # 薄弱知识点 = 平均分 < 60 的知识点
    weak_count = 0
    if review_count > 0:
        weak_count = (
            db.query(ReviewRecord.kp_id)
            .filter_by(session_id=session_id)
            .group_by(ReviewRecord.kp_id)
            .having(func.avg(ReviewRecord.ai_score) < 60)
            .count()
        )

    result = {
        "session_name": session.name if session else "未设置",
        "kp_count": kp_count,
        "review_count": review_count,
        "avg_score": avg_score,
        "doc_count": doc_count,
        "weak_kp_count": weak_count,
    }
    # 写入缓存
    _dashboard_cache[session_id] = {
        "data": result,
        "max_record_id": _get_max_record_id(session_id),
        "cached_at": time.time(),
    }
    return result


def _clear_caches(session_id: int):
    """清除某个会话的缓存（由上层在数据变更时调用）"""
    _analyze_cache.pop(session_id, None)
    _dashboard_cache.pop(session_id, None)


def analyze(session_id: int, llm_report: bool = True) -> dict:
    """薄弱分析 — 委托给 agent.analyzer

    Args:
        session_id: 当前会话 ID
        llm_report: 是否生成 LLM 分析报告

    Returns:
        {"kp_stats": [...], "global_stats": {...}, "llm_report": "..."}

    缓存 60 秒，有新的答题记录时自动失效。
    """
    global _analyze_cache
    cached = _analyze_cache.get(session_id)
    if _is_cache_valid(cached, session_id):
        return cached["data"]

    result = _run_analyze(session_id, llm_report=llm_report)
    if "error" not in result:
        _analyze_cache[session_id] = {
            "data": result,
            "max_record_id": _get_max_record_id(session_id),
            "cached_at": time.time(),
        }
    return result
