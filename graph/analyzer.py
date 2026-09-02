"""薄弱分析模块

从 review_records 读取答题记录，分析薄弱知识点：
  1. 按 KP 分组统计（平均分、次数、趋势）
  2. 全局 weakness / missing_kps 高频统计
  3. LLM 生成自然语言薄弱分析报告

后续（第二层）：BM25 薄弱扩散 → 调度器优先出题
"""

from collections import Counter
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from graph.node import get_llm
from storage.db import db
from storage.schemas import KnowledgePoint, ReviewRecord


# ========================================
# 一、数据查询
# ========================================

def get_kp_map(session_id: int) -> dict[int, dict]:
    """构建 kp_id → {title, content} 映射"""
    kps = db.query(KnowledgePoint).filter_by(session_id=session_id).all()
    return {kp.id: {"title": kp.title, "content": kp.content} for kp in kps}


def get_review_records(session_id: int) -> list:
    """获取当前会话所有答题记录，按时间升序"""
    return (
        db.query(ReviewRecord)
        .filter_by(session_id=session_id)
        .order_by(ReviewRecord.created_at.asc())
        .all()
    )


# ========================================
# 二、统计计算
# ========================================

def compute_kp_stats(records: list) -> list[dict]:
    """按知识点分组统计

    返回按平均分升序（最弱在前）的列表：
      {kp_id, title, avg_score, attempts, latest_score, score_trend, weaknesses, missing_kps}
    """
    groups: dict[int, dict] = {}

    for r in records:
        if r.kp_id not in groups:
            groups[r.kp_id] = {
                "kp_id": r.kp_id,
                "scores": [],
                "all_weaknesses": [],
                "all_missing_kps": [],
            }
        g = groups[r.kp_id]
        g["scores"].append(r.ai_score or 0)
        g["all_weaknesses"].extend(r.ai_weaknesses or [])
        g["all_missing_kps"].extend(r.ai_missing_kps or [])

    # 计算统计量
    stats = []
    for kp_id, g in groups.items():
        scores = g["scores"]
        avg = sum(scores) / len(scores)
        latest = scores[-1]
        # 趋势：至少 3 次才判断，否则标注 "数据不足"
        if len(scores) >= 3:
            # 比较最近 1/3 和最早 1/3 的平均分
            third = max(1, len(scores) // 3)
            early_avg = sum(scores[:third]) / third
            recent_avg = sum(scores[-third:]) / third
            diff = recent_avg - early_avg
            if diff > 5:
                trend = "上升 ↑"
            elif diff < -5:
                trend = "下降 ↓"
            else:
                trend = "稳定 →"
        else:
            trend = "数据不足"

        # 高频 weakness（取 top 3）
        weak_counter = Counter(g["all_weaknesses"])
        top_weaknesses = [item for item, _ in weak_counter.most_common(3)]

        # 高频 missing_kps（取 top 3）
        missing_counter = Counter(g["all_missing_kps"])
        top_missing = [item for item, _ in missing_counter.most_common(3)]

        stats.append({
            "kp_id": kp_id,
            "avg_score": round(avg, 1),
            "attempts": len(scores),
            "latest_score": latest,
            "score_trend": trend,
            "top_weaknesses": top_weaknesses,
            "top_missing_kps": top_missing,
        })

    # 最弱的排前面
    stats.sort(key=lambda x: x["avg_score"])
    return stats


def aggregate_global(records: list) -> dict:
    """全局统计：所有记录汇总的 weakness / missing_kps 词频"""
    all_weaknesses = []
    all_missing = []
    for r in records:
        all_weaknesses.extend(r.ai_weaknesses or [])
        all_missing.extend(r.ai_missing_kps or [])

    return {
        "weakness_freq": Counter(all_weaknesses).most_common(10),
        "missing_kps_freq": Counter(all_missing).most_common(10),
        "total_records": len(records),
        "avg_score_all": round(
            sum(r.ai_score or 0 for r in records) / len(records), 1
        ) if records else 0,
    }


# ========================================
# 三、LLM 报告生成
# ========================================

def generate_llm_report(
    kp_stats: list[dict],
    global_stats: dict,
    kp_map: dict[int, dict],
) -> str:
    """调 DeepSeek 生成自然语言薄弱分析报告"""
    if not kp_stats:
        return "暂无答题记录，无法生成分析报告。"

    # 构建统计摘要
    stats_lines = ["## 各知识点得分排行（最弱在前）\n"]
    for s in kp_stats:
        title = kp_map.get(s["kp_id"], {}).get("title", f"KP#{s['kp_id']}")
        stats_lines.append(
            f"- {title}: {s['avg_score']}分 "
            f"(共{s['attempts']}次, 趋势{s['score_trend']})"
        )

    stats_lines.append(f"\n## 全局统计\n")
    stats_lines.append(f"- 总答题数: {global_stats['total_records']}")
    stats_lines.append(f"- 全局平均分: {global_stats['avg_score_all']}")

    stats_lines.append(f"\n## 高频弱点\n")
    for w, c in global_stats["weakness_freq"]:
        stats_lines.append(f"- 「{w}」出现 {c} 次")

    stats_lines.append(f"\n## 高频缺失知识点\n")
    for m, c in global_stats["missing_kps_freq"]:
        stats_lines.append(f"- 「{m}」出现 {c} 次")

    summary = "\n".join(stats_lines)

    prompt = f"""你是一个学习分析顾问。请根据以下用户的学习统计，写一份中文薄弱分析报告。

要求：
1. 语言简洁、一针见血，不要套话
2. 指出最薄弱的知识领域, 点出具体薄弱点
3. 给出可操作的复习建议
4. 150-300 字

统计数据：
{summary}"""

    response = get_llm().invoke([
        SystemMessage(content="你是学习分析专家，输出精炼的中文分析报告。"),
        HumanMessage(content=prompt),
    ])

    return response.content


# ========================================
# 四、入口函数
# ========================================

# ========================================
# 五、CLI 展示
# ========================================

def display_analysis(result: dict):
    """在终端展示分析报告"""
    if "error" in result:
        print(f"[!] {result['error']}")
        return

    global_stats = result["global_stats"]
    kp_stats = result["kp_stats"]

    # === 总览 ===
    print("=" * 50)
    print("[薄弱点分析报告]")
    print("=" * 50)
    print(f"总答题数：{global_stats['total_records']}")
    print(f"全局平均分：{global_stats['avg_score_all']}")

    # 薄弱层级
    weak = [s for s in kp_stats if s["avg_score"] < 60]
    mid = [s for s in kp_stats if 60 <= s["avg_score"] < 75]
    strong = [s for s in kp_stats if s["avg_score"] >= 75]
    print(f"薄弱 KPs（<60分）：{len(weak)} 个")
    print(f"待加强 KPs（60-75分）：{len(mid)} 个")
    print(f"良好 KPs（≥75分）：{len(strong)} 个")

    # === 薄弱知识点排行榜 ===
    print(f"\n{'=' * 50}")
    print("[薄弱知识点排行榜]")
    print(f"{'=' * 50}")
    if weak:
        print(f"{'排名':>4} {'知识点':<24} {'均分':>6} {'次数':>4} {'趋势':>8}")
        print(f"{'-' * 50}")
        for i, s in enumerate(weak, 1):
            print(f"{i:>4} {s['title'][:22]:<24} {s['avg_score']:>6} {s['attempts']:>4} {s['score_trend']:>8}")
            if s["top_missing_kps"]:
                missing_str = "、".join(s["top_missing_kps"])
                print(f"      [缺失] {missing_str}")
    else:
        print("  暂无薄弱知识点！")

    # === 待加强知识点 ===
    if mid:
        print(f"\n{'-' * 50}")
        print("[待加强知识点]")
        print(f"{'-' * 50}")
        for i, s in enumerate(mid, 1):
            print(f"  {i}. {s['title']}  —  {s['avg_score']}分（{s['attempts']}次, {s['score_trend']}）")

    # === 高频弱点 ===
    print(f"\n{'-' * 50}")
    print("[高频弱点 Top 10]")
    print(f"{'-' * 50}")
    for i, (w, c) in enumerate(global_stats["weakness_freq"], 1):
        print(f"  {i:>2}. 「{w}」 x {c}")

    # === 高频缺失知识点 ===
    print(f"\n{'-' * 50}")
    print("[高频缺失知识点 Top 10]")
    print(f"{'-' * 50}")
    for i, (m, c) in enumerate(global_stats["missing_kps_freq"], 1):
        print(f"  {i:>2}. 「{m}」 x {c}")

    # === LLM 报告 ===
    if "llm_report" in result:
        print(f"\n{'=' * 50}")
        print("[AI 分析报告]")
        print(f"{'=' * 50}")
        print(result["llm_report"])
    print()


def analyze(session_id: int, llm_report: bool = True) -> dict:
    """执行薄弱分析

    Args:
        session_id: 会话 ID
        llm_report: 是否调 LLM 生成自然语言报告

    Returns:
        { kp_stats, global_stats, llm_report(可选) }
    """
    records = get_review_records(session_id)
    if not records:
        return {"error": "暂无答题记录"}

    kp_map = get_kp_map(session_id)
    kp_stats = compute_kp_stats(records)
    global_stats = aggregate_global(records)

    # 给 kp_stats 补上 title
    for s in kp_stats:
        s["title"] = kp_map.get(s["kp_id"], {}).get("title", f"KP#{s['kp_id']}")

    result = {
        "kp_stats": kp_stats,
        "global_stats": global_stats,
    }

    if llm_report:
        result["llm_report"] = generate_llm_report(kp_stats, global_stats, kp_map)

    return result
