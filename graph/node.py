"""LangGraph 节点定义

一个节点 = 图里的一个"工位"。
每个节点独立完成一件事：调 LLM、查数据库、或者等用户输入。
"""

from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from graph.state import AgentState
from storage.db import db
from storage.schemas import KnowledgePoint, ReviewRecord
from rag.bm25 import BM25Index
from rag.hybrid import HybridRetriever
from tools.tools import search_web


_llm = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    return _llm


# ===== 条件边函数 =====

def scheduler_should_continue(state: AgentState) -> str:
    if state.get("exit_review"):
        return "end"
    return "question_gen"


def judge_should_continue(state: AgentState) -> str:
    if state.get("exit_review"):
        return "end"
    return "scheduler"


# ===== 混合检索缓存 =====
# 每个 session 的 HybridRetriever（BM25 + 向量），按 session_id 缓存
# 知识点变更时重建
_retriever_caches: dict[int, HybridRetriever] = {}


def _get_retriever(session_id: int) -> HybridRetriever:
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


# ===== 智能调度（双车道） =====

def _compute_weak_kp_ids(sid: int) -> set[int]:
    """从 review_records 找出薄弱知识点（平均分 < 60）"""
    from sqlalchemy import func
    rows = (
        db.query(ReviewRecord.kp_id)
        .filter_by(session_id=sid)
        .group_by(ReviewRecord.kp_id)
        .having(func.avg(ReviewRecord.ai_score) < 60)
        .all()
    )
    return {r.kp_id for r in rows}


def _build_review_queue(sid: int, all_kps: list[KnowledgePoint]) -> list[dict]:
    """构建双车道出题队列

    快车道：薄弱 KPs + BM25 扩散发现的关联 KPs
    慢车道：剩余正常 KPs
    合并：2:1 交错（快:慢）

    返回：[{kp_id, reason, lane}, ...]
    """
    weak_ids = _compute_weak_kp_ids(sid)

    if not weak_ids:
        # 没有薄弱点，按原顺序
        return [{"kp_id": kp.id, "reason": "常规复习", "lane": "normal"}
                for kp in all_kps]

    from sqlalchemy import func

    # ===== Lane 1: 薄弱 KPs 本身（最弱排最前） =====
    avg_scores = dict(
        db.query(ReviewRecord.kp_id, func.avg(ReviewRecord.ai_score))
        .filter(ReviewRecord.kp_id.in_(weak_ids))
        .group_by(ReviewRecord.kp_id)
        .all()
    )
    weak_kps = sorted(
        [kp for kp in all_kps if kp.id in weak_ids],
        key=lambda kp: avg_scores.get(kp.id, 0),
    )
    lane1 = []
    for wk in weak_kps:
        score = avg_scores.get(wk.id, 0)
        lane1.append({
            "kp_id": wk.id,
            "reason": f"薄弱知识点（均分 {score:.0f}）",
            "lane": "weak",
        })

    # ===== Lane 1 续: 混合检索扩散（关联但不弱的 KPs） =====
    retriever = _get_retriever(sid)
    related: dict[int, float] = {}  # kp_id -> max score

    for wk in weak_kps:
        results = retriever.search(wk.content, top_k=10)
        for idx, _, score in results:
            if score < 0.3:
                continue
            kp = all_kps[idx]
            if kp.id not in weak_ids and kp.id not in related:
                related[kp.id] = score

    for rid in sorted(related, key=lambda x: related[x], reverse=True):
        lane1.append({
            "kp_id": rid,
            "reason": f"相关薄弱（BM25 {related[rid]:.2f}）",
            "lane": "weak",
        })

    # ===== Lane 2: 剩余正常 KPs =====
    lane1_ids = {e["kp_id"] for e in lane1}
    lane2 = [
        {"kp_id": kp.id, "reason": "常规复习", "lane": "normal"}
        for kp in all_kps if kp.id not in lane1_ids
    ]

    # ===== 合并：2 快 + 1 慢 =====
    queue = []
    l1, l2 = 0, 0
    while l1 < len(lane1) or l2 < len(lane2):
        for _ in range(2):
            if l1 < len(lane1):
                queue.append(lane1[l1])
                l1 += 1
        if l2 < len(lane2):
            queue.append(lane2[l2])
            l2 += 1

    return queue


# ===== Memory — 历史答题上下文 =====

def _build_memory_context(sid: int, kp_id: int, max_history: int = 3) -> str:
    """查 KPs 的历史答题记录，构建 Memory 上下文字符串

    让 LLM 知道"用户之前答过什么、哪里弱、有没有进步"。
    """
    records = (
        db.query(ReviewRecord)
        .filter_by(session_id=sid, kp_id=kp_id)
        .order_by(ReviewRecord.created_at.desc())
        .limit(max_history)
        .all()
    )
    if not records:
        return ""

    lines = ["\n## 历史答题记录（按时间倒序）\n"]
    for i, r in enumerate(records, 1):
        weaknesses = "、".join(r.ai_weaknesses or [])
        missing = "、".join(r.ai_missing_kps or [])
        lines.append(f"[第{r.id}次] 得分 {r.ai_score}")
        if weaknesses:
            lines.append(f"  不足：{weaknesses}")
        if missing:
            lines.append(f"  缺失：{missing}")

    return "\n".join(lines)


# ===== 轻量 ReAct 辅助函数 =====

def _react_call(prompt: str, system: str, max_turns: int = 5) -> dict:
    """轻量 ReAct — LLM 带标准工具调用

    使用 bind_tools 绑定搜索工具，LLM 可自主决定搜索。
    支持 DeepSeek、OpenAI、Claude 等标准 tool calling 协议。
    不依赖 LangGraph 的 ToolNode，节点内部自闭环。

    Args:
        prompt: 用户消息
        system: 系统提示
        max_turns: 最大工具调用轮数（防止死循环）

    Returns:
        解析后的 JSON dict
    """
    llm = get_llm().bind_tools([search_web])
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]

    for _ in range(max_turns):
        response = llm.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  [ReAct] LLM 调用了 {tc['name']}({tc['args']})")
                result = search_web.invoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # 没有工具调用 → 这是最终输出
            parser = JsonOutputParser()
            return parser.parse(response.content)

    # 超过最大轮数 → 保留搜索历史，强制输出 JSON
    print(f"  [ReAct] 超过 {max_turns} 轮，基于已获取资料强制输出")
    messages.append(HumanMessage(content="基于以上所有搜索结果，直接输出最终的 JSON 结果。不要搜索。"))
    final = get_llm().invoke(messages)
    parser = JsonOutputParser()
    return parser.parse(final.content)


# ===== 节点 =====

def scheduler(state: AgentState) -> AgentState:
    """从数据库取出下一个待复习知识点（智能调度）

    1. 如果队列为空或已耗尽 → 重建队列
    2. 从队列取下一个知识点
    """
    sid = state.get("session_id", 0)
    all_kps = db.query(KnowledgePoint).filter_by(session_id=sid).all()
    if not all_kps:
        return {"exit_review": True}

    queue = state.get("review_queue")
    pos = state.get("queue_pos", 0)

    if not queue or pos >= len(queue):
        queue = _build_review_queue(sid, all_kps)
        pos = 0

    if not queue:
        return {"exit_review": True}

    entry = queue[pos]
    kp = next(kp for kp in all_kps if kp.id == entry["kp_id"])

    return {
        "current_kp": {"id": kp.id, "title": kp.title, "content": kp.content},
        "review_queue": queue,
        "queue_pos": pos + 1,
        "review_reason": entry.get("reason", ""),
        "exit_review": False,
    }


def planner(state: AgentState) -> AgentState:
    """提取知识点并存入数据库（BM25 语义去重）"""
    sid = state.get("session_id", 0)

    # 构建已有知识点的 BM25 索引做语义去重
    existing_kps = db.query(KnowledgePoint).filter_by(session_id=sid).all()
    existing_titles = [kp.title for kp in existing_kps]
    existing_contents = [kp.content for kp in existing_kps]

    if existing_contents:
        dedup_idx = BM25Index()
        dedup_idx.build(existing_contents)
    else:
        dedup_idx = None

    prompt = f"""你是知识点提取专家。请从以下资料中提取核心知识点。

要求：
1. 每个知识点包括：title（名称）、content（简要说明）
2. 按逻辑顺序排列
3. 返回 JSON 数组格式，不要多余文字
4. 以下是你已经提取过的知识点，不要提取重复或高度相似的内容

已有知识点：
{chr(10).join(f'  - {t}' for t in existing_titles) if existing_titles else '  （暂无）'}

示例输出：
[{{"title": "Go GMP 模型", "content": "Goroutine、M、P 三者的关系"}}]

资料内容：
{state["raw_content"]}"""

    response = get_llm().invoke([
        SystemMessage(content="你只输出 JSON，不要输出其他内容。"),
        HumanMessage(content=prompt),
    ])

    parser = JsonOutputParser()
    kps = parser.parse(response.content)

    # BM25 二次去重：LLM 可能漏判，这里补充做语义相似度过滤
    for kp in kps:
        if dedup_idx and existing_contents:
            results = dedup_idx.search(kp["content"], top_k=1)
            if results and results[0][2] > 0.8:
                print(f'  [!] BM25 去重跳过："{kp["title"]}"（相似度 {results[0][2]:.2f}）')
                continue

        db.add(KnowledgePoint(
            session_id=sid,
            document_id=state["document_id"],
            title=kp["title"],
            content=kp["content"],
        ))

    db.commit()

    # 清空检索缓存，下次重建
    if sid in _retriever_caches:
        del _retriever_caches[sid]

    # 为新知识点生成向量 embedding，存入 DB
    try:
        from rag.vector import _get_model
        new_kps = (
            db.query(KnowledgePoint)
            .filter_by(session_id=sid, document_id=state["document_id"])
            .all()
        )
        if new_kps:
            model = _get_model()
            for kp in new_kps:
                if kp.embedding is None:
                    vec = model.encode(kp.content, normalize_embeddings=True,
                                       show_progress_bar=False)
                    kp.embedding = vec.tolist()
            db.commit()
    except Exception:
        db.rollback()
        # embedding 失败不阻塞导入，降级到纯 BM25

    return {"knowledge_points": kps}


# 预生成题目的 TTL：超过此时限且从未使用过的缓存题目作废重出
_PREWARM_TTL = timedelta(hours=24)


def question_gen(state: AgentState) -> AgentState:
    """基于知识点出题（优先使用预缓存题目）

    优化策略：
      1. 先查 questions 表，有缓存的题目直接用（零 LLM 调用）
      2. 没有缓存才调 LLM 生成，并存表供下次复用
      3. ReAct 搜索仅在知识点明显过短时触发（max_turns=1）
    """
    kp = state["current_kp"]
    sid = state.get("session_id", 0)

    # === 优先查缓存：有已保存的题目直接用 ===
    from storage.schemas import Question
    cached = (
        db.query(Question)
        .filter_by(session_id=sid, kp_id=kp["id"])
        .order_by(Question.use_count.asc())
        .first()
    )
    if cached:
        # TTL 检查：预生成但从未被使用的题，超过时限则作废
        if cached.use_count == 0 and cached.created_at and datetime.utcnow() - cached.created_at > _PREWARM_TTL:
            db.delete(cached)
            db.commit()
            print(f"  [缓存] 预生成题目「{cached.title}」已过期，重新生成")
        else:
            cached.use_count += 1
            db.commit()
            print(f"  [缓存] 复用题目「{cached.title}」（已用 {cached.use_count-1} 次）")
            return {"current_question": cached.question_text}

    # === 无缓存 → LLM 生成 ===
    retriever = _get_retriever(sid)
    related = retriever.search(kp["content"], top_k=3)
    related_context = ""
    for _, doc, score in related:
        if doc != kp["content"] and score > 0.3:
            related_context += f"\n- {doc[:200]}"
    context_section = ""
    if related_context:
        context_section = f"\n相关知识（供参考出题）：{related_context}\n"

    memory = _build_memory_context(sid, kp["id"])

    prompt = f"""你是一个出题专家。请基于以下知识点出一道简答题。{context_section}{memory}

要求：
1. 出的题目要有 title（题目标题）和 question（题目内容）
2. 只出一道，不要多
3. 返回 JSON 格式
4. 如果存在历史答题记录，避免出完全一样的题，针对用户的历史薄弱点出题

示例输出：
{{"title": "GMP 模型", "question": "请简述 Go 语言 GMP 模型中 G、M、P 的作用"}}

知识点名称：{kp["title"]}
知识点内容：{kp["content"]}"""

    system = "你只输出 JSON，不要输出其他内容。"
    question_data = _react_call(prompt, system, max_turns=2)

    # === 存入缓存 ===
    try:
        db.add(Question(
            session_id=sid,
            kp_id=kp["id"],
            title=question_data.get("title", kp["title"]),
            question_text=question_data["question"],
            use_count=0,
        ))
        db.commit()
    except Exception:
        db.rollback()

    return {"current_question": question_data["question"]}


def wait_input(state: AgentState) -> AgentState:
    """等待用户输入 — 图在这里暂停"""
    answer = interrupt("请输入你的回答")
    if answer == "__exit__":
        return {"exit_review": True}
    return {"user_answer": answer}


def judge(state: AgentState) -> AgentState:
    """评判用户回答，并存入数据库（含历史 Memory）"""
    if state.get("exit_review"):
        return {"evaluation": {}}

    # Memory：用户针对这个知识点的历史记录
    memory = _build_memory_context(state.get("session_id", 0), state["current_kp"]["id"])

    prompt = f"""你是一个专业的面试官。请根据标准答案评判用户的回答。{memory}

要求：
1. 给出 0-100 的分数
2. 写出总体评语（如有历史记录，对比历史指出进步或不足）
3. 列出优点和不足
4. 列出用户没答出来的知识点（missing_kps），用于后续复习

返回 JSON 格式，示例：
{{
    "score": 72,
    "comment": "基本正确，但对 P 的作用理解不够深入",
    "strengths": ["正确说明了 G 是 goroutine"],
    "weaknesses": ["没有提到工作窃取机制"],
    "missing_kps": ["GMP 中 P 的本地运行队列", "work stealing"]
}}

标准答案：{state["current_kp"]["content"]}

题目：{state["current_question"]}

用户回答：{state["user_answer"]}"""

    response = get_llm().invoke([
        SystemMessage(content="你只输出 JSON，不要输出其他内容。"),
        HumanMessage(content=prompt),
    ])

    parser = JsonOutputParser()
    evaluation = parser.parse(response.content)

    db.add(ReviewRecord(
        session_id=state.get("session_id", 0),
        kp_id=state["current_kp"]["id"],
        question=state["current_question"],
        user_answer=state["user_answer"],
        ai_score=evaluation["score"],
        ai_comment=evaluation.get("comment", ""),
        ai_strengths=evaluation.get("strengths", []),
        ai_weaknesses=evaluation.get("weaknesses", []),
        ai_missing_kps=evaluation.get("missing_kps", []),
    ))
    db.commit()

    return {"evaluation": evaluation}
