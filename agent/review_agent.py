"""复习子 Agent — ReviewAgent

复习 Agent 是一个"有状态的长对话子 Agent"，负责维持一轮复习：
出题 → 等你回答 → 判分 → 选下一题 → 出题 → …

内部实现分三层：
  1. 节点（scheduler / question_gen / wait_input / judge）—— V1 原 graph/node.py 迁入
  2. 复习图（LangGraph，interrupt + MemorySaver）        —— V1 原 graph/graph.py 迁入
  3. 对外 turn API（start / submit_answer / get_next / exit / prewarm）
     —— V1 原 service/review_service.py 迁入

为什么保留 LangGraph + interrupt 而不是改成纯顺序函数？
  - 这张图已经是"跨请求保持复习状态"的现成机制，chat 和 button 两个入口都在用
  - interrupt 让图在 wait_input 冻住，checkpointer 存状态，Command(resume) 唤醒
  - 对调用方（supervisor / Review 页）它仍是可调用的：submit_answer 一把跑完 judge+下一题

状态：
  _active_reviews 只用来跟踪"哪些复习正在进行"，不存重要数据。
  对话历史 / 审题上下文仍在 checkpointer（MemorySaver）里，服务重启即失。
"""

import uuid
from datetime import datetime, timedelta
from threading import Thread

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.state import AgentState
from core.llm import get_llm, react_json
from rag.retriever import get_session_retriever, invalidate_retriever
from service.stats_service import _clear_caches
from storage.db import db
from storage.schemas import KnowledgePoint, ReviewRecord


# ========================================
# 条件边函数
# ========================================

def scheduler_should_continue(state: AgentState) -> str:
    if state.get("exit_review"):
        return "end"
    return "question_gen"


def judge_should_continue(state: AgentState) -> str:
    if state.get("exit_review"):
        return "end"
    return "scheduler"


# ========================================
# 智能调度（双车道）
# ========================================

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
    retriever = get_session_retriever(sid)
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


# ========================================
# Memory — 历史答题上下文
# ========================================

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


# ========================================
# 复习节点
# ========================================

def scheduler(state: AgentState) -> AgentState:
    """从数据库取出下一个待复习知识点（智能调度）"""
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
    retriever = get_session_retriever(sid)
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
    question_data = react_json(prompt, system, max_turns=2)

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


# ========================================
# 复习图（LangGraph，interrupt + checkpointer）
# ========================================
# 流程：
#   scheduler ──(无知识点)→ END
#      │
#      └──(有)─→ question_gen → wait_input（interrupt 冻住）→ judge
#                                                              │
#                                          (退出)→ END  (继续)→ scheduler

def _build_review_graph():
    builder = StateGraph(AgentState)

    builder.add_node("scheduler", scheduler)
    builder.add_node("question_gen", question_gen)
    builder.add_node("wait_input", wait_input)
    builder.add_node("judge", judge)

    builder.set_entry_point("scheduler")

    builder.add_conditional_edges(
        "scheduler",
        scheduler_should_continue,
        {"question_gen": "question_gen", "end": END},
    )
    builder.add_edge("question_gen", "wait_input")
    builder.add_edge("wait_input", "judge")
    builder.add_conditional_edges(
        "judge",
        judge_should_continue,
        {"scheduler": "scheduler", "end": END},
    )

    return builder.compile(checkpointer=MemorySaver())


_review_graph = _build_review_graph()


# ========================================
# ReviewAgent — 对外 turn API（吸收 V1 review_service）
# ========================================

class ReviewAgent:
    """复习子 Agent：负责一次复习的暂停 / 恢复循环。

    为什么是单例而不是每次请求新建？
      要维护 _active_reviews 字典，新建就丢了。

    thread_id = uuid，每次 start() 生成，用来定位 checkpointer 里的状态。
    _active_reviews 只在内存，服务重启后丢失。
    """

    def __init__(self):
        # _active_reviews = {thread_id: {thread_id, session_id}, ...}
        self._active_reviews: dict[str, dict] = {}

    # ---------- 启动 ----------

    def start(self, session_id: int) -> dict:
        """启动一次复习

        1. 检查会话是否有知识点
        2. 生成 thread_id（uuid）
        3. 构造初始 AgentState
        4. review_graph.invoke() → 跑到 wait_input 暂停
        5. 返回第一题内容
        """
        # 没知识点无法复习
        kp_count = db.query(KnowledgePoint).filter_by(session_id=session_id).count()
        if kp_count == 0:
            raise ValueError("当前会话没有知识点，请先导入资料")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        state: AgentState = {
            "messages": [],
            "session_id": session_id,
            "raw_content": "",
            "document_id": 0,
            "knowledge_points": [],
            "current_kp": {},
            "current_question": "",
            "user_answer": "",
            "evaluation": {},
            "kp_index": 0,
            "exit_review": False,
            "review_queue": [],
            "queue_pos": 0,
            "review_reason": "",
        }

        # invoke 会跑到 wait_input 的 interrupt() 才返回
        result = _review_graph.invoke(state, config)

        # 记录活跃复习
        self._active_reviews[thread_id] = {
            "thread_id": thread_id,
            "session_id": session_id,
        }

        return {
            "thread_id": thread_id,
            "question": result.get("current_question", ""),
            "kp_title": result.get("current_kp", {}).get("title", ""),
            "kp_content": result.get("current_kp", {}).get("content", ""),
            "review_reason": result.get("review_reason", ""),
        }

    # ---------- 作答 ----------

    def submit_answer(self, thread_id: str, answer: str) -> dict:
        """提交回答 → AI 判分

        用 Command(resume) 唤醒暂停的图：
          1. resume=answer → wait_input 收到回答，继续执行
          2. judge 节点判分
          3. 条件边决定继续还是结束
          4. 如果是继续，跑到下一个 wait_input 又暂停

        Returns:
            {"evaluation": {...}, "exit": bool}
            exit=true 表示复习已结束（没有下一题了）
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = _review_graph.invoke(Command(resume=answer), config)

        # 有新的答题记录，清除该会话的分析缓存
        session_id = self._active_reviews.get(thread_id, {}).get("session_id", 0)
        if session_id:
            _clear_caches(session_id)

        evaluation = result.get("evaluation", {})
        exit_review = result.get("exit_review", False)

        if exit_review or not result.get("current_question"):
            # 复习结束，清理活跃记录
            self._active_reviews.pop(thread_id, None)
            return {"evaluation": evaluation, "exit": True}

        return {"evaluation": evaluation, "exit": False}

    # ---------- 取下一题 ----------

    def get_next(self, thread_id: str) -> dict:
        """获取当前 thread 的下一题

        关键区别：不调 invoke，不跑图。
        直接从 checkpointer 读取缓存的 state。
        用于重新打开前端时恢复复习状态。

        Raises:
            ValueError: thread 不存在或已过期
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = _review_graph.get_state(config)
        except Exception:
            raise ValueError("复习会话不存在或已过期")

        values = getattr(state, "values", state)
        question = values.get("current_question", "")
        if not question:
            raise ValueError("暂无下一题")

        return {
            "thread_id": thread_id,
            "question": question,
            "kp_title": values.get("current_kp", {}).get("title", ""),
            "kp_content": values.get("current_kp", {}).get("content", ""),
            "review_reason": values.get("review_reason", ""),
        }

    # ---------- 退出 ----------

    def exit(self, thread_id: str) -> dict:
        """主动结束复习

        发送 __exit__ 标记，让 wait_input 设置 exit_review=True，
        条件边走 end 分支，图正常结束。
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            _review_graph.invoke(Command(resume="__exit__"), config)
        except Exception:
            pass  # thread 可能已经过期了
        self._active_reviews.pop(thread_id, None)
        return {"exit": True}

    # ---------- 查询 / 预生成 ----------

    def prewarm(self, session_id: int):
        """预生成题目：为指定 session 中没有缓存的前 2 个知识点生成题目

        在后台线程执行，不阻塞调用方。
        切换 session 后自动触发，让首次复习零等待。
        幂等的——已有缓存的跳过。
        """
        Thread(target=_do_prewarm, args=(session_id,), daemon=True).start()


def _do_prewarm(session_id: int):
    """后台执行：为知识点预生成题目

    遍历该 session 所有知识点，对尚未有缓存题目的前 2 个，
    调 LLM 各出一道题存入 questions 表。
    幂等的——已有缓存的跳过。异常不抛出——预生成失败不影响正常使用。
    """
    from storage.schemas import Question

    try:
        kps = (
            db.query(KnowledgePoint)
            .filter_by(session_id=session_id)
            .order_by(KnowledgePoint.id)
            .all()
        )
        if not kps:
            return

        llm = get_llm()
        count = 0
        for kp in kps[:2]:
            exists = (
                db.query(Question)
                .filter_by(session_id=session_id, kp_id=kp.id)
                .first()
            )
            if exists:
                continue

            prompt = f"""你是一个出题专家。请基于以下知识点出一道简答题。

只输出 JSON，格式如下：
{{"title": "题目标题", "question": "题目内容"}}

知识点名称：{kp.title}
知识点内容：{kp.content}"""

            try:
                response = llm.invoke([
                    SystemMessage(content="你只输出 JSON，不要输出其他内容。"),
                    HumanMessage(content=prompt),
                ])
                import json
                data = json.loads(response.content)
                db.add(Question(
                    session_id=session_id,
                    kp_id=kp.id,
                    title=data.get("title", kp.title),
                    question_text=data["question"],
                    use_count=0,
                ))
                db.commit()
                count += 1
            except Exception:
                db.rollback()

        if count:
            print(f"  [预生成] session={session_id}，缓存了 {count} 道题")
    except Exception:
        pass  # 预生成失败不影响正常使用


# 全局单例：全应用共享一个 ReviewAgent（维护 _active_reviews）
review_agent = ReviewAgent()
