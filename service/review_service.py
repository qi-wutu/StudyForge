"""复习管理 — 驱动 review_graph 的暂停/恢复循环

复习图是一个循环图，但被拆到多个 HTTP 端点里分步调用。
核心机制是 LangGraph 的 interrupt + checkpointer：

  1. start() → 图跑到 wait_input 节点遇到 interrupt
              → 图"冻住"，但 state 已被 MemorySaver 存下
              → 返回题目给用户

  2. submit_answer() → Command(resume=回答) 唤醒图
                     → judge 判分 → question_gen 出下一题
                     → 又跑到 wait_input 冻住
                     → 返回评价 + 下一题

  3. get_next() → 不跑图，光读 checkpointer 缓存的当前 state

  4. exit() → Command(resume="__exit__") 唤醒图
             → wait_input 看到 __exit__，设置 exit_review=True
             → 图正常结束

  thread_id = uuid，每次 start() 生成一个，用来定位 checkpointer 里的状态。
  _active_reviews 只在内存，服务重启后丢失。
"""

import uuid
from threading import Thread

from langgraph.types import Command

from graph.graph import review_graph
from graph.state import AgentState
from service.stats_service import _clear_caches
from storage.db import db
from storage.schemas import KnowledgePoint


class ReviewService:
    """复习服务 — 管理每个 thread 的复习会话"""

    def __init__(self):
        # _active_reviews = {thread_id: {thread_id, session_id}, ...}
        # 只用来跟踪"哪些复习正在进行"，不存重要数据
        self._active_reviews: dict[str, dict] = {}

    def start(self, session_id: int) -> dict:
        """启动一次复习

        流程：
          1. 检查会话是否有知识点
          2. 生成 thread_id（uuid）
          3. 构造初始 AgentState
          4. review_graph.invoke() → 跑到 wait_input 暂停
          5. 返回第一题内容

        Returns:
            {thread_id, question, kp_title, kp_content, review_reason}
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
        result = review_graph.invoke(state, config)

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
        result = review_graph.invoke(Command(resume=answer), config)

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
            state = review_graph.get_state(config)
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

    def exit(self, thread_id: str) -> dict:
        """主动结束复习

        发送 __exit__ 标记，让 wait_input 设置 exit_review=True，
        条件边走 end 分支，图正常结束。
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            review_graph.invoke(Command(resume="__exit__"), config)
        except Exception:
            pass  # thread 可能已经过期了
        self._active_reviews.pop(thread_id, None)
        return {"exit": True}

    def list_active(self) -> list[str]:
        """查看当前正在进行的复习会话（thread_id 列表）"""
        return list(self._active_reviews.keys())

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

    幂等的——已有缓存的跳过。
    异常不抛出——预生成失败不影响正常使用。
    """
    from graph.node import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
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


# 全局单例
# 为什么是单例而不是每次请求新建？
#   因为要维护 _active_reviews 字典，新建就丢了。
review_service = ReviewService()
