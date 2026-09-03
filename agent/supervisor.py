"""主 Agent — Supervisor

把「自然语言输入 → 意图 → 分发到子 Agent」做成一个对话门面。

对应 V1.1 的 service/chat_service.py；V1.2 把它正式收进 agent/ 层，
作为"主 Agent 调度多个子 Agent"的协调者：

    Supervisor.chat(session_id, message)
        → 意图识别（nlu/，V1.3 将升级为 LLM 兜底）
        → 按意图分发给 复习 / 问答 / 导入 / 分析 四个子 Agent
        → 返回结构化结果给前端渲染

当前实现仍是"规则分发 + if 分发"（supervisor 的功能等同体），
真正"一张 LangGraph 主图调度多个子图"（interrupt 上移）在后续 V2 落地。

每个 session 维护一个 Conversation：聊天历史 + 当前活跃的复习 thread。
"""

from nlu.intent import classify_intent, extract_import_content
from service import stats_service
from agent.import_agent import import_content
from agent.qa_agent import answer_question
from agent.review_agent import review_agent


class Supervisor:
    """主 Agent：识别意图并把任务交给对应子 Agent"""

    def __init__(self):
        # session_id -> Conversation
        self._convs: dict[int, dict] = {}

    # ---------- 会话状态 ----------

    def _get_conv(self, session_id: int) -> dict:
        if session_id not in self._convs:
            self._convs[session_id] = {
                "session_id": session_id,
                "messages": [],
                "review_thread_id": None,
                "pending_question": False,  # 是否有一道待回答的题
            }
        return self._convs[session_id]

    # ---------- 入口 ----------

    def chat(self, session_id: int, message: str) -> dict:
        """处理一条用户消息，返回结构化结果。"""
        conv = self._get_conv(session_id)
        conv["messages"].append({"role": "user", "text": message})

        intent = classify_intent(message, review_active=conv["pending_question"])
        result = self._dispatch(conv, intent, message)

        # 追加一条 assistant 显示文本
        conv["messages"].append({"role": "assistant", "text": _display_text(result)})
        return result

    # ---------- 意图分发（路由到子 Agent） ----------

    def _dispatch(self, conv: dict, intent: str, message: str) -> dict:
        sid = conv["session_id"]

        # --- 复习子 Agent：退出 ---
        if intent == "exit_review":
            if conv["review_thread_id"]:
                review_agent.exit(conv["review_thread_id"])
                conv["review_thread_id"] = None
                conv["pending_question"] = False
                return {"type": "chat", "text": "好，复习结束了。想继续、提问、看薄弱分析都可以随时说。"}
            return {"type": "chat", "text": "现在没有进行中的复习哟。说「开始复习」或「考我」就能练起来。"}

        # --- 分析子 Agent ---
        if intent == "analyze":
            data = stats_service.analyze(sid, llm_report=True)
            if "error" in data:
                return {"type": "chat", "text": data["error"]}
            return {"type": "analysis", "data": data}

        # --- 导入子 Agent ---
        if intent == "import":
            content = extract_import_content(message)
            if not content:
                return {"type": "chat", "text": "把内容发我就行，比如「导入：xxx」。内容太短的话直接去「导入」页面更顺手。"}
            res = import_content(sid, content, "对话导入")
            n = len(res.get("knowledge_points", []))
            return {"type": "imported", "data": {"count": n}}

        # --- 复习子 Agent：开始 ---
        if intent == "start_review":
            try:
                thread_id, question = self._start_review(sid)
            except ValueError as e:
                return {"type": "chat", "text": str(e)}
            conv["review_thread_id"] = thread_id
            conv["pending_question"] = True
            return {"type": "question", "data": question}

        # --- 复习子 Agent：下一题提示 ---
        if intent == "next":
            if not conv["review_thread_id"]:
                return {"type": "chat", "text": "还没开始复习呢，说「开始复习」我就出题。"}
            # V1 的复习循环是「答完自动出下一题」；这里提示用户作答或退出
            return {"type": "chat", "text": "当前这道题答完，我会接着出下一题。你也可以直接回答，或说「退出」结束复习。"}

        # --- 问答子 Agent ---
        if intent == "qa":
            text, has_context = answer_question(sid, message)
            return {"type": "answer", "text": text, "has_context": has_context}

        # --- 复习子 Agent：作答 ---
        if intent == "answer" and conv["review_thread_id"]:
            return self._submit_answer(conv, message)

        # --- 兜底引导 ---
        return {
            "type": "chat",
            "text": "我可以：说「开始复习」练一道题、直接问我知识点（比如「什么是 GMP」）、说「我哪里薄弱」做分析，或者「导入：内容」收资料。",
        }

    # ---------- 复习编排 ----------

    def _start_review(self, sid: int):
        """调复习子 Agent 开始一次复习，返回 (thread_id, question 卡片)"""
        data = review_agent.start(sid)
        # 复习线程只在一个会话里唯一活跃即可，若已有则先退出
        return data["thread_id"], {
            "question": data.get("question", ""),
            "kp_title": data.get("kp_title", ""),
            "kp_content": data.get("kp_content", ""),
            "review_reason": data.get("review_reason", ""),
        }

    def _submit_answer(self, conv: dict, answer: str) -> dict:
        thread = conv["review_thread_id"]
        res = review_agent.submit_answer(thread, answer)

        if res.get("exit"):
            conv["review_thread_id"] = None
            conv["pending_question"] = False
            return {"type": "review_result", "evaluation": res.get("evaluation", {}), "exit": True, "next": None}

        # 没退出 → 下一题已在 checkpointer 里
        try:
            nxt = review_agent.get_next(thread)
            next_card = {
                "question": nxt.get("question", ""),
                "kp_title": nxt.get("kp_title", ""),
                "kp_content": nxt.get("kp_content", ""),
                "review_reason": nxt.get("review_reason", ""),
            }
        except Exception:
            next_card = None
            conv["review_thread_id"] = None
            conv["pending_question"] = False

        conv["pending_question"] = next_card is not None
        return {"type": "review_result", "evaluation": res.get("evaluation", {}), "exit": False, "next": next_card}


def _display_text(result: dict) -> str:
    """给对话历史用的纯文本（前端主要靠结构化结果渲染）。"""
    t = result.get("type")
    if t == "chat":
        return result.get("text", "")
    if t == "question":
        return "开始复习：" + (result.get("data", {}).get("question") or "")
    if t == "review_result":
        ev = result.get("evaluation", {})
        return f"评分 {ev.get('score')}：{ev.get('comment','')}"
    if t == "answer":
        return result.get("text", "")
    if t == "analysis":
        return "你的薄弱分析已生成，见下方卡片。"
    if t == "imported":
        return f"已导入 {result.get('data',{}).get('count',0)} 个知识点。"
    return ""


# 全局单例（同 review_agent 的理由：要维护跨请求的 Conversation 状态）
supervisor = Supervisor()
