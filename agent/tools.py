"""Agent 工具集 — Supervisor 的「LLM 可调用能力」

V1.3：把 4 个子 Agent + 统计服务的能力包成 LangChain @tool，
Supervisor 把 build_tools() 的结果交给 core.llm.run_agent()，
由 LLM 自主决定调哪个工具、多轮往复 —— 这就是「通用 Agent」的大脑。

每个 Agent 工具返回 JSON 字符串（LLM 好读、结构可复用），
Supervisor 事后 json.loads 还原成渲染用的卡片。
tools 需要「当前会话 + 当前复习线程」上下文，所以用工厂函数按会话构造：
LLM 全程不接触内部 thread_id，由闭包从 conv 里读写，状态永远和对话同步。
"""

import json

from langchain_core.tools import tool

from agent.import_agent import import_content as _do_import
from agent.qa_agent import answer_question as _answer_question
from agent.review_agent import review_agent
from service import stats_service


def build_tools(session_id: int, conv: dict) -> list:
    """为一次对话构造该会话可用的工具集（6 个 Agent 工具 + 通用搜索）。

    Args:
        session_id: 当前会话 ID（工具闭包持有，不暴露给 LLM）
        conv:       Supervisor 持有的 Conversation：
                    {session_id, messages, review_thread_id, pending_question, current_card}
                    工具的闭包读写它，保证复习线程状态始终与对话同步。

    Returns:
        可直接 bind_tools / run_agent 的 LangChain Tool 列表
    """

    # ---------- 复习子 Agent ----------

    @tool
    def start_review() -> str:
        """开始一轮新的复习：AI 从当前会话已导入的知识点里出一题。用户说「开始复习 / 考我 / 出题 / 测测我」时调用。前提是该会话已导入资料并提取出知识点。"""
        try:
            data = review_agent.start(session_id)
        except ValueError as e:
            return json.dumps({"ok": False, "card": "chat", "text": str(e)}, ensure_ascii=False)
        card = {
            "question": data.get("question", ""),
            "kp_title": data.get("kp_title", ""),
            "kp_content": data.get("kp_content", ""),
            "review_reason": data.get("review_reason", ""),
        }
        conv["review_thread_id"] = data["thread_id"]
        conv["pending_question"] = True
        conv["current_card"] = card
        return json.dumps({"ok": True, "card": "question", "data": card}, ensure_ascii=False)

    @tool
    def submit_answer(answer: str) -> str:
        """提交对当前复习题的回答：AI 判分 + 评语，并接出下一题。用户在复习中针对当前这道题作答时调用。"""
        thread = conv.get("review_thread_id")
        if not thread:
            return json.dumps({"ok": False, "card": "chat", "text": "现在没有进行中的复习。说「开始复习」先出一题再作答。"}, ensure_ascii=False)
        try:
            res = review_agent.submit_answer(thread, answer)
        except Exception as e:  # noqa: BLE001 —— thread 可能已过期
            conv["review_thread_id"] = None
            conv["pending_question"] = False
            conv["current_card"] = None
            return json.dumps({"ok": False, "card": "chat", "text": f"这次复习好像失效了（{e}），说「开始复习」重开一轮吧。"}, ensure_ascii=False)

        if res.get("exit"):
            conv["review_thread_id"] = None
            conv["pending_question"] = False
            conv["current_card"] = None
            return json.dumps({
                "ok": True, "card": "review_result",
                "data": {"evaluation": res.get("evaluation", {}), "exit": True, "next": None},
            }, ensure_ascii=False)

        # 没退出 → 下一题已就绪在 checkpointer 里，读出来接上
        try:
            nxt = review_agent.get_next(thread)
            next_card = {
                "question": nxt.get("question", ""),
                "kp_title": nxt.get("kp_title", ""),
                "kp_content": nxt.get("kp_content", ""),
                "review_reason": nxt.get("review_reason", ""),
            }
        except Exception:  # noqa: BLE001
            next_card = None
            conv["review_thread_id"] = None
            conv["pending_question"] = False
            conv["current_card"] = None

        conv["pending_question"] = next_card is not None
        conv["current_card"] = next_card
        return json.dumps({
            "ok": True, "card": "review_result",
            "data": {"evaluation": res.get("evaluation", {}), "exit": False, "next": next_card},
        }, ensure_ascii=False)

    @tool
    def exit_review() -> str:
        """主动结束当前复习。用户在复习中想停下来说「退出 / 结束 / 算了」时调用。"""
        thread = conv.get("review_thread_id")
        if not thread:
            return json.dumps({"ok": True, "card": "chat", "text": "现在没有进行中的复习。说「开始复习」随时能练。"}, ensure_ascii=False)
        review_agent.exit(thread)
        conv["review_thread_id"] = None
        conv["pending_question"] = False
        conv["current_card"] = None
        return json.dumps({"ok": True, "card": "chat", "text": "好，复习结束了。想提问、看薄弱分析或再来一轮都可以随时说。"}, ensure_ascii=False)

    # ---------- 问答子 Agent ----------

    @tool
    def answer_question(question: str) -> str:
        """回答关于当前会话学习资料的知识问题（基于已导入的资料检索后作答，不瞎编）。用户问「什么是 X / 讲讲 Y / 怎么理解 Z」等知识点时调用。"""
        text, has_context = _answer_question(session_id, question)
        return json.dumps({"ok": True, "card": "answer", "text": text, "has_context": has_context}, ensure_ascii=False)

    # ---------- 分析子 Agent ----------

    @tool
    def analyze_weakness() -> str:
        """生成薄弱分析：从答题记录统计薄弱知识点排行 + AI 文字报告。用户说「我哪里薄弱 / 分析一下 / 我的短板 / 当前水平」时调用。"""
        data = stats_service.analyze(session_id, llm_report=True)
        if "error" in data:
            return json.dumps({"ok": False, "card": "chat", "text": data["error"]}, ensure_ascii=False)
        return json.dumps({"ok": True, "card": "analysis", "data": data}, ensure_ascii=False)

    # ---------- 导入子 Agent ----------

    @tool
    def import_content(content: str) -> str:
        """导入学习资料：把用户贴的一段内容提取成知识点，供后续复习/提问使用。用户发「导入：xxx」或要求把一段文字收进资料库时调用。"""
        if len(content.strip()) < 20:
            return json.dumps({"ok": False, "card": "chat", "text": "内容太短，我不好提取知识点。粘贴完整的章节或文档片段试试？"}, ensure_ascii=False)
        res = _do_import(session_id, content.strip(), "对话导入")
        n = len(res.get("knowledge_points", []))
        return json.dumps({"ok": True, "card": "imported", "data": {"count": n}}, ensure_ascii=False)

    return [
        start_review,
        submit_answer,
        exit_review,
        answer_question,
        analyze_weakness,
        import_content,
        general_search,
    ]


# ---------- 通用搜索（会话无关，模块级即可） ----------

@tool
def general_search(query: str) -> str:
    """搜索互联网获取最新/通用资料。当用户问的是当前学习资料之外的东西、或需要较新的信息时调用，把结果列表返回。"""
    from tools.tools import search_web  # 惰性 import：避免纯对话也被拖进 ddgs
    return search_web.invoke({"query": query})
