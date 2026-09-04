"""主 Agent — Supervisor（V1.3：LLM 大脑）

从 V1.1 的「规则路由器」升级为「LLM 工具调用循环」：

    Supervisor.chat(session_id, message)
        → 确定性安全快路（退出/下一题/导入前缀——避免误判、不花 LLM）
        → 拼 [System 提示 + 最近对话历史 + 当前消息]
        → run_agent() 跑工具循环：LLM 自主决定调 4 个子 Agent 的能力 / 通用搜索 / 直接答
        → 按 tool_log 里最后一张卡片决定返回 type；text 前言来自 LLM 的最终话
        → 把自然语言回复记进对话历史，供下一轮喂回模型

对话历史真正喂给了 LLM（V1.1 只记展示文本、从未给模型）——
这是「能跟他聊天、像通用 Agent」的关键。

每个 session 维护一个 Conversation：LLM 友好历史 + 当前活跃的复习 thread + 当前题目卡片。
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.tools import build_tools
from core.llm import run_agent
from nlu.intent import extract_import_content, fast_path_intent

# 喂给 LLM 的最近历史条数上限（防上下文膨胀）
_HISTORY_MAX = 8


class Supervisor:
    """主 Agent：LLM 大脑 + 工具循环，调度 4 个子 Agent 的能力"""

    def __init__(self):
        # session_id -> Conversation
        self._convs: dict[int, dict] = {}

    # ---------- 会话状态 ----------

    def _get_conv(self, session_id: int) -> dict:
        if session_id not in self._convs:
            self._convs[session_id] = {
                "session_id": session_id,
                "messages": [],            # [{role, text}] LLM 友好历史（供展示 + 回填模型）
                "review_thread_id": None,  # 当前活跃复习的 LangGraph thread
                "pending_question": False,  # 是否有一道待回答的题
                "current_card": None,       # 当前待回答题目的卡片 {question, kp_title, ...}
            }
        return self._convs[session_id]

    # ---------- 入口 ----------

    def chat(self, session_id: int, message: str) -> dict:
        """处理一条用户消息，返回结构化结果（前端按 type 渲染）。"""
        conv = self._get_conv(session_id)

        # 1) 确定性安全快路：明确命令不经过 LLM，防止误判成"作答/提问"。
        #    薄薄一层，命中即返回；其余全部交给 LLM 循环（见 build_tools 的 7 个工具）。
        fast = fast_path_intent(message)
        if fast is not None:
            return self._fast_path(conv, fast, message)

        # 2) 常规路径：LLM 工具调用循环
        conv["messages"].append({"role": "user", "text": message})
        messages = [SystemMessage(content=self._system_prompt(conv))]
        messages += self._to_history(conv["messages"][-_HISTORY_MAX:])
        messages.append(HumanMessage(content=message))

        final_text, tool_log = run_agent(
            messages,
            build_tools(session_id, conv),
        )

        # 3) 由 tool_log 最后一张结构化卡片决定返回 type
        result = self._map_result(final_text, tool_log)

        # 4) 记录一条 assistant 自然语言回复（不落整张卡片，避免历史被撑爆）
        self._remember(conv, self._assistant_text(result))
        return result

    # ---------- 确定性安全快路 ----------

    def _fast_path(self, conv: dict, intent: str, message: str) -> dict:
        """退出 / 下一题 / 导入 —— 复用 build_tools 里的同名工具，保持行为单一来源。"""
        conv["messages"].append({"role": "user", "text": message})
        tools = {t.name: t for t in build_tools(conv["session_id"], conv)}

        if intent == "exit_review":
            result_str = tools["exit_review"].invoke({})
        elif intent == "next":
            if conv.get("review_thread_id"):
                text = "答完当前这道题，我会自动接下一题。你也可以直接输入答案，或说「退出」结束复习。"
            else:
                text = "还没开始复习呢。说「开始复习」我就出题。"
            self._remember(conv, text)
            return {"type": "chat", "text": text}
        else:  # import —— 前缀确定但内容要再校验一次
            content = extract_import_content(message)
            if not content:
                text = "把内容发我就行，比如「导入：xxx」。内容太短的话，直接去「导入」页粘贴更顺手。"
                self._remember(conv, text)
                return {"type": "chat", "text": text}
            result_str = tools["import_content"].invoke({"content": content})

        result = self._card_to_result(json.loads(result_str), final_text="")
        self._remember(conv, self._assistant_text(result))
        return result

    # ---------- 结果映射 ----------

    def _map_result(self, final_text: str, tool_log: list) -> dict:
        """从 tool_log 里取最后一张结构化卡片，映射成 ChatResult。"""
        last = None
        for _name, _args, result_str in tool_log:
            try:
                payload = json.loads(result_str)
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict) and payload.get("card"):
                last = payload  # 覆盖式：保留最后一张卡（LLM 多轮后以最终为准）
        if last is None:
            return {"type": "chat", "text": final_text}
        return self._card_to_result(last, final_text)

    @staticmethod
    def _card_to_result(card: dict, final_text: str) -> dict:
        """把工具返回的卡片 JSON 还原成前端可渲染的 ChatResult。

        卡片类型（question / review_result / analysis / imported）额外带
        text=LLM 的最终话作为前言气泡；answer 直接用工具里的回答文本。
        """
        if not card.get("ok", True):
            return {"type": "chat", "text": card.get("text") or card.get("error") or "出错了"}
        kind = card.get("card", "chat")

        if kind == "question":
            return {"type": "question", "text": final_text, "data": card.get("data", {})}
        if kind == "review_result":
            d = card.get("data", {})
            return {
                "type": "review_result",
                "text": final_text,
                "evaluation": d.get("evaluation", {}),
                "exit": bool(d.get("exit")),
                "next": d.get("next"),
            }
        if kind == "analysis":
            return {"type": "analysis", "text": final_text, "data": card.get("data", {})}
        if kind == "imported":
            return {"type": "imported", "text": final_text, "data": card.get("data", {})}
        if kind == "answer":
            return {
                "type": "answer",
                "text": card.get("text", final_text),
                "has_context": card.get("has_context", True),
            }
        # chat 卡（exit / 引导 / 错误提示）
        return {"type": "chat", "text": card.get("text", final_text)}

    # ---------- 系统提示 ----------

    @staticmethod
    def _system_prompt(conv: dict) -> str:
        reviewing = conv.get("review_thread_id") is not None
        card = conv.get("current_card") or {}
        pending = (
            f"知识点「{card.get('kp_title', '')}」出的题：{card.get('question', '')}"
            if reviewing and card.get("question")
            else "无"
        )
        return f"""你是 StudyForge 的 AI 学习教练，陪用户复习 / 答疑 / 做薄弱分析。

你只能用下面 7 个工具获取能力。**用户的资料、复习进度、答题记录等状态，只能靠调用工具拿到，绝不凭空猜测、绝不编造。** 看到明确的动作请求时，先调用对应工具，再按工具返回的内容向用户总结：

- start_review：开始复习出题 —— 用户说「开始复习 / 考我 / 出题 / 测测我 / 练练」时调用
- submit_answer：把用户的回答提交判分 —— 复习中有「待回答题目」、用户是在答题时调用
- exit_review：结束当前复习 —— 用户说「退出 / 结束 / 算了 / 停」时调用
- answer_question：回答用户资料里的知识点问题 —— 「什么是 X / 讲讲 Y / A和B 区别」等
- analyze_weakness：生成薄弱分析 —— 「我哪里薄弱 / 分析一下 / 我的短板 / 当前水平」时调用
- import_content：导入资料成知识点 —— 用户发「导入：xxx」或贴一段想收录的文字时调用
- general_search：搜互联网 —— 问的是资料之外 / 需要较新信息的通用问题时调用

原则：
1. **动作先于说话**：出题、判分、分析、导入这些必须通过调工具完成；工具返回后你总结给用户，不要自己假装出了题或判了分。真正拿不准用户想干嘛时才纯文字闲聊并顺势引导。
2. 复习语境消歧：复习中系统会给你「待回答题目」。用户消息若是在答这道题 → submit_answer；若是新提问 → answer_question；若是命令（分析/退出/再来一轮/导入）→ 对应工具；答完题后一般接着出下一题或问是否继续。
3. 可以从话里识别学习主题（如「复习数据库索引」里的「数据库索引」），在回复中点出来让他感到被理解；目前出题不会真的按主题过滤。
4. 一次用户消息通常只做一个主要动作；最后都给一句自然的中文收尾。

【当前复习状态】
进行中：{'是' if reviewing else '否'}
待回答题目：{pending}"""

    # ---------- 历史辅助 ----------

    @staticmethod
    def _to_history(msgs: list) -> list:
        out = []
        for m in msgs:
            text = (m.get("text") or "").strip()
            if not text:
                continue
            out.append(
                HumanMessage(content=text)
                if m.get("role") == "user"
                else AIMessage(content=text)
            )
        return out

    @staticmethod
    def _remember(conv: dict, text: str):
        """记一条 assistant 自然语言回复（历史只存文本，卡片数据不进历史）。"""
        if text:
            conv["messages"].append({"role": "assistant", "text": text})

    @staticmethod
    def _assistant_text(result: dict) -> str:
        """给对话历史/记忆用的纯文本（前端主要靠结构化结果渲染卡片）。"""
        t = result.get("type")
        if t == "chat":
            return result.get("text", "")
        if t == "answer":
            return result.get("text", "")
        if t == "question":
            return result.get("text") or ("出题：" + (result.get("data", {}).get("question") or ""))
        if t == "review_result":
            return result.get("text") or (
                f"评分 {result.get('evaluation', {}).get('score')}："
                f"{result.get('evaluation', {}).get('comment', '')}"
            )
        if t == "analysis":
            return result.get("text") or "你的薄弱分析已生成，见下方卡片。"
        if t == "imported":
            return result.get("text") or f"已导入 {result.get('data', {}).get('count', 0)} 个知识点。"
        return ""


# 全局单例（要维护跨请求的 Conversation 状态）
supervisor = Supervisor()
