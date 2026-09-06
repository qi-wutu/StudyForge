"""会话上下文 — 一个 session 的「状态盒子 + 简单压缩」

V1.3 之前这些状态散在 supervisor._convs 的裸 dict 里，靠 conv["key"] 魔改。
这里收成类型化对象（有名字、有类型、有方法），Supervisor 和 build_tools 不再
随手塞 key，改走属性 / 方法——写错字段会立刻报错，而不是静默出 bug。

V1.5 加了**简单压缩**：对话历史不无限膨胀——超过阈值后，把最老的一批
折叠进一个「前情提要」摘要串，只留最近的原文。长续聊也能保留大概脉络，
而不是把 Oldest 硬截断掉（V1.3 只做 [-8:] 截断，等于把更早的对话直接丢了）。

行为与原 V1.3 等价，只是把「乱兜」换成「分格子」+ 不漏掉早期对话。

字段：
    messages          对话历史 [{role, text}]（LLM 友好，供展示 + 回填模型）
    review_thread_id  当前活跃复习的 LangGraph thread id
    pending_question  是否有一道待回答的题
    current_card      当前待回答题目的卡片 {question, kp_title, ...}
    summary           前情提要：被折叠的早期对话的压缩摘要
"""

from dataclasses import dataclass, field

# 原始历史最大条数：超过就把最老的折叠进 summary，避免无限膨胀
_MAX_STORED = 40
# feed 给 LLM 时保留的最近条数（Supervisor 拼历史用）
HISTORY_MAX = 8


@dataclass
class SessionContext:
    session_id: int
    messages: list = field(default_factory=list)   # [{role, text}] LLM 友好历史
    review_thread_id: str | None = None            # 当前活跃复习的 LangGraph thread
    pending_question: bool = False                 # 是否有一道待回答的题
    current_card: dict | None = None               # 当前待回答题目的卡片
    summary: str = ""                              # 前情提要（被折叠的早期对话摘要）

    # ---------- 对话历史 ----------

    def remember(self, role: str, text: str):
        """记一条对话文本（历史只存文本，卡片数据不进历史）。

        超出 _MAX_STORED 时自动把最老的折叠进 summary，并保留最近 _MAX_STORED 条。
        """
        if text:
            self.messages.append({"role": role, "text": text})
            if len(self.messages) > _MAX_STORED:
                self.compress()

    def compress(self, keep_recent: int = HISTORY_MAX, max_chars: int = 500) -> str:
        """简单压缩：只保留最近 keep_recent 条原文，把更早的折叠成一句摘要。

        摘要方式：取被折叠对话里「用户说过的」要点，凑成「前情提要」，拼到 self.summary。
        纯确定性、无 LLM 调用（重对话热路径不花钱）；未来可换成 LLM 摘要。

        Returns:
            新生成的前情提要串（同时写入 self.summary）。
        """
        if len(self.messages) <= keep_recent:
            return self.summary

        dropped = self.messages[:-keep_recent]
        self.messages = self.messages[-keep_recent:]

        frags = []
        for m in dropped:
            t = (m.get("text") or "").strip()
            if m.get("role") == "user" and t:
                frags.append(t[:60])
        gist = "，".join(frags[-5:])           # 只取最近 5 条用户输入，防摘要也膨胀
        self.summary = "（前情提要：%s）" % (gist[:max_chars])

        return self.summary

    @property
    def summary_note(self) -> str:
        """喂给 System Prompt 的前情提要前缀；无摘要时返回空串。"""
        return self.summary

    # ---------- 复习状态 ----------

    @property
    def reviewing(self) -> bool:
        """当前是否在复习中（有活跃 thread）。"""
        return self.review_thread_id is not None

    def set_review(self, thread_id: str | None, card: dict | None):
        """进入/更新复习态：设定当前 thread 与待回答题目。"""
        self.review_thread_id = thread_id
        self.pending_question = card is not None
        self.current_card = card

    def clear_review(self):
        """退出复习：清空 thread / 待答题 / 题目卡。"""
        self.review_thread_id = None
        self.pending_question = False
        self.current_card = None
