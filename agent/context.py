"""会话上下文 — 一个 session 的「状态盒子 + token 预算压缩」

V1.3 之前这些状态散在 supervisor._convs 的裸 dict 里，靠 conv["key"] 魔改。
这里收成类型化对象（有名字、有类型、有方法），Supervisor 和 build_tools 不再
随手塞 key，改走属性 / 方法——写错字段会立刻报错，而不是静默出 bug。

压缩策略演进：
  V1.5  按**消息条数**（超过 40 条就折叠最老的，只留最近 8 条）
  V1.6  按 **token 预算**（本文件）

为什么换成 token：条数根本不反映成本——40 条可能是 40 句「好的」（几百 token），
也可能是 40 段粘贴的资料（几万 token），只能再拿一个 [-8:] 兜一道补丁。
token 才是真正的成本单位，预算可算、可解释、可复现，于是那一层补丁可以删掉。

摘要策略：**LLM 摘要优先 + 确定性折叠兜底**（V1.6）。压缩跑在聊天主流程里，
LLM 一失败就是整个请求崩，所以必须有兜底。同时摘要是**滚动**的——旧摘要会和
新折叠的对话一起交给 LLM 合并，不再被下一次压缩覆盖丢掉。

字段：
    messages          对话历史 [{role, text}]（LLM 友好，供展示 + 回填模型）
    review_thread_id  当前活跃复习的 LangGraph thread id
    pending_question  是否有一道待回答的题
    current_card      当前待回答题目的卡片 {question, kp_title, ...}
    summary           前情提要正文（不含包装，供 prompt 使用；格式化见 summary_note）
"""

from dataclasses import dataclass, field

# ========================================
# token 预算（V1.6）
# ========================================
# 分词器用 tiktoken cl100k_base：实测中文 ≈1 token/字，英文 ≈4 字符/token。
# 它并非 DeepSeek 官方分词器，数中文偏保守（估得偏大）——对「预算护栏」而言
# 偏大是安全方向。真要精确，只改 _count_tokens 一个函数。
MAX_MSG_TOKENS = 1000         # 单条消息上限：防一次粘贴巨长文本撑爆预算
HISTORY_TOKEN_BUDGET = 8000   # 历史总预算：超过即触发压缩
KEEP_RECENT_TOKENS = 4000     # 压缩时保留的最近原文预算
SUMMARY_TOKEN_BUDGET = 1000   # 摘要自身预算

# 不变量——破坏任何一条都会出真 bug：
#   1) KEEP_RECENT + SUMMARY < HISTORY：压完（≤5000）必低于触发线（8000），
#      否则「压完立刻再压」，摘要被反复覆盖，越压越丢。
#   2) MAX_MSG <= KEEP_RECENT：保证压缩时至少留得下最后一条，否则保留段可能为空。
assert KEEP_RECENT_TOKENS + SUMMARY_TOKEN_BUDGET < HISTORY_TOKEN_BUDGET
assert MAX_MSG_TOKENS <= KEEP_RECENT_TOKENS

_enc = None


def _get_enc():
    """惰性加载分词器——不在 import 时拖慢启动（首启还要下 BPE 文件）。"""
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _count_tokens(text: str) -> int:
    """单点 token 计数——将来换精确分词器只改这里。"""
    return len(_get_enc().encode(text or ""))


def _truncate_to_tokens(text: str, limit: int) -> str:
    """按 token 边界截断（不切碎字符）。"""
    if not text:
        return text
    enc = _get_enc()
    ids = enc.encode(text)
    if len(ids) <= limit:
        return text
    return enc.decode(ids[:limit])


# ========================================
# 摘要：LLM 优先 + 确定性兜底
# ========================================

_SUMMARY_MAX_CHARS = 200  # 给 LLM 的目标字数（硬上限仍是 SUMMARY_TOKEN_BUDGET）


def _summarize(dropped: list, prev_summary: str) -> str:
    """把一段历史折叠成摘要：LLM 优先，失败回退确定性折叠。"""
    return _summarize_llm(dropped, prev_summary) or _summarize_fallback(dropped, prev_summary)


def _summarize_llm(dropped: list, prev_summary: str) -> str | None:
    """让 LLM 把「旧摘要 + 新折叠的对话」合并成一段新摘要。失败返回 None。

    注意 prev_summary 也一并喂进去——这是「滚动摘要」的关键：
    旧摘要的内容会被合并进新摘要，而不是被覆盖丢掉。
    """
    lines = []
    for m in dropped:
        text = (m.get("text") or "").strip()
        if text:
            lines.append(f"{'用户' if m.get('role') == 'user' else '助手'}：{text}")
    if not lines:
        return None

    prev = f"已有的旧摘要（请一并合并进去）：{prev_summary}\n" if prev_summary else ""
    prompt = f"""下面是学习助手与用户的一段对话。请把它压缩成一段不超过 {_SUMMARY_MAX_CHARS} 字的「前情提要」，供后续对话当上下文用。

要求：
1. 只保留对后续交流有用的信息：用户在学什么、薄弱在哪、聊过哪些话题、有没有未完成的事
2. 不要逐条复述，要提炼总结
3. 不要编造对话里没有的内容
4. 直接输出摘要正文，不要任何前缀、标题或解释

{prev}对话：
{chr(10).join(lines)}"""

    try:
        from core.llm import get_llm          # 惰性 import：避免模块级耦合
        resp = get_llm().invoke(prompt)
        content = getattr(resp, "content", "")
        return (content.strip() or None) if isinstance(content, str) else None
    except Exception as e:  # noqa: BLE001 —— 压缩在聊天主流程里，绝不能把请求带崩
        print(f"  [Context] LLM 摘要失败({e.__class__.__name__})，回退确定性折叠")
        return None


def _summarize_fallback(dropped: list, prev_summary: str) -> str:
    """确定性兜底：取用户说过的话的要点拼接，不调 LLM。"""
    frags = [
        (m.get("text") or "").strip()[:60]
        for m in dropped
        if m.get("role") == "user" and (m.get("text") or "").strip()
    ]
    gist = "，".join(frags[-5:])              # 只取最近 5 条，防摘要自己膨胀
    return "；".join(p for p in (prev_summary, gist) if p)


# ========================================
# SessionContext
# ========================================

@dataclass
class SessionContext:
    session_id: int
    messages: list = field(default_factory=list)   # [{role, text}] LLM 友好历史
    review_thread_id: str | None = None            # 当前活跃复习的 LangGraph thread
    pending_question: bool = False                 # 是否有一道待回答的题
    current_card: dict | None = None               # 当前待回答题目的卡片
    summary: str = ""                              # 前情提要正文（不含包装）

    # ---------- 对话历史 ----------

    def remember(self, role: str, text: str):
        """记一条对话文本（历史只存文本，卡片数据不进历史）。

        单条先按 MAX_MSG_TOKENS 封顶，累计超过 HISTORY_TOKEN_BUDGET 时自动压缩。
        """
        if not text:
            return
        self.messages.append({
            "role": role,
            "text": _truncate_to_tokens(text, MAX_MSG_TOKENS),
        })
        if self._history_tokens() > HISTORY_TOKEN_BUDGET:
            self.compress()

    def _history_tokens(self) -> int:
        """当前历史占用的 token 数（不含摘要——摘要另有 SUMMARY_TOKEN_BUDGET 管）。"""
        return sum(_count_tokens(m.get("text") or "") for m in self.messages)

    def compress(self) -> str:
        """按 token 预算压缩：保留最近 KEEP_RECENT_TOKENS 的原文，更早的折叠进摘要。

        Returns:
            更新后的摘要正文（同时写入 self.summary）。
        """
        keep_from = self._keep_boundary()
        if keep_from <= 0:
            return self.summary               # 没有要折叠的，保持原样

        dropped = self.messages[:keep_from]
        self.messages = self.messages[keep_from:]
        self.summary = _truncate_to_tokens(
            _summarize(dropped, self.summary), SUMMARY_TOKEN_BUDGET
        )
        return self.summary

    def _keep_boundary(self) -> int:
        """从尾部往前攒，返回「保留段」的起始下标；0 表示无需折叠。

        至少保留最后一条原文——这由 MAX_MSG_TOKENS <= KEEP_RECENT_TOKENS 保证。
        """
        last = len(self.messages) - 1
        acc = 0
        for i in range(last, -1, -1):
            t = _count_tokens(self.messages[i].get("text") or "")
            if acc + t > KEEP_RECENT_TOKENS and i < last:
                return i + 1
            acc += t
        return 0

    @property
    def summary_note(self) -> str:
        """喂给 System Prompt 的前情提要前缀；无摘要时返回空串。"""
        return f"（前情提要：{self.summary}）" if self.summary else ""

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
