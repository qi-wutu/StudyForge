"""Agent 状态定义

LangGraph 的核心概念之一：State（状态）。
State 是"节点之间传数据的格式"，类似 Go 里的 struct 或前端的 store。

每个节点可以：
  - 读 state：state["xxx"]
  - 写 state：return {"xxx": value}  （LangGraph 自动合并回 state）

注意：
  messages 用了 add_messages 这个 reducer（而不是直接覆盖）。
  这意味着每次往 messages 加新消息，历史消息不会丢。
  但目前项目没有用多轮对话，messages 字段是预留的。
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph 图的全局状态 — 所有节点共享"""

    # 对话历史（预留，目前没用）
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # === 会话隔离 ===
    session_id: int               # 当前会话 ID（节点根据这个过滤数据）

    # === 导入资料时用的字段 ===
    raw_content: str              # 用户导入的原始文件内容（Markdown/文本）
    document_id: int              # 数据库 documents 表的 id（planner 存知识点时关联用）

    # === 知识点 ===
    knowledge_points: list[dict]  # planner 提取的结果：[{"title":"...", "content":"..."}]

    # === 复习流程用的字段 ===
    current_kp: dict              # 当前正在考的知识点：{"id": 1, "title": "...", "content": "..."}
    current_question: str         # 当前题目
    user_answer: str              # 用户刚刚输入的回答
    evaluation: dict              # judge 给的评判结果：{score, comment, strengths, ...}

    # === 循环控制 ===
    kp_index: int                 # 当前复习到第几个知识点（scheduler 用来自增）
    exit_review: bool             # 是否退出复习循环（条件边判断用）

    # === 智能调度（双车道） ===
    review_queue: list            # [{kp_id, reason, lane}, ...] 出题队列
    queue_pos: int                # 当前队列位置
    review_reason: str            # 当前题目的调度原因（展示用）
