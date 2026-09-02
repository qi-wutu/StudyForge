"""资料导入 — 调用 import_graph 提取知识点

    用户粘贴文本或上传文件
               ↓
    import_content(session_id, content, title)
               ↓
    构造 AgentState → import_graph.invoke(state)
               ↓
    planner 节点（LLM 提取知识点）
               ↓
    知识写入 MySQL + 生成向量 embedding
               ↓
    返回提取结果给前端

整个流程是同步的——导入完才返回。
"""

from graph.graph import import_graph
from graph.state import AgentState
from storage.db import db
from storage.schemas import Document


def import_content(session_id: int, content: str, title: str) -> dict:
    """导入资料，触发 AI 知识点提取

    Args:
        session_id: 所属会话 ID
        content: 原始文本内容（Markdown/纯文本）
        title: 文档标题（用于显示）

    Returns:
        {"document_id": int, "knowledge_points": [{"title", "content"}, ...]}
    """
    # 1. 原始文档存 MySQL，拿到 ID
    doc = Document(session_id=session_id, title=title, content=content)
    db.add(doc)
    db.commit()

    # 2. 构造 LangGraph 输入状态
    state: AgentState = {
        "messages": [],
        "session_id": session_id,
        "raw_content": content,
        "document_id": doc.id,
        "knowledge_points": [],
        "current_kp": {},
        "current_question": "",
        "user_answer": "",
        "evaluation": {},
        "kp_index": 0,
        "exit_review": False,
    }

    # 3. 跑 import_graph（一次性图，跑完就结束）
    result = import_graph.invoke(state)

    # 4. 返回 LLM 提取的知识点列表
    kps = result.get("knowledge_points", [])
    return {"document_id": doc.id, "knowledge_points": kps}
