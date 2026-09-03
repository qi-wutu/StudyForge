"""导入子 Agent — ImportAgent

职责：用户粘贴文本 / 上传文件 → 提取知识点（BM25 语义去重 + 向量 embedding）→ 存库。

内部是一个一次性 LangGraph（import_graph：planner → END），
对应 V1 原 service/import_service.py + graph/node.py 的 planner。

对外暴露模块函数 import_content()（chat 与 button 两个入口共用）：
    import_content(session_id, content, title) -> {"document_id", "knowledge_points"}
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph

from agent.state import AgentState
from core.llm import get_llm
from rag.bm25 import BM25Index
from rag.retriever import invalidate_retriever
from storage.db import db
from storage.schemas import Document, KnowledgePoint


# ========================================
# 节点：planner（提取知识点）
# ========================================

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
    invalidate_retriever(sid)

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


# ========================================
# 导入图（一次性）
# ========================================

_import_builder = StateGraph(AgentState)
_import_builder.add_node("planner", planner)
_import_builder.set_entry_point("planner")
_import_builder.add_edge("planner", END)
_import_graph = _import_builder.compile()


# ========================================
# 对外入口
# ========================================

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
    result = _import_graph.invoke(state)

    # 4. 返回 LLM 提取的知识点列表
    kps = result.get("knowledge_points", [])
    return {"document_id": doc.id, "knowledge_points": kps}
