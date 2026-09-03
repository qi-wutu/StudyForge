"""问答子 Agent — QAAgent

职责：用户用自然语言问知识点（"什么是 GMP 模型"），
从该会话的知识点里检索相关内容，让 LLM 基于检索到的资料回答（grounded，不瞎编）。
必要信息不足时可引导用户导入资料（暂不主动上网搜——那留给 V1.3 增强）。

复用：混合检索（rag.retriever.get_session_retriever）+ 通用 LLM（core.llm）。
对应 V1.1 的 service/qa_service.py。
"""

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm import get_llm
from rag.retriever import get_session_retriever
from storage.db import db
from storage.schemas import KnowledgePoint


def answer_question(session_id: int, question: str) -> tuple[str, bool]:
    """回答一个知识问题。

    Returns:
        (回答文本, 是否有可用资料)
        has_context=False 表示该会话没有资料/检索不到相关内容，
        上层据此提示用户导入或改用通用提问。
    """
    kps = db.query(KnowledgePoint).filter_by(session_id=session_id).count()
    if kps == 0:
        return "这个会话还没导入资料，我暂时没法基于你的资料回答。你可以先导入，或者告诉我你想学/测什么。", False

    retriever = get_session_retriever(session_id)
    results = retriever.search(question, top_k=4)

    # 过滤掉低相关片段
    relevant = [
        doc.strip() for _, doc, score in results
        if doc and doc.strip() and score > 0.3
    ]
    if not relevant:
        return "你的资料里好像没有直接相关的内容。要我换个问法，或者把相关资料导入进来吗？", False

    context = "\n".join(f"- {d[:500]}" for d in relevant)

    prompt = f"""你是学习助手。请基于下面这些学习资料，回答用户的问题。

要求：
1. 只依据资料回答，资料没提到的就直说「资料里没有」，不要编造
2. 回答简洁、准确，像给同学讲题一样

资料：
{context}

用户问题：{question}"""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="你是基于给定资料回答的学习助手。"),
        HumanMessage(content=prompt),
    ])
    return response.content.strip(), True
