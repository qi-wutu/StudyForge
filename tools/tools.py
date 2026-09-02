"""搜索工具 — 供 LLM ReAct 调用

把 web_search 包装成 LangChain Tool，
LLM 在 question_gen 节点中可以自主决定是否搜索。
"""

from langchain_core.tools import tool

from tools.engine import web_search


@tool
def search_web(query: str) -> str:
    """搜索网络获取最新资料。当你觉得当前知识不足以出一
    道好题时，可以搜索来获取更丰富的信息。

    Args:
        query: 搜索关键词
    """
    results = web_search(query)
    if not results:
        return "未搜索到相关结果。"

    lines = [f"找到 {len(results)} 条结果：\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"{i}. {title}")
        if body:
            lines.append(f"   {body[:200]}")
        if href:
            lines.append(f"   来源: {href}")
        lines.append("")
    return "\n".join(lines)
