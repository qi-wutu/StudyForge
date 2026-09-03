"""LLM 助手 — 所有子 Agent 共用的底层

从 V1 的 graph/node.py 拆出：
  - get_llm()       惰性单例的 ChatOpenAI（OpenAI 兼容协议，支持任意大模型）
  - react_json()    轻量 ReAct：LLM 可自主调用搜索工具，最终强制输出 JSON

子 Agent（review / import / qa / analyzer）都只依赖这里的 get_llm / react_json，
不再各自 new LLM。
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """获取全局惰性单例 LLM 实例"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    return _llm


def react_json(prompt: str, system: str, max_turns: int = 5) -> dict:
    """轻量 ReAct — LLM 带标准工具调用，最终输出 JSON dict

    使用 bind_tools 绑定搜索工具，LLM 可自主决定搜索。
    支持 DeepSeek、OpenAI、Claude 等标准 tool calling 协议。
    不依赖 LangGraph 的 ToolNode，子 Agent 内部自闭环。

    Args:
        prompt: 用户消息
        system: 系统提示
        max_turns: 最大工具调用轮数（防止死循环）

    Returns:
        解析后的 JSON dict
    """
    # 惰性 import：只在真正用搜索时拉工具栈，避免纯 LLM 调用也被拖进 ddgs
    from tools.tools import search_web

    llm = get_llm().bind_tools([search_web])
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]

    for _ in range(max_turns):
        response = llm.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  [ReAct] LLM 调用了 {tc['name']}({tc['args']})")
                result = search_web.invoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # 没有工具调用 → 这是最终输出
            parser = JsonOutputParser()
            return parser.parse(response.content)

    # 超过最大轮数 → 保留搜索历史，强制输出 JSON
    print(f"  [ReAct] 超过 {max_turns} 轮，基于已获取资料强制输出")
    messages.append(HumanMessage(content="基于以上所有搜索结果，直接输出最终的 JSON 结果。不要搜索。"))
    final = get_llm().invoke(messages)
    parser = JsonOutputParser()
    return parser.parse(final.content)
