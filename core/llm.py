"""LLM 助手 — 所有子 Agent 共用的底层

从 V1 的 graph/node.py 拆出：
  - get_llm()       惰性单例的 ChatOpenAI（OpenAI 兼容协议，支持任意大模型）
  - react_json()    轻量 ReAct：LLM 可自主调用搜索工具，最终强制输出 JSON
  - run_agent()     通用工具调用循环（V1.3）：LLM 绑定一组工具自主多轮决策

子 Agent（review / import / qa / analyzer）都只依赖这里的 get_llm / react_json，
不再各自 new LLM。
"""

import json

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


# ========================================
# run_agent — 通用工具调用循环（V1.3）
# ========================================

_TOOL_RESULT_MAX = 2000  # 回填给 LLM 的 ToolMessage 截断长度，防上下文爆炸


def run_agent(
    messages: list,
    tools: list,
    max_turns: int = 6,
) -> tuple[str, list]:
    """通用工具调用循环 — Supervisor 的「LLM 大脑」

    LLM 绑定一组工具，自主决定调哪个、多轮往复，最终给出一句自然语言回复。
    支持 DeepSeek / OpenAI / Claude 标准 tool calling 协议（同 bind_tools 路线）。

    Args:
        messages:  已拼好的消息序列（建议 [System] + 最近历史 + 当前用户消息）
        tools:     可调用的 LangChain Tool 列表
        max_turns: 最大工具调用轮数（防死循环）

    Returns:
        final_text: LLM 最终的自然语言回复（最后一次无 tool_calls 的 AIMessage 文本）
        tool_log:   [(tool_name, args, result_str), ...]，按调用顺序。
                    result_str 是 json.dumps 的结构化卡片 —— supervisor 据此
                    决定返回 type 并渲染，LLM 拿到的却是截断版（见 _TOOL_RESULT_MAX）。
    """
    tool_map = {t.name: t for t in tools}
    llm = get_llm().bind_tools(tools)
    tool_log: list[tuple] = []

    for _ in range(max_turns):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # 没有工具调用 → 这就是最终回复
            return _as_text(response.content), tool_log

        for tc in response.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or {}
            print(f"  [Agent] LLM 调用了 {name}({args})")

            tool = tool_map.get(name)
            if tool is None:
                result_str = json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
            else:
                try:
                    result_str = tool.invoke(args)
                    if not isinstance(result_str, str):
                        result_str = json.dumps(result_str, ensure_ascii=False)
                except Exception as e:  # noqa: BLE001 —— 单工具失败不拖垮整个循环
                    result_str = json.dumps({"error": f"工具执行失败: {e}"}, ensure_ascii=False)

            tool_log.append((name, args, result_str))
            # 塞回给 LLM 的 ToolMessage 截断；完整卡片在 tool_log 里
            messages.append(ToolMessage(
                content=result_str[:_TOOL_RESULT_MAX],
                tool_call_id=tc.get("id", ""),
            ))

    # 超过 max_turns → 强制要求基于已有结果直接回答，不再调用工具
    print(f"  [Agent] 超过 {max_turns} 轮，要求直接回答")
    messages.append(HumanMessage(content="基于以上工具结果，用一句自然的回复总结给用户，不要再调用任何工具。"))
    final = llm.invoke(messages)
    return _as_text(final.content), tool_log


def _as_text(content) -> str:
    """把 LLM 返回的 content（str 或内容块列表）规整成纯文本"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            # tool_use 块只有元信息，最终会以 AIMessage.tool_calls 处理，忽略
        elif block is not None:
            parts.append(str(block))
    return "\n".join(p for p in parts if p)
