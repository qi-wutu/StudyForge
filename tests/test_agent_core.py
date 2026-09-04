"""V1.3 — 工具层 + 工具循环的单元测试（全部 mock 底层子 Agent / LLM，不真调 API）

覆盖三块：
  1. build_tools() 把 6 个 Agent 工具正确串到 review/qa/analyze/import，
     返回 JSON 卡片、并按闭包读写 conv 里的复习线程状态。
  2. core.llm.run_agent() 的工具调用循环：无工具路径 / 单工具路径 /
     超轮数强制回答 / 工具抛错不崩。
  3. Supervisor 的确定性安全快路（退出/导入前缀）不经过 LLM。
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import core.llm
from agent.tools import build_tools


def make_conv(**over):
    conv = {
        "session_id": 1,
        "messages": [],
        "review_thread_id": None,
        "pending_question": False,
        "current_card": None,
    }
    conv.update(over)
    return conv


# ========================================
# 一、build_tools — 工具接线
# ========================================

class FakeReview:
    """代替 review_agent 单例，测试工具怎么调用它。"""

    def __init__(self):
        self.calls = []

    def start(self, session_id):
        self.calls.append(("start", session_id))
        return {
            "thread_id": "thread-1",
            "question": "请解释 GMP 模型",
            "kp_title": "GMP",
            "kp_content": "G-M-P 分别对应...",
            "review_reason": "薄弱优先",
        }

    def submit_answer(self, thread_id, answer):
        self.calls.append(("submit", thread_id, answer))
        if answer == "退出式回答":
            return {"evaluation": {"score": 0, "comment": ""}, "exit": True}
        return {
            "evaluation": {"score": 80, "comment": "基本正确", "strengths": [], "weaknesses": []},
            "exit": False,
        }

    def get_next(self, thread_id):
        self.calls.append(("get_next", thread_id))
        return {
            "thread_id": thread_id,
            "question": "GMP 里 work stealing 怎么减少竞争?",
            "kp_title": "GMP",
            "kp_content": "...",
            "review_reason": "",
        }

    def exit(self, thread_id):
        self.calls.append(("exit", thread_id))
        return {"exit": True}


def _fake_answer_question(session_id, question):
    return ("回答文本", True)


def _fake_do_import(session_id, content, title):
    return {"document_id": 1, "knowledge_points": [{"title": "a"}, {"title": "b"}]}


def _fake_analyze(session_id, llm_report=True):
    return {"kp_stats": [], "global_stats": {}, "llm_report": "报告"}


def test_start_review_sets_conv_and_cards(monkeypatch):
    fake = FakeReview()
    monkeypatch.setattr("agent.tools.review_agent", fake)
    conv = make_conv()
    tools = {t.name: t for t in build_tools(1, conv)}

    payload = json.loads(tools["start_review"].invoke({}))

    assert payload == {"ok": True, "card": "question", "data": {
        "question": "请解释 GMP 模型", "kp_title": "GMP",
        "kp_content": "G-M-P 分别对应...", "review_reason": "薄弱优先",
    }}
    assert fake.calls == [("start", 1)]
    assert conv["review_thread_id"] == "thread-1"
    assert conv["pending_question"] is True
    assert conv["current_card"]["question"] == "请解释 GMP 模型"


def test_start_review_no_kp_returns_error(monkeypatch):
    class NoKp:
        def start(self, session_id):
            raise ValueError("当前会话没有知识点，请先导入资料")
    monkeypatch.setattr("agent.tools.review_agent", NoKp())
    tools = {t.name: t for t in build_tools(1, make_conv())}

    payload = json.loads(tools["start_review"].invoke({}))
    assert payload["ok"] is False
    assert payload["card"] == "chat"
    assert "没有知识点" in payload["text"]


def test_submit_answer_answers_then_reads_next(monkeypatch):
    fake = FakeReview()
    monkeypatch.setattr("agent.tools.review_agent", fake)
    conv = make_conv(review_thread_id="thread-1", pending_question=True)
    tools = {t.name: t for t in build_tools(1, conv)}

    payload = json.loads(tools["submit_answer"].invoke({"answer": "我的回答"}))

    assert fake.calls == [("submit", "thread-1", "我的回答"), ("get_next", "thread-1")]
    assert payload["ok"] is True
    assert payload["card"] == "review_result"
    assert payload["data"]["exit"] is False
    assert payload["data"]["evaluation"]["score"] == 80
    assert payload["data"]["next"]["question"].startswith("GMP 里 work stealing")
    # 下一题接管为新的待回答题
    assert conv["pending_question"] is True
    assert conv["current_card"]["question"].startswith("GMP 里 work stealing")


def test_submit_answer_exit_clears_thread(monkeypatch):
    fake = FakeReview()
    monkeypatch.setattr("agent.tools.review_agent", fake)
    conv = make_conv(review_thread_id="thread-1", pending_question=True)
    tools = {t.name: t for t in build_tools(1, conv)}

    payload = json.loads(tools["submit_answer"].invoke({"answer": "退出式回答"}))

    assert payload["card"] == "review_result"
    assert payload["data"]["exit"] is True
    assert payload["data"]["next"] is None
    assert conv["review_thread_id"] is None
    assert conv["pending_question"] is False


def test_submit_answer_without_review_guides(monkeypatch):
    monkeypatch.setattr("agent.tools.review_agent", FakeReview())
    tools = {t.name: t for t in build_tools(1, make_conv())}
    payload = json.loads(tools["submit_answer"].invoke({"answer": "x"}))
    assert payload["ok"] is False
    assert "开始复习" in payload["text"]


def test_exit_review_idempotent(monkeypatch):
    fake = FakeReview()
    monkeypatch.setattr("agent.tools.review_agent", fake)
    # 无进行中复习：直接给引导，不调底层
    tools = {t.name: t for t in build_tools(1, make_conv())}
    p1 = json.loads(tools["exit_review"].invoke({}))
    assert p1["card"] == "chat" and fake.calls == []

    # 有进行中复习：调 exit 并清状态
    conv = make_conv(review_thread_id="thread-1")
    tools = {t.name: t for t in build_tools(1, conv)}
    p2 = json.loads(tools["exit_review"].invoke({}))
    assert fake.calls == [("exit", "thread-1")]
    assert conv["review_thread_id"] is None and conv["pending_question"] is False


def test_answer_question_card(monkeypatch):
    monkeypatch.setattr("agent.tools._answer_question", _fake_answer_question)
    tools = {t.name: t for t in build_tools(1, make_conv())}
    payload = json.loads(tools["answer_question"].invoke({"question": "什么是 GMP"}))
    assert payload == {"ok": True, "card": "answer", "text": "回答文本", "has_context": True}


def test_analyze_weakness_ok_and_error(monkeypatch):
    monkeypatch.setattr("service.stats_service.analyze", _fake_analyze)
    tools = {t.name: t for t in build_tools(1, make_conv())}
    payload = json.loads(tools["analyze_weakness"].invoke({}))
    assert payload["card"] == "analysis" and payload["ok"] is True

    monkeypatch.setattr("service.stats_service.analyze", lambda sid, llm_report=True: {"error": "暂无答题记录"})
    tools = {t.name: t for t in build_tools(1, make_conv())}
    payload = json.loads(tools["analyze_weakness"].invoke({}))
    assert payload["ok"] is False and payload["card"] == "chat"
    assert "暂无答题记录" in payload["text"]


def test_import_content_card(monkeypatch):
    monkeypatch.setattr("agent.tools._do_import", _fake_do_import)
    tools = {t.name: t for t in build_tools(1, make_conv())}
    payload = json.loads(tools["import_content"].invoke({"content": "这是一段超过二十个字的资料内容用来测试导入" * 2}))
    assert payload == {"ok": True, "card": "imported", "data": {"count": 2}}


def test_import_content_too_short_does_not_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("太短不应真正触发导入")
    monkeypatch.setattr("agent.tools._do_import", boom)
    tools = {t.name: t for t in build_tools(1, make_conv())}
    payload = json.loads(tools["import_content"].invoke({"content": "太短"}))
    assert payload["ok"] is False and payload["card"] == "chat"


# ========================================
# 二、run_agent — 工具调用循环
# ========================================

class FakeLLM:
    """脚本式假 LLM：依次返回预设 AIMessage，并记录每次 invoke 拿到的消息。"""

    def __init__(self, steps):
        self._steps = list(steps)
        self.seen = []

    def bind_tools(self, tools):
        self.tools = {t.name: t for t in tools}
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self._steps.pop(0)
        return step(messages) if callable(step) else step


def _msg(content="", tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls)


def test_run_agent_pure_chat_no_tools(monkeypatch):
    fake = FakeLLM([AIMessage(content="你好呀，想学点啥？")])
    monkeypatch.setattr(core.llm, "get_llm", lambda: fake)

    final, log = core.llm.run_agent([HumanMessage(content="你好")], tools=[])
    assert final == "你好呀，想学点啥？"
    assert log == []
    assert fake.tools == {}


def test_run_agent_single_tool_then_final(monkeypatch):
    from langchain_core.tools import tool

    @tool
    def echo(text: str) -> str:
        """原样返回。"""
        return f"echo:{text}"

    def second_step(messages):
        # 第二条消息应是塞回给模型的 ToolMessage
        assert isinstance(messages[-1], ToolMessage)
        assert messages[-1].content == "echo:hi"
        assert messages[-1].tool_call_id == "call_1"
        return AIMessage(content="收到你的 hi 了")

    fake = FakeLLM([
        _msg(tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "call_1", "type": "tool_call"}]),
        second_step,
    ])
    monkeypatch.setattr(core.llm, "get_llm", lambda: fake)

    final, log = core.llm.run_agent([HumanMessage(content="hi")], tools=[echo])

    assert final == "收到你的 hi 了"
    assert log == [("echo", {"text": "hi"}, "echo:hi")]


def test_run_agent_tool_error_does_not_crash(monkeypatch):
    from langchain_core.tools import tool

    @tool
    def boom() -> str:
        """必炸。"""
        raise RuntimeError("工具炸了")

    def second_step(messages):
        last = messages[-1]
        assert isinstance(last, ToolMessage)
        assert "工具炸了" in last.content
        return AIMessage(content="我换种方式回答你")

    fake = FakeLLM([
        _msg(tool_calls=[{"name": "boom", "args": {}, "id": "call_1", "type": "tool_call"}]),
        second_step,
    ])
    monkeypatch.setattr(core.llm, "get_llm", lambda: fake)

    final, log = core.llm.run_agent([HumanMessage(content="跑一下")], tools=[boom])
    assert final == "我换种方式回答你"
    assert log and "error" in log[0][2]


def test_run_agent_force_answer_after_max_turns(monkeypatch):
    """超过 max_turns 后：追加「别调工具直接答」再收尾，不会死循环。"""
    fake = FakeLLM([
        _msg(tool_calls=[{"name": "nope", "args": {}, "id": "c1", "type": "tool_call"}]),
        _msg(tool_calls=[{"name": "nope", "args": {}, "id": "c2", "type": "tool_call"}]),
        lambda messages: (AIMessage(content="基于已有结果直接答") if any(
            isinstance(m, HumanMessage) and "不要再调用任何工具" in (m.content or "")
            for m in messages
        ) else AIMessage(content="兜底")),
    ])
    monkeypatch.setattr(core.llm, "get_llm", lambda: fake)

    # tools=[]：调用"nope"是未知工具，返回错误串，但仍推进轮数
    final, log = core.llm.run_agent([HumanMessage(content="hi")], tools=[], max_turns=2)

    assert final == "基于已有结果直接答"
    assert len(log) == 2
    assert all(name == "nope" and "未知工具" in result for name, _a, result in log)


# ========================================
# 三、Supervisor 安全快路 — 不经过 LLM
# ========================================

def test_supervisor_exit_fast_path_skips_llm(monkeypatch):
    from agent.supervisor import supervisor

    def boom(*a, **k):
        raise AssertionError("安全快路不应调用 LLM")
    monkeypatch.setattr(core.llm, "get_llm", boom)

    # 每次用独立 sid，避免污染其他测试的单例状态
    result = supervisor.chat(90001, "退出")
    assert result["type"] == "chat"
    assert "没有进行中的复习" in result["text"]


def test_supervisor_import_fast_path_short_skips_llm(monkeypatch):
    from agent.supervisor import supervisor

    def boom(*a, **k):
        raise AssertionError("安全快路不应调用 LLM")
    monkeypatch.setattr(core.llm, "get_llm", boom)

    result = supervisor.chat(90002, "导入：太短")
    assert result["type"] == "chat"
    assert "导入" in result["text"]
