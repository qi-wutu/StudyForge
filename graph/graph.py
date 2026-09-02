"""LangGraph 图定义

什么是"图"？
  图 = 把节点（Node）用边（Edge）连起来，形成一条执行流程。
  节点就是 node.py 里的函数，边决定了执行顺序。

本文件定义了两个图：
  import_graph  — 导入资料 → 提取知识点（一次性，跑完就结束）
  review_graph  — 复习循环：出题 → 等输入 → 判题 → 循环

两个图互相独立，不存在"一个图调用另一个图"。
main.py 根据用户命令选择跑哪个图。
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from graph.node import (
    planner, scheduler, question_gen, wait_input, judge,
    scheduler_should_continue, judge_should_continue,
)


# ========================================
# 图一：import_graph（一次性）
# ========================================
# 流程：planner(提取知识点) → END
# 功能：读文件 → DeepSeek 提取知识点 → 存 MySQL
#
# StateGraph(AgentState) = 这个图的状态类型是 AgentState
# add_node("节点名", 节点函数) = 注册一个工位
# set_entry_point("planner") = 从 planner 节点开始
# add_edge("planner", END) = planner 执行完后直接结束
# compile() = 编译成可执行的图

import_builder = StateGraph(AgentState)
import_builder.add_node("planner", planner)
import_builder.set_entry_point("planner")
import_builder.add_edge("planner", END)
import_graph = import_builder.compile()


# ========================================
# 图二：review_graph（循环）
# ========================================
# 流程：
#
#   scheduler (选知识点)
#       │
#       ├── (没有知识点) ──→ END
#       │
#       └── (有知识点) ──→ question_gen (出题)
#                               │
#                           wait_input (等用户打字)
#                               │
#                             judge (判分 + 存库)
#                               │
#                               ├── (退出) ──→ END
#                               │
#                               └── (继续) ──→ scheduler (下一题)
#
#
# 关键点：
#   - add_conditional_edges = 条件边，节点执行完后根据返回值走不同分支
#   - wait_input 里的 interrupt 让图暂停，等 main.py 恢复
#   - judge 判完根据 exit_review 决定是循环还是结束

review_builder = StateGraph(AgentState)

# 注册四个节点
review_builder.add_node("scheduler", scheduler)       # 从 DB 选知识点
review_builder.add_node("question_gen", question_gen)  # 调 LLM 出题
review_builder.add_node("wait_input", wait_input)      # 暂停等用户输入
review_builder.add_node("judge", judge)                # 调 LLM 判题

# 入口：从 scheduler 开始
review_builder.set_entry_point("scheduler")

# scheduler 的条件边
# scheduler_should_continue 返回 "question_gen" 或 "end"
review_builder.add_conditional_edges(
    "scheduler",
    scheduler_should_continue,
    {"question_gen": "question_gen", "end": END},
)

# question_gen → wait_input → judge（顺序执行，没有分支）
review_builder.add_edge("question_gen", "wait_input")
review_builder.add_edge("wait_input", "judge")

# judge 的条件边
# judge_should_continue 返回 "scheduler"（循环） 或 "end"（结束）
review_builder.add_conditional_edges(
    "judge",
    judge_should_continue,
    {"scheduler": "scheduler", "end": END},
)

review_graph = review_builder.compile(checkpointer=MemorySaver())
