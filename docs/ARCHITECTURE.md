# StudyForge 架构文档

> 版本：2.2（V1.3 LLM 大脑 · Web 版）  
> 最后更新：2026-09-03

---

## 架构现状一句话（V1.3）

V1.3 把 supervisor 从「规则路由器」升级成 **LLM 大脑**——它会自己决定调哪个能力、能记得你刚说过/刚导入的。底层机制**没变**——LangGraph 复习循环图、混合检索、LLM 判分、Agent Memory 全部原样保留，只是顶层调度换成了 **LLM 工具调用循环**：

```
HTTP 层   backend/router.py
   │  只做 HTTP 解析 + JSON 返回，不碰业务
   ▼
Agent 层  agent/
   supervisor（主 Agent：LLM 大脑）    ←── nlu/intent.py 只拦「退出/下一题/导入」确定性快路
   │  core/llm.py::run_agent() 跑工具循环：LLM 自主决定调哪个、多轮往复
   ▼
   agent/tools.py::build_tools() —— 把 4 个子 Agent 的能力打包成 7 个可调用工具
   ├─ start_review / submit_answer / exit_review → review_agent（LangGraph 复习图 + turn API）
   ├─ answer_question                            → qa_agent（混合检索 + grounded 回答）
   ├─ analyze_weakness                           → analyzer（统计聚合 + LLM 报告）
   └─ import_content / general_search            → import_agent / tools 搜索
   │
   ▼ 复用底座
service/（session/stats 无状态读服务 + 缓存）
core/（get_llm + run_agent） rag/（BM25 + 向量） tools/（搜索） storage/（MySQL）
```

下文凡是讲图节点机制（scheduler / question_gen / judge / planner）的地方，都以 `agent/` 下的实现为准：

- 复习循环图定义在 `agent/review_agent.py` 的 `_build_review_graph()`
- 导入一次性图在 `agent/import_agent.py`
- 图共享状态 `AgentState` 在 `agent/state.py`
- 通用工具循环 `run_agent()` 在 `core/llm.py`；工具打包 `build_tools()` 在 `agent/tools.py`

> 演进叙事：V1.1 给 V1 加了对话皮层，V1.2 把代码重构成「主 Agent + 4 子 Agent」的多 Agent 结构，V1.3 把顶层 supervisor 从规则 if/else 升级成 **LLM 工具调用循环**（见第 3.2 / 12 节）。现在这套「决策 → 调工具 → 多轮」跑在命令式循环里，把它画成一张真正的 LangGraph 主图（interrupt 上移到主图边界）是 V2 的核心。

---

## 目录

1. [项目定位](#1-项目定位)
2. [整体架构概览](#2-整体架构概览)
3. [分层架构详解](#3-分层架构详解)
   - 3.1 [Controller 层（backend/）](#31-controller-层-backend)
   - 3.2 [Agent 层（agent/）](#32-agent-层-agent)
   - 3.3 [Service 层（service/）](#33-service-层-service)
   - 3.4 [底座层（core/ rag/ tools/ storage/）](#34-底座层core-rag-tools-storage)
4. [数据流详解](#4-数据流详解)
   - 4.1 [导入流程](#41-导入流程)
   - 4.2 [复习流程](#42-复习流程)
   - 4.3 [分析流程](#43-分析流程)
5. [数据库设计](#5-数据库设计)
6. [缓存策略](#6-缓存策略)
7. [LangGraph 图详解](#7-langgraph-图详解)
8. [混合检索（RAG）](#8-混合检索-rag)
9. [前端架构](#9-前端架构)
10. [API 规范](#10-api-规范)
11. [部署方案](#11-部署方案)
12. [与 V2 的关系](#12-与-v2-的关系)

---

## 1. 项目定位

StudyForge 是一个 **AI 自适应复习系统**，核心业务流程是：

```
导入资料 → AI 提取知识点 → 智能出题 → 用户回答 → AI 判分 → 薄弱分析 → 针对性复习
```

它不是一个刷题工具（不是选择题/填空题），而是**模拟面试官**：出主观题、让用户自由回答、AI 给评分和反馈、找出薄弱点、下一次优先出薄弱点的题。

### 适用场景

- 面试准备（Go 八股、Redis、操作系统等）
- 课程复习（期末前的概念性内容）
- 技术知识体系梳理

### 非目标

- 标准化考试刷题（选择题/判断题）
- 多人协作学习平台
- 教学内容创作

---

## 2. 整体架构概览

```
┌──────────────────────────────────────────────────────────────────────┐
│                    用户浏览器（React 19 + TS + Vite）                  │
│  Dashboard / 对话 / 会话 / 导入 / 复习 / 分析                          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP / JSON（?session_id=X）
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FastAPI（Python）                                                     │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ Controller 层  backend/router.py                                 │  │
│  │   只做：解析参数 → 调 agent/service → 返回 JSON                  │  │
│  └──────────────────────────────┬──────────────────────────────────┘  │
│                                 ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ Agent 层  agent/  （V1.3：LLM 大脑）                             │  │
│  │   supervisor  主 Agent：读对话历史 → LLM 决定调哪个工具     │  │
│  │   tools.build_tools()   把 4 个子 Agent 打包成 7 个工具     │  │
│  │   core.llm.run_agent()  跑多轮工具循环（core/llm.py）       │  │
│  │   review / import / qa / analyzer：四个子 Agent             │  │
│  │   nlu/intent.py 退为「退出/下一题/导入」确定性快路          │  │
│  └──────────────┬──────────────────────┬────────────────────────────┘  │
│                 │ 复用底座               │ 无状态读服务                 │
│  ┌────────────────────────────┐   ┌───────────────────────────────┐   │
│  │ core/  llm.py（get_llm +   │   │ service/                       │   │
│  │        react_json 轻量ReAct）│   │  session_service（会话 CRUD）  │   │
│  │ rag/   BM25 + 向量混合检索  │   │  stats_service（Dashboard /    │   │
│  │ tools/ 搜索（DDG + Bing）   │   │  知识点统计 / 薄弱分析，缓存）  │   │
│  │ storage/ MySQL + ORM（6表） │   └───────────────────────────────┘   │
│  └────────────────────────────┘                                      │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
          │                         │
          ▼                         ▼
   ┌────────────┐          ┌──────────────────┐
   │   MySQL    │          │  DeepSeek API    │
   │   6 张表   │          │（OpenAI 兼容）    │
   └────────────┘          └──────────────────┘
```

### 设计原则

1. **分层分离** — Controller 不碰业务，Agent/Service 不碰 HTTP，底座专心管 LLM/检索/数据
2. **职责收敛** — 每个子 Agent 只做一件事（复习 / 导入 / 问答 / 分析），接口可独立调用、独立测试
3. **图内/图外分离** — LangGraph 节点内只关注"该做什么事"，不关心 HTTP、不关心存储
4. **隔离优先** — 会话（Session）间数据全隔离，没有跨会话引用
5. **降级设计** — 向量检索失败 → 降级 BM25；搜索 API 失败 → 跳过搜索；embedding 失败 → 不阻塞导入
6. **缓存不是必须的** — 所有缓存都是内存级，服务重启后重建，不会丢业务数据

---

## 3. 分层架构详解

### 3.1 Controller 层（backend/）

#### backend/main.py

```python
# 职责：创建 FastAPI 实例、挂中间件、挂路由、挂静态文件
app = FastAPI()
app.add_middleware(CORSMiddleware, ...)  # 跨域
app.include_router(router)               # API 路由
app.mount("/", StaticFiles(...))         # 前端静态文件
```

只有几十行，不含任何业务逻辑。启动事件里调 `init_db()` 做建表和迁移。

#### backend/router.py

```python
# 每个端点只做三件事：
@router.get("/api/stats")
def get_stats(session_id: Optional[int] = Query(None)):
    sid = session_service.resolve_session_id(session_id)  # 1. 拿 session_id
    return stats_service.get_dashboard_stats(sid)          # 2. 调 service/agent
                                                           # 3. 返回（隐式 JSON）
```

```python
# 自然语言入口 → 主 Agent（V1.3：LLM 工具循环）
@router.post("/api/chat")
def chat(req: ChatRequest, session_id: Optional[int] = Query(None)):
    sid = session_service.resolve_session_id(session_id)
    return supervisor.chat(sid, req.message)
```

- 不直接调 `db.query()`、不调 `graph.invoke()`
- 不维护任何状态（活跃复习 thread、对话记忆都在 Agent 层维护）
- 异常统一转 HTTP 错误码
- session_id 由前端 localStorage 持有，通过 `?session_id=X` 传过来

**为什么 router 不直接调图 / 节点？**  
为了可测试性。如果 router 里嵌了 LangGraph 调用，写单元测试就得 mock HTTP 请求。现在只需要测 `review_agent.start()`、`supervisor.chat()`、`stats_service.analyze()` 这类公开方法就行。

---

### 3.2 Agent 层（agent/）

V1.2 之前，代码是「`graph/` 上帝模块 + 一堆 service」，负责调度的逻辑散落在 `router` 和 `chat_service` 里，讲不清"Agent"。V1.2 把能力按职责切分，**每个子 Agent 独立成模块**，由主 Agent `supervisor` 统一调度。

| 文件 | 角色 | 职责 / 公开接口 |
| --- | --- | --- |
| `supervisor.py` | 主 Agent（LLM 大脑） | `Supervisor.chat(session_id, message)`：确定性安全快路 → LLM 工具循环 → 返回结构化结果 |
| `tools.py` | 工具集 | `build_tools(session_id, conv)`：把 4 个子 Agent + 统计能力打包成 7 个可调用工具 |
| `review_agent.py` | 复习子 Agent | LangGraph 循环图（调度/出题/判分）+ `ReviewAgent`：`start/answer/next/exit` |
| `import_agent.py` | 导入子 Agent | `import_content(session_id, content, title)` → `{document_id, knowledge_points}` |
| `qa_agent.py` | 问答子 Agent | `answer_question(session_id, question)` → `(text, has_context)` |
| `analyzer.py` | 分析子 Agent | `analyze(session_id)` → `{kp_stats, global_stats, llm_report}` |
| `state.py` | 共享类型 | `AgentState` TypedDict — 所有图的共享状态 |

> 复习 / 导入仍用 LangGraph，只是**图定义和节点函数跟着对应 Agent 走**（`_build_review_graph()` / `import_agent` 的 planner 图），节点机制在第 7 节详述。`graph/` 目录已在 V1.2 删除。

#### Supervisor：主 Agent（V1.3 LLM 大脑）

V1.3 的 supervisor 不再是规则路由器，而是跑在 **LLM 工具调用循环** 上的"大脑"：

```python
class Supervisor:
    """主 Agent：LLM 大脑——读对话历史，自己决定调哪个工具"""

    def chat(self, session_id: int, message: str) -> dict:
        conv = self._get_conv(session_id)          # 每 session 一份对话记忆
        fast = fast_path_intent(message)           # ① 确定性安全快路（薄薄一层）
        if fast is not None:
            return self._fast_path(conv, fast, message)
        messages = [SystemMessage(content=self._system_prompt(conv))]
        messages += self._to_history(conv["messages"][-8:])   # ② 最近对话喂给 LLM
        messages.append(HumanMessage(content=message))
        final_text, tool_log = run_agent(
            messages, build_tools(session_id, conv),          # ③ 跑工具循环
        )
        return self._map_result(final_text, tool_log)         # ④ 按最后一张卡定 type
```

`supervisor.chat()` 内部做四件事：

1. **确定性安全快路** — `nlu/intent.py::fast_path_intent()` 只拦「退出 / 下一题 / 导入前缀」三种最确定、最不该让 LLM 猜的命令，不花 LLM。
2. **对话历史真正喂给 LLM** — V1.1 的历史只存展示文本、从未给模型；V1.3 拼 `[System 提示 + 最近 8 条对话 + 当前消息]`，所以它记得你刚说过/刚导入的。
3. **跑工具循环** — `run_agent()`（`core/llm.py`）里 LLM 绑定 `build_tools()` 的 7 个工具，自主决定调哪个、可多轮往复、超轮强制收尾。
4. **结果映射** — 由 `tool_log` 里"最后一张结构化卡片"决定返回 type；question / review_result / analysis / imported 卡额外带 LLM 的最终话作前言气泡。

**工具循环里的调度**（路由决策权已从规则表交给 LLM，下表是它"通常会这么选"）：

| 用户说的话（示意） | LLM 调用的工具 | 返回 type |
| --- | --- | --- |
| 「开始复习 / 考我 / 出题」 | `start_review` | `question` |
| （复习中）直接作答 | `submit_answer` | `review_result` |
| 「退出 / 结束 / 不考了」 | `exit_review` | `chat` |
| 「我哪里薄弱 / 分析一下」 | `analyze_weakness` | `analysis` |
| 「导入：xxx」 | `import_content` | `imported` |
| 「什么是 GMP？」 | `answer_question` | `answer` |
| 资料之外 / 要较新信息 | `general_search` | `chat` |
| 闲聊 / 拿不准 | （不调工具）直接回 | `chat` |

**复习中的消歧（关键设计，写进 System Prompt）**：提示里会带出当前「待回答题目」。复习中用户发普通消息优先视为**作答**（→ `submit_answer`），明显是新提问走 `answer_question`，是命令（退出/分析/导入）走对应工具。

**为什么留一层规则快路？** 纯 LLM 是概率性的，偶尔会把「退出」接成闲聊。像退出 / 下一题 / 导入这种最确定的廉价命令用正则兜住，既不误判也省一次 LLM 调用；这层随时可删，不影响主流程。安全快路复用 `build_tools` 里的同名工具（`exit_review` / `import_content`），行为保持单一来源。

> 局限（也是 V2 的入口）：这套「LLM 决策 → 调工具 → 多轮」目前跑在 `run_agent()` 的命令式循环里，interrupt 没有上移到主图边界、子能力也没在图上显式编排。把决策层画成一张可中断、可检查点的 LangGraph 主图，就是 V2 的核心工作。

#### nlu/ — 意图皮层（V1.3 起退为确定性安全快路）

```
nlu/intent.py
  classify_intent(message, review_active=False) -> intent    # 8 类完整保留，测试锁定
    ├─ 正则强命令（无条件最高优先）
    │    exit_review / start_review / next / analyze / import
    ├─ 复习中：非命令、非提问 → answer
    └─ 提问判定（问号结尾 / 疑问词开头）→ qa，否则 smalltalk

  fast_path_intent(message) -> str | None                    # V1.3 新增：确定性快路
    └─ 只拦最确定的小撮命令：exit_review / next / import（前缀）
```

V1.3 之后，supervisor 的**主流程不再走 `classify_intent`**——对话直接交给 LLM 工具循环。`nlu/` 保留两样东西：

- **8 类正则完整保留**（`classify_intent`，13 个用例锁在 `tests/test_intent.py`）——V1.1 的规则能力没丢，只是不再当主路由。
- **`fast_path_intent()` 确定性快路**——只拦「退出 / 下一题 / 导入」这类绝不该让 LLM 猜的命令，不花 LLM、防误判。

规则快路 = **零 LLM 调用、可预测、可单测**。V1.3 新增的工具层/循环单测在 `tests/test_agent_core.py`（16 个用例，mock 底层子 Agent 与 LLM）。

#### ReviewAgent（复习子 Agent）详解

复习图是一个**循环图**，通过 `interrupt()` 把一次复习拆到多次 HTTP 调用里分步执行。V1.2 之后由 `ReviewAgent` 封装这组 turn API：

```python
class ReviewAgent:
    """复习 Agent — 管理每个 thread 的复习会话

    复习图是循环图，靠 interrupt() 拆到多个 turn 里分步调用：

    start():
        → 生成 thread_id (uuid)
        → review_graph.invoke(state, config)
        → 图跑到 wait_input 的 interrupt() 暂停
        → 返回第一题（thread_id, question, kp_title, ...）

    submit_answer(thread_id, answer):
        → Command(resume=answer) 唤醒图
        → 继续：judge → scheduler → question_gen → wait_input
        → 又暂停，返回评价 + 下一题

    get_next(thread_id):
        → review_graph.get_state(config)  # 不调 invoke，不跑图
        → 从 checkpointer 读当前 state

    exit(thread_id):
        → Command(resume="__exit__")
        → wait_input 看到 __exit__，设 exit_review=True，图正常结束
    """
```

关键设计点：

- **图内部**：节点函数（`scheduler` / `question_gen` / `wait_input` / `judge`）和 `_build_review_graph()` 都在 `review_agent.py` 内，LangGraph + MemorySaver checkpointer 原样保留。
- **单例**：`review_agent = ReviewAgent()` 模块级实例化，所有请求共用——因为它要维护 `_active_reviews` 字典（thread_id → 会话状态）。
- **thread_id**：每次 `start()` 生成 uuid 作为复习会话标识。未完成复习在内存里，服务重启后丢失，需要重新 start（一次复习通常几分钟，设计上接受）。
- **预生成**：`prewarm(session_id)` 在切换会话时后台静默预生成题目，不阻塞用户操作。

#### 为什么子 Agent 可以独立工作

每个子 Agent 只依赖底座（core/rag/storage）或无状态 service，不依赖 HTTP 层：

```text
supervisor ──┬──► review_agent（→ rag.retriever / core.llm / stats_service._clear_caches）
             ├──► import_agent（→ rag.bm25 / rag.retriever.invalidate / core.llm）
             ├──► qa_agent    （→ rag.retriever / core.llm）
             └──► analyzer    （→ core.llm / storage 查询）
```

将来接 V2（LangGraph 主图）时，每个子 Agent 的 turn 接口可以直接作为图上的节点或 Tool 复用，不用再搬家。

---

### 3.3 Service 层（service/）

V1.2 之后 service 层收敛为**无状态读服务**：不带图、不带对话状态，只做查询 + 聚合 + 缓存。

| 文件 | 职责 | 关键函数 |
| --- | --- | --- |
| `session_service.py` | 会话的 CRUD、ID 解析 | `resolve_session_id()`, `create_session()`, `list_sessions()` |
| `stats_service.py` | Dashboard / 知识点统计 / 薄弱分析 + 缓存 | `get_dashboard_stats()`, `list_knowledge_points()`, `analyze()` |

#### session_service.py 详解

```python
# 无状态设计：后端不存"当前会话"
# session_id 由前端 localStorage 持有，每个请求通过 ?session_id=X 传过来

def resolve_session_id(session_id: int | None = None) -> int:
    """解析会话 ID：前端传了就用，没传就 fallback 到 default"""
    if session_id is not None:
        s = db.query(Session).filter_by(id=session_id).first()
        if s:
            return s.id
    # fallback：default 会话
    ...
```

为什么要这样设计？
1. **REST 无状态** — 后端不记客户端状态，每个请求自包含
2. **多标签页不乱** — 每个标签页可以看不同的会话
3. **负载均衡友好** — 请求打到任何一台机器都能正确处理
4. **前端更可控** — 切换会话只需改 localStorage，不发请求也可以

#### stats_service.py 详解

```python
# 两级缓存：
#   1. 15 分钟 TTL
#   2. max_record_id 版本号（有新答题记录立即失效）

_CACHE_TTL = 900

def _is_cache_valid(entry, session_id) -> bool:
    if time.time() - entry["cached_at"] > _CACHE_TTL:
        return False
    if _get_max_record_id(session_id) != entry["max_record_id"]:
        return False
    return True
```

- 分析结果缓存 15 分钟。如果期间有新的答题记录（`max_record_id` 变化），立即重新计算。
- `analyze()` 的数据聚合逻辑委托给 `agent/analyzer.py`（薄弱分析子 Agent），缓存仍留在 service 层。
- `submit_answer` 成功后（review_agent 内部）会调 `_clear_caches(session_id)` 显式清除缓存。

---

### 3.4 底座层（core/ rag/ tools/ storage/）

#### core/ — 跨 Agent 共享的 LLM 底座

| 文件 | 内容 |
| --- | --- |
| `llm.py` | `get_llm()`（ChatOpenAI 单例，.env 驱动）+ `react_json()`（单工具 ReAct）+ `run_agent()`（V1.3 通用工具循环） |

V1.2 把原来 `graph/node.py` 里的 LLM 工厂和 ReAct 循环抽到这里，供所有 Agent 复用；V1.3 又加了 `run_agent()`——把「绑定一组工具 → LLM 自主决定调哪个 → 多轮往复 → 超轮收尾」的通用循环抽到共享底座，`supervisor` 和子 Agent 都能用。

#### rag/ — 混合检索

| 文件 | 内容 |
| --- | --- |
| `bm25.py` | 手写 BM25 索引（jieba 分词 + 逆文档频率） |
| `hybrid.py` | Hybrid Search（BM25 + 向量加权融合） |
| `vector.py` | 向量索引（bge-small-zh-v1.5, 512 维） |
| `retriever.py` | 会话检索器（按 session 缓存，review / import / qa 共用） |

#### tools/ — 工具箱

| 文件 | 内容 |
| --- | --- |
| `engine.py` | 搜索引擎（DuckDuckGo 优先 → Bing 后备） |
| `tools.py` | `search_web` tool 定义 + ReAct 循环 |

#### storage/ — 数据持久化

| 文件 | 内容 |
| --- | --- |
| `db.py` | SQLAlchemy engine + scoped_session + `init_db()` 迁移 |
| `schemas.py` | 6 个 ORM 模型 |

---

## 4. 数据流详解

### 4.1 导入流程

```
用户（前端）
  │
  │ POST /api/import 或 POST /api/import/file
  ▼
router.import_document() / import_file()
  │ 从 session_service 拿到 session_id
  │ 构造 title + content
  ▼
import_agent.import_content(session_id, content, title)
  │ 1. 原始文档存 documents 表
  │ 2. 构造 AgentState
  │ 3. 调 import_graph（planner 一次性图）
  ▼
agent/import_agent.py 的 import_graph（planner 节点）
  ├── 查已有知识点 → 构建 BM25 去重索引
  ├── 调 LLM 提取新知识点（Prompt：从内容中提取 title + content）
  ├── BM25 去重（相似度 > 0.8 跳过）
  ├── 批量存入 knowledge_points 表
  ├── 使检索缓存失效（rag/retriever.invalidate_retriever）
  └── 为每个新知识点生成向量 embedding（bge-small-zh）
       保存到 knowledge_points.embedding 字段
  │
  ▼
返回 knowledge_points 列表给前端
```

**关键细节：**

- LLM 提取是 JSON 格式的输出，通过 `JsonOutputParser` 解析
- BM25 去重是"二次过滤"——先让 LLM 判断不重复，再用 BM25 兜底
- Embedding 失败不阻塞导入，降级到纯 BM25 检索
- 检索缓存 `rag/retriever.py` 在导入后失效，下次复习时重建含新知识点的索引

### 4.2 复习流程

复习是目前最复杂的交互。一张 LangGraph 循环图 + interrupt 被拆到 4 个 HTTP 端点上分步调用：

```
┌─────────────────────────────────────────────────────────────────────┐
│  POST /api/review/start                                             │
│                                                                     │
│  review_agent.start(session_id)                                     │
│    → 生成 uuid thread_id                                            │
│    → review_graph.invoke(initial_state, {thread_id})                │
│                                                                     │
│  scheduler 节点：                                                    │
│    → 查该会话所有知识点                                              │
│    → 查 review_records 找薄弱 KPs（平均分 < 60）                    │
│    → 构建双车道出题队列                                              │
│       ├── 快车道：薄弱 KPs + 混合检索扩散关联 KPs                   │
│       └── 慢车道：剩余正常 KPs                                      │
│    → 取第一个知识点                                                  │
│                                                                     │
│  question_gen 节点：                                                 │
│    → 查 questions 表是否有缓存题目（use_count 最少的）              │
│    → 有缓存 → 直接返回，零 LLM 调用                                 │
│    → 无缓存 → 调 LLM 出题                                           │
│       ├── 混合检索相关知识点作为上下文                               │
│       ├── 查历史答题记录（Agent Memory）→ 针对薄弱点出题            │
│       └── ReAct：LLM 可自主搜索网络补全知识                         │
│    → 新题目存入 questions 表供后续复用                               │
│                                                                     │
│  wait_input 节点：                                                   │
│    → interrupt("请输入你的回答")                                     │
│    → 图暂停！返回题目给前端                                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  POST /api/review/{thread_id}/answer  {"answer": "..."}             │
│                                                                     │
│  review_agent.submit_answer(thread_id, answer)                      │
│    → Command(resume=answer) → review_graph.invoke()                 │
│    → 从暂停处继续执行                                                │
│                                                                     │
│  judge 节点：                                                        │
│    → 查该知识点的历史答题记录（Agent Memory）                       │
│    → 调 LLM 判分：score (0-100) + comment + strengths               │
│                  + weaknesses + missing_kps                          │
│    → 存入 review_records 表                                          │
│    → 清除 stats 缓存（_clear_caches）                                │
│                                                                     │
│  条件边 judge_should_continue：                                      │
│    ├── exit_review=True → END（复习结束）                            │
│    └── 否则 → 继续到 scheduler（下一题）                            │
│                                                                     │
│  循环：scheduler → question_gen → wait_input（再次暂停）            │
│  → 返回评价 + 下一题                                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  GET /api/review/{thread_id}/next                                   │
│  review_agent.get_next(thread_id)                                   │
│    → review_graph.get_state(config)  ← 不调 invoke                  │
│    → 从 MemorySaver 读当前 state                                    │
│    → 返回缓存的 current_question                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  POST /api/review/{thread_id}/exit                                  │
│  review_agent.exit(thread_id)                                       │
│    → Command(resume="__exit__")                                     │
│    → wait_input 收到 __exit__，设 exit_review=True                  │
│    → 条件边走 end → 图正常结束                                      │
│    → 清除 _active_reviews 中的记录                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**对话入口里的复习（V1.3）**：走 `POST /api/chat` 时，supervisor 持有每个 session 当前的 `review_thread_id`。用户说「开始复习」→ LLM 调 `start_review` 工具自动 `start()` 并把 thread 记到会话上；之后复习中答一句，LLM 就调 `submit_answer` 判分一次——所以对话页里复习是"出一道 → 答一道 → 判一道 → 接着聊"的自然循环。

### 4.3 分析流程

```
前端请求 GET /api/analyze（或对话里说「我哪里薄弱」）
  │
  router.analyze() / supervisor（analyze 意图）
  │ 从 session_service 拿 session_id
  ▼
  stats_service.analyze(session_id, llm_report=True)
  │
  ├─ 检查缓存 _analyze_cache[session_id]
  │  ├─ 缓存有效（TTL 内 + 无新答题记录）→ 直接返回
  │  └─ 缓存失效 → 继续
  │
  ▼
  agent/analyzer.py 的 analyze(session_id)   （由 stats_service 委托）
  │
  ├─ get_review_records(session_id)
  │    → SELECT * FROM review_records WHERE session_id=? ORDER BY created_at ASC
  │
  ├─ compute_kp_stats(records)
  │    → 按 kp_id 分组：平均分、最新分、趋势（上升/下降/稳定）、
  │      高频 weakness Top 3、高频 missing_kps Top 3
  │    → 按 avg_score 升序排列（最弱在前）
  │
  ├─ aggregate_global(records)
  │    → 全局：总答题数、全局平均分、weakness 词频 Top 10、
  │      missing_kps 词频 Top 10
  │
  └─ generate_llm_report(kp_stats, global_stats, kp_map)
       → 调 DeepSeek 生成自然语言分析报告（150-300 字）
       → 最弱领域 + 具体薄弱点 + 复习建议
  │
  ▼
  返回结果 → stats_service 写入缓存 _analyze_cache[session_id]
```

---

## 5. 数据库设计

### 5.1 ER 图

```
sessions
┌──────────────┐        documents
│ id (PK)      │──┐     ┌──────────────┐
│ name (UNIQUE)│  │     │ id (PK)      │
│ created_at   │  └─────│ session_id   │
└──────────────┘        │ title        │
       │                │ content      │
       │                │ created_at   │
       │                └──────────────┘
       │
       │                knowledge_points
       │                ┌──────────────┐
       ├─────────────── │ id (PK)      │
       │                │ session_id   │
       │                │ document_id  │
       │                │ title        │
       │                │ content      │
       │                │ embedding    ├── JSON（512维向量数组）
       │                │ created_at   │
       │                └──────────────┘
       │                      │
       │                      │ kp_id
       │                ┌─────┴─────────┐
       │                │               │
       │    questions    │   review_records
       │   ┌────────────┘   ┌──────────────┐
       │   │ id (PK)        │ id (PK)      │
       │   │ session_id     │ session_id   │
       │   │ kp_id          │ kp_id        │
       │   │ title          │ question     │
       │   │ question_text  │ user_answer  │
       │   │ use_count      │ ai_score     │
       │   │ created_at     │ ai_comment   │
       │   └────────────────│ ai_strengths ├── JSON
                            │ ai_weaknesses├── JSON
                            │ ai_missing_kps├── JSON
                            │ created_at   │
                            └──────────────┘
```

### 5.2 每张表详解

#### sessions

```sql
CREATE TABLE sessions (
    id         INT PRIMARY KEY AUTO_INCREMENT,
    name       VARCHAR(255) NOT NULL UNIQUE COMMENT '会话名，如 golang',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

- 数据量：通常个位数（一个人不会同时学几十个主题）
- 写频率：极低（创建/切换）
- `name` 唯一约束防止重名

#### documents

```sql
CREATE TABLE documents (
    id         INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL DEFAULT 0,
    title      VARCHAR(255) NOT NULL COMMENT '文件名或标题',
    content    TEXT NOT NULL COMMENT '原始内容',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

- 存储原始的导入内容，不修改
- Content 是 TEXT 类型，不是 LONGTEXT（单个文件通常不超过 65KB）
- session_id 索引由 ORM 层保证（无显式索引，数据量小时 OK）

#### knowledge_points

```sql
CREATE TABLE knowledge_points (
    id          INT PRIMARY KEY AUTO_INCREMENT,
    session_id  INT NOT NULL DEFAULT 0,
    document_id INT NOT NULL,
    title       VARCHAR(255) NOT NULL COMMENT '知识点名称',
    content     TEXT NOT NULL COMMENT '知识点内容（标准答案）',
    embedding   JSON NULL COMMENT '512维向量，Hybrid Search',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

- `embedding` 字段是 JSON 类型，存 512 个 float 的数组
- JSON 类型不支持索引，但数据量小时（几百个 KPs）全量线性扫描做向量检索也够快
- 未来如果数据量大，可以用 MySQL 8.4 的向量索引或外部向量数据库
- embedding 在导入子 Agent（planner 节点）中生成，失败时不阻塞

#### questions

```sql
CREATE TABLE questions (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    session_id    INT NOT NULL DEFAULT 0,
    kp_id         INT NOT NULL,
    title         VARCHAR(255) NOT NULL COMMENT '题目标题',
    question_text TEXT NOT NULL COMMENT '题目内容',
    use_count     INT DEFAULT 0 COMMENT '被用过几次（负载均衡）',
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

- 题目缓存表，减少 LLM 调用。同一个知识点可能会有多道题（不同次复习生成的不同题目）
- `use_count` 升序排列取第一条，优先用使用次数最少的题目
- 缓存没有淘汰机制——目前的设计是"题目越多越好"

#### review_records

```sql
CREATE TABLE review_records (
    id             INT PRIMARY KEY AUTO_INCREMENT,
    session_id     INT NOT NULL DEFAULT 0,
    kp_id          INT NOT NULL,
    question       TEXT NOT NULL COMMENT '当时出的题',
    user_answer    TEXT NOT NULL COMMENT '用户怎么答的',
    ai_score       INT COMMENT 'DeepSeek 给的分数（0-100）',
    ai_comment     TEXT COMMENT '评语',
    ai_strengths   JSON COMMENT '优点列表',
    ai_weaknesses  JSON COMMENT '不足列表',
    ai_missing_kps JSON COMMENT '缺失知识点列表',
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

- 核心数据表，所有分析的基础
- `ai_strengths` / `ai_weaknesses` / `ai_missing_kps` 是 JSON 数组
- 没有外键约束——`kp_id` 引用的知识点可能被删除，但答题记录保留
- 数据量预估：一次复习 20 题 → 20 条/次，正常使用几千条/年

### 5.3 为什么不建外键？

```python
# 没有 ForeignKey 约束
class KnowledgePoint(Base):
    session_id = Column(Integer, nullable=False, default=0)
    document_id = Column(Integer, nullable=False)
```

有意识的设计选择：

| 理由 | 说明 |
| --- | --- |
| **性能** | 外键在每次 insert/update 时做约束检查，对于高频写的 review_records 有影响 |
| **灵活性** | 不绑定外键意味着可以批量删、批量导，不会因为引用关系卡住 |
| **隔离** | session 间全隔离，外键跨 session 引用没有意义 |

代价是应用层要保证引用完整性——删除 session 时要手动删关联的所有表数据。

### 5.4 连接池

```python
engine = create_engine(DATABASE_URL, pool_size=5, pool_pre_ping=True)
# scoped_session → 每个线程独立 session，避免 pymysql 包序错乱
db = scoped_session(SessionLocal)
```

- `pool_size=5`：最多 5 个并发数据库连接
- `pool_pre_ping=True`：每次取连接前先 ping 一下，避免拿到的连接已断开
- `scoped_session`：每个线程独立的 session，Web 并发请求不会共用连接

---

## 6. 缓存策略

### 6.1 缓存总览

| 缓存 | 类型 | 位置 | 作用域 | 有效期 | 失效时机 |
| --- | --- | --- | --- | --- | --- |
| `_llm` | Python 对象 | `core/llm.py` | 进程全局 | 永久 | 服务重启 |
| `_retriever_caches` | HybridRetriever 实例 | `rag/retriever.py` | 按 session_id | 永久 | 导入新知识点 |
| `_analyze_cache` | 分析结果 dict | `service/stats_service.py` | 按 session_id | 15 分钟 | 新答题记录 |
| `_dashboard_cache` | Dashboard 数据 dict | `service/stats_service.py` | 按 session_id | 15 分钟 | 新答题记录 |
| `_active_reviews` | thread 映射 dict | `agent/review_agent.py` | 进程全局 | 复习期间 | 复习结束/服务重启 |
| `supervisor._convs` | 对话记忆 dict | `agent/supervisor.py` | 按 session_id | 对话期间 | 服务重启 |
| MemorySaver | AgentState 快照 | `agent/review_agent.py`（图编译时挂载） | 按 thread_id | 进程生命周期 | 服务重启 |

### 6.2 为什么不用 Redis？

当前项目规模（单机、单人/小团队使用）下：

- 不需要跨进程共享缓存
- 不需要缓存持久化（重建代价很低）
- 不需要缓存淘汰算法（数据量小，内存占用可忽略）

Python 进程内 dict 足够。如果未来需要多进程部署，`_active_reviews`、`supervisor._convs` 和 MemorySaver 需要换 Redis/PostgreSQL。

### 6.3 TTL 设计

```
分析缓存 TTL = 15 分钟
  原因：用户做一次复习、切个页面、再回来通常不会超过 15 分钟
  覆盖：即使 TTL 没到，有新答题记录也会失效（max_record_id 检查）
  手动清除：judge 判分后调 _clear_caches()

检索器缓存 TTL = 永久（显式清除）
  原因：知识点导入是低频操作，导入时手动清除
```

---

## 7. LangGraph 图详解

V1.2 之前图定义在 `graph/graph.py`、节点在 `graph/node.py`；V1.2 把图和节点并入各自的 Agent 文件，**结构和机制一个字没改**。下面的代码是机制示意，实际定义在：

- 复习循环图 → `agent/review_agent.py` 的 `_build_review_graph()`
- 导入一次性图 → `agent/import_agent.py`
- 共享状态 → `agent/state.py::AgentState`

### 7.1 import_graph（一次性图）

```python
import_builder = StateGraph(AgentState)
import_builder.add_node("planner", planner)
import_builder.set_entry_point("planner")
import_builder.add_edge("planner", END)
import_graph = import_builder.compile()
```

**只有一个节点**：planner。调 LLM 提取知识点，存库，结束。

没有条件边、没有循环、没有 checkpointer。跑完就完事。

### 7.2 review_graph（循环图）

```python
review_builder = StateGraph(AgentState)
review_builder.add_node("scheduler", scheduler)
review_builder.add_node("question_gen", question_gen)
review_builder.add_node("wait_input", wait_input)
review_builder.add_node("judge", judge)

review_builder.set_entry_point("scheduler")

# 条件边
review_builder.add_conditional_edges(
    "scheduler", scheduler_should_continue,
    {"question_gen": "question_gen", "end": END},
)
review_builder.add_edge("question_gen", "wait_input")
review_builder.add_edge("wait_input", "judge")
review_builder.add_conditional_edges(
    "judge", judge_should_continue,
    {"scheduler": "scheduler", "end": END},
)

review_graph = review_builder.compile(checkpointer=MemorySaver())
```

**四个节点 + 一个 checkpointer：**

| 节点 | 函数 | 核心逻辑 |
| --- | --- | --- |
| `scheduler` | 从 DB 取知识点 | 双车道出题队列：薄弱点优先 + 混合检索扩散 |
| `question_gen` | 生成题目 | 查缓存 → 没缓存就 LLM 出题（ReAct 可选搜索） |
| `wait_input` | 等待用户输入 | `interrupt()` 暂停图 |
| `judge` | 评判回答 | LLM 综合评分 + 存库 |

**循环机制：**

```
scheduler → question_gen → wait_input（interrupt，图暂停）
                             ↓
                          用户回答
                             ↓
                           judge
                             ├─ exit → END
                             └─ 继续 → scheduler（下一轮）
```

- `MemorySaver` 是 LangGraph 内置的内存 checkpointer，不依赖外部存储
- 每次 `interrupt` 前的状态自动保存，`Command(resume=...)` 可恢复

### 7.3 AgentState（图状态）

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: int
    
    # 导入字段
    raw_content: str
    document_id: int
    knowledge_points: list[dict]
    
    # 复习字段
    current_kp: dict          # {"id": 1, "title": "...", "content": "..."}
    current_question: str
    user_answer: str
    evaluation: dict           # {"score": 72, "comment": "...", "strengths": [...], ...}
    
    # 控制字段
    kp_index: int
    exit_review: bool
    review_queue: list         # [{"kp_id": 1, "reason": "...", "lane": "weak"}, ...]
    queue_pos: int
    review_reason: str
```

- `TypedDict` 是 LangGraph 的推荐状态类型，每个节点读/写特定字段
- `messages` 用了 `add_messages` reducer（保留历史消息），目前没有用到多轮对话，是预留
- 导入图和复习图共享同一个 `AgentState` 定义，但只用各自需要的字段
- V1.2 里定义在 `agent/state.py`，导入 / 复习 / 问答子 Agent 都从这里 import

### 7.4 条件边函数

```python
def scheduler_should_continue(state: AgentState) -> str:
    if state.get("exit_review"):
        return "end"
    return "question_gen"

def judge_should_continue(state: AgentState) -> str:
    if state.get("exit_review"):
        return "end"
    return "scheduler"
```

`exit_review` 是唯一的退出信号。`wait_input` 收到 `__exit__` 时设置它。

---

## 8. 混合检索（RAG）

### 8.1 架构图

```
HybridRetriever
  │
  ├── BM25Index (bm25.py)
  │     ├── build(docs) → jieba 分词 → IDF 计算
  │     └── search(query, top_k) → BM25 分数
  │
  └── VectorIndex (vector.py)
        ├── build(docs) → bge-small-zh-v1.5 encode → numpy 存向量
        ├── search(query, top_k) → 余弦相似度（归一化向量 dot product）
        └── _model → 单例 embedding 模型（lazy load）

  结果融合：
    BM25 分数 → minmax 归一化 → 0.3
    向量分数 → minmax 归一化 → 0.7
    → 加权求和 → 排序 → Top-K
```

检索器按会话缓存（`rag/retriever.py`），复习出题、问答、导入去重共用同一份索引；导入新知识点后整体失效重建。

### 8.2 BM25 实现

```python
# rag/bm25.py
from rank_bm25 import BM25Okapi
import jieba

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.docs = []
    
    def build(self, docs: list[str]):
        self.docs = docs
        tokenized = [list(jieba.cut(d)) for d in docs]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, top_k: int = 5):
        tokenized = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized)
        # 返回 [(idx, doc, score), ...] 按 score 降序
```

手写 BM25 而不是调用向量数据库，是为了面试可讲、部署零依赖。

### 8.3 向量检索实现

```python
# rag/vector.py
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _model

class VectorIndex:
    def __init__(self, dim=512):
        self.dim = dim
        self.vectors = None  # numpy array, shape=(n_docs, dim)
        self.docs = []
    
    def build(self, docs, embeddings=None):
        self.docs = docs
        if embeddings:
            self.vectors = np.array(embeddings)
        else:
            model = _get_model()
            self.vectors = model.encode(docs, normalize_embeddings=True)
    
    def search(self, query, top_k=5):
        model = _get_model()
        q_vec = model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.vectors, q_vec)  # 归一化向量 → dot = cosine
        # 返回 [(idx, doc, score), ...]
```

关键点：
- 向量已归一化（`normalize_embeddings=True`），所以 dot product = 余弦相似度
- embedding 存在 MySQL 的 JSON 字段里，构建时传入避免重复 encode
- `bge-small-zh-v1.5` 模型 33MB，第一次加载时下载（设 `HF_ENDPOINT` 绕过中国网络问题）

### 8.4 加权融合

```python
# rag/hybrid.py
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7

def search(self, query: str, top_k: int = 5):
    # 1. 扩大候选池（top_k * 2）
    bm25_results = self.bm25.search(query, top_k * 2)
    vec_results = self.vector.search(query, top_k * 2)
    
    # 2. minmax 归一化
    bm25_norm = self._minmax(bm25_results)
    vec_norm = self._minmax(vec_results)
    
    # 3. 加权融合
    fused = {}  # idx -> score
    for idx, doc, score in bm25_norm:
        fused[idx] = score * BM25_WEIGHT
    for idx, doc, score in vec_norm:
        fused[idx] = fused.get(idx, 0) + score * VECTOR_WEIGHT
    
    # 4. 排序取 Top-K
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, self.docs[idx], score) for idx, score in ranked]
```

- `top_k * 2`：扩大候选池，让融合后有足够选项
- `minmax` 归一化：把 BM25 分数和向量余弦分数拉到 [0, 1] 区间，确保加权公平
- 权重 0.7/0.3 由 `benchmark/run_rag.py` 权重扫描得出（召回优先选 0.7，排序优先选 0.3）

### 8.5 何时候降级

| 场景 | 行为 |
| --- | --- |
| embedding 模型未加载 | 切到纯 BM25 |
| BM25 没有文档 | 返回空结果 |
| 向量索引未构建 | 切到纯 BM25 |
| 某个 KP 没有 embedding | 只为有 embedding 的 KPs 做向量检索 |

---

## 9. 前端架构

### 9.1 技术选型

| 项目 | 选择 | 理由 |
|------|------|------|
| 构建 | Vite 6 | 零配置起步，HMR 开发体验，TS 原生支持 |
| 框架 | React 19 | 组件化，useReducer 状态机 |
| 语言 | TypeScript 5.7 | 全栈类型安全，编译期捕获错误 |
| 路由 | react-router-dom v7 | HashRouter，兼容 FastAPI 静态文件 serve |
| 样式 | 保留原 style.css | CSS 变量 + 亮色/暗色模式，576 行不动 |

### 9.2 开发模式与生产模式

**开发模式**（Vite 开发服务器 + FastAPI 后端）：

```text
npm run dev         → Vite :5173（HMR，代理 /api → :8000）
python main.py      → FastAPI :8000
```

Vite 通过 `vite.config.ts` 的 proxy 配置转发 `/api` 请求到 FastAPI，前端只管调 `/api/stats`。

**生产模式**（FastAPI serve 构建产物）：

```text
npm run build       → frontend/dist/
                    FastAPI serve dist/ → :8000
```

`backend/main.py` 优先挂载 `frontend/dist/`，不存在时 fallback 到 `frontend/`（兼容纯开发模式）。

### 9.3 文件结构

```
frontend/
├── package.json           # Vite + React 19 + TypeScript + react-router-dom
├── vite.config.ts         # /api proxy → localhost:8000
├── tsconfig.json          # TypeScript 配置
├── index.html             # <div id="root"><script type="module" src="/src/main.tsx">
├── style.css              # 保留原样（576行，含亮色/暗色）
└── src/
    ├── main.tsx           # ReactDOM.createRoot + HashRouter + 主题初始化
    ├── App.tsx            # Layout（Topbar + Sidebar + Content）+ 6 条页面路由
    ├── api.ts             # fetch 封装 + 全部 TypeScript 类型定义
    ├── hooks/
    │   ├── useSession.ts  # localStorage session_id 读写 + 初始化
    │   └── useReview.ts   # 复习状态机 useReducer（6 阶段）
    └── pages/
        ├── Dashboard.tsx  # 概览：统计卡片 + 快捷操作 + 入门指引
        ├── Chat.tsx       # 对话（V1.3）：LLM 大脑对话页
        ├── Sessions.tsx   # 会话列表 + 创建 + 切换
        ├── Import.tsx     # 导入：上传文件 / 粘贴文本（4 阶段视图）
        ├── Review.tsx     # 复习：6 阶段状态机渲染
        └── Analysis.tsx   # 薄弱分析：排行榜 + 词频 + LLM 报告
```

### 9.4 组件树

```
<HashRouter>
  <App>
    ├── <header.topbar>          ← Logo + SessionBadge
    ├── <nav.sidebar>            ← 6× NavLink + 主题切换
    └── <main.main-content>      ← Routes
          ├── #/ → <Dashboard />           // 概览
          ├── #/chat → <Chat />            // 对话（V1.3）
          ├── #/sessions → <Sessions />    // 会话管理
          ├── #/import → <Import />        // 导入资料
          ├── #/review → <ReviewPage />    // 复习
          ├── #/analysis → <Analysis />    // 薄弱分析
          └── 404 → 未找到页面
  </App>
</HashRouter>
```

- 使用 `HashRouter` 而非 `BrowserRouter`：所有路径以 `#/` 开头，兼容 FastAPI 的 `StaticFiles(html=True)` 无冲突。
- `NavLink` 自带 `className` 回调，当前页面自动高亮。

### 9.5 页面一览

| 页面 | 路由 | API 调用 | 说明 |
| --- | --- | --- | --- |
| 概览 | `#/` | `/api/stats`, `/api/sessions` | 5 个统计卡片 + 快捷操作 + 入门指引 |
| 对话 | `#/chat` | `POST /api/chat` | 自然语言交流入口：LLM 工具循环分发（V1.3） |
| 会话 | `#/sessions` | `/api/sessions` CRUD | 创建、切换当前会话 |
| 导入 | `#/import` | `/api/import`, `/api/import/file` | 上传文件 / 粘贴文本，4 阶段视图 |
| 复习 | `#/review` | `/api/review/start/answer/next/exit` | 6 阶段状态机，最复杂的交互 |
| 分析 | `#/analysis` | `/api/analyze` | 薄弱排行榜 + 高频词频 + LLM 报告 |

**对话页（Chat.tsx）**：自己维护一份本地消息列表 `msgs[]`，另有一份 `question` 状态表示"当前是否有一道待回答的题"。后端返回结构化结果（`type: question / review_result / analysis / answer / chat / imported`）后按 type 渲染成不同卡片——出题、判分卡、分析报告都能嵌在对话流里。

### 9.6 状态管理

**会话状态**（useSession hook）：
```typescript
// localStorage 读写，后端无状态
function useSession() {
  return { sessionId, switchSession, createSession, initSession };
}
```

**复习状态机**（useReview hook — useReducer）：

```
idle → loading → answering → submitting → evaluated → answering
  ↑                                  ↑         ↓
  └───────────────────────────── ended ←───────┘
```

| 阶段 | UI | 触发 |
|------|----|------|
| `idle` | 开始按钮 / 空状态 | 用户点"开始" |
| `loading` | "出题中..." | POST /api/review/start |
| `answering` | 题目 + 输入框 + 提交 | 用户提交回答 |
| `submitting` | "AI 判分中..." | POST /review/{id}/answer |
| `evaluated` | 评分 + 评语 + "下一题" | API 返回评价 |
| `ended` | "复习结束" 庆祝画面 | 退出复习 |

reducer 处理 9 种 action 类型（START / READY / SUBMIT / EVALUATED / NEXT / CONTINUE / END / RESET / ERROR），所有状态集中在 `ReviewState` 一个对象，对比原 `_lastReviewData` 全局变量方案，状态可预测、可测试。

> 对话页不走 useReview——它是"纯聊"形态：问答、出题、判分都由后端 supervisor 编排，前端只做消息气泡 + 卡片渲染。

### 9.7 API 层

```typescript
// api.ts — 自动带 session_id、类型泛型、错误统一处理
export async function api<T>(method: string, path: string, body?: unknown): Promise<T> {
  const sid = localStorage.getItem("sf-session-id");
  if (sid && path.startsWith("/api/")) {
    path += `${sep}session_id=${sid}`;
  }
  const res = await fetch(path, { method, headers, body: JSON.stringify(body) });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}
```

所有 API 端点都有 TypeScript 类型定义（Session, Stats, ReviewStartData, Evaluation, **ChatResult**（6 种响应联合）等）。

### 9.8 构建产物

```text
frontend/dist/
  index.html                 0.43 kB
  assets/index-*.css        10.60 kB  (style.css 构建后)
  assets/index-*.js        254.92 kB  (React + 组件代码，gzip 80 kB)
```

---

## 10. API 规范

### 10.1 端点一览

| 方法 | 路径 | 请求体 | 返回体 |
| --- | --- | --- | --- |
| GET | `/api/sessions` | — | `[{id, name, created_at}]` |
| POST | `/api/sessions` | `{name}` | `{id, name, created_at}` |
| POST | `/api/sessions/{id}/switch` | — | `{id, name}` |
| GET | `/api/sessions/current` | — | `{id, name}` |
| POST | `/api/chat` | `{message}` | `ChatResult`（chat/question/review_result/answer/analysis/imported 六选一） |
| POST | `/api/import` | `{content, title?}` | `{document_id, knowledge_points}` |
| POST | `/api/import/file` | multipart/file | `{document_id, knowledge_points}` |
| GET | `/api/knowledge-points` | — | `[{id, title, content, avg_score, review_count}]` |
| POST | `/api/review/start` | — | `{thread_id, question, kp_title, kp_content, review_reason}` |
| POST | `/api/review/{id}/answer` | `{answer}` | `{evaluation, exit}` |
| GET | `/api/review/{id}/next` | — | `{question, kp_title, kp_content, review_reason}` |
| POST | `/api/review/{id}/exit` | — | `{exit: true}` |
| GET | `/api/review/active` | — | `{active: [thread_id, ...]}` |
| GET | `/api/analyze` | `?no_llm=true` | `{kp_stats, global_stats, llm_report?}` |
| GET | `/api/stats` | — | `{session_name, kp_count, review_count, avg_score, doc_count, weak_kp_count}` |

### 10.2 错误处理

```python
# 业务异常 → ValueError → HTTP 400
# 资源不存在 → ValueError("不存在") → HTTP 404

@router.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    try:
        return session_service.create_session(req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
```

所有业务异常统一用 `ValueError` 抛出，router 层拦截转 HTTP 错误码。

### 10.3 CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 开发阶段允许所有
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

生产环境应限制 `allow_origins` 为实际部署域名。

---

## 11. 部署方案

### 11.1 当前部署

```python
# main.py — 开发/生产通用
import uvicorn
load_dotenv()

def main():
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
```

```bash
# 启动
python main.py

# 或开发模式（热重载）
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

前端由 FastAPI 挂载静态文件提供，不需要额外 Web 服务器。

### 11.2 生产部署建议

```nginx
# Nginx 反向代理示例
server {
    listen 80;
    server_name studyforge.example.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # 静态文件由 Nginx 直接 serve
    location /static/ {
        alias /path/to/frontend/;
        expires 7d;
    }
}
```

### 11.3 依赖清单

```text
Web 框架：       fastapi, uvicorn
AI/LLM：        openai, langchain-core, langchain-openai, langgraph
向量模型：       sentence-transformers
数据库：         sqlalchemy, pymysql
中文分词：       jieba, rank-bm25
搜索：           ddgs, requests
配置：           python-dotenv
```

`requirements.txt` 已包含全部依赖。

---

## 12. 与 V2 的关系

StudyForge 的演进路线是「点哪用哪 → 聊天就能学」。V1.1 已铺好对话入口，V1.2 已把代码长成多 Agent 结构，离 V2 只剩把"分发"升级成真正的主图。

### 12.1 V1.1 已做 — 自然语言交流入口

给 V1 加了一个"听得懂话"的皮层（见 [ROADMAP](ROADMAP.md)）：

```
前端 Chat 页 ── POST /api/chat ──► supervisor（对话分发，规则版）
                                        ├── nlu/intent.py   意图识别（8 类，规则优先）
                                        ├── review_agent    复习（复用，不变）
                                        ├── stats_service   分析（复用，不变）
                                        ├── import_agent    导入（复用，不变）
                                        └── qa_agent        轻量问答（检索 + grounded 回答）
```

- V1.1 新增：`nlu/`、前端 `pages/Chat.tsx`、`POST /api/chat`、轻量问答；现有功能端点 / 复习图 / 存储全部复用，V1 核心零改动
- 对话记忆态在 supervisor 的 `_convs`（同 `_active_reviews`，服务重启即失）

### 12.2 V1.2 已做 — 架构重构 · 子 Agent 化

把"对话分发"从散在 service 里的规则版，升级成**真代码结构**上的多 Agent：

- 4 个专职子 Agent（`review_agent` / `import_agent` / `qa_agent` / `analyzer`）独立成模块、接口可单独调用
- 主 Agent `Supervisor`（`agent/supervisor.py`）：意图识别 → 分发到子 Agent
- 拆掉上帝模块 `graph/node.py`：LLM 工厂/ReAct → `core/llm.py`，检索缓存 → `rag/retriever.py`；`graph/` 目录删除
- 复习图（LangGraph + interrupt + MemorySaver）**原样保留**，只是迁进 `review_agent.py`；行为零回归
- service 层收敛为 session / stats 无状态读服务；API 端点路径全不变，前端零改动

一句话：V1.2 让"几个 Agent、各自干嘛"从纸面叙事变成真代码结构。但 supervisor 目前仍是**规则 if/else 分发**，还不是真正的一张 LangGraph 主图（interrupt 上移到主图边界，是 V2 核心）。见 [issue #1](https://github.com/qi-wutu/StudyForge/issues/1)（已关闭）。

### 12.3 V1.3 已做 — LLM 大脑 · 通用 Agent 化

V1.3 把 supervisor 从「规则路由器」升级成 **LLM 工具调用循环**，是 V1 现在的对话大脑（见 [ROADMAP](ROADMAP.md)）：

- **`core/llm.py::run_agent()`** — 通用工具循环：LLM 绑定一组工具，自主决定调哪个、可多轮往复、超轮强制收尾、工具异常不崩
- **`agent/tools.py::build_tools()`** — 把 4 个子 Agent + 通用搜索能力打包成 **7 个可调用工具**（开始复习 / 作答判分 / 退出复习 / 问答 / 薄弱分析 / 导入 / 通用搜索）
- **`Supervisor.chat()` 重写** — 确定性安全快路（退出/下一题/导入前缀，防误判 & 不花 LLM）→ 把对话历史真正喂给 LLM → 跑工具循环 → 由工具结果决定返回卡片
- **`nlu/` 退为确定性安全快路** — 8 类正则完整保留（`classify_intent` 测试锁定），只额外暴露 `fast_path_intent()` 拦最确定的小撮命令
- **前端卡片加 LLM 前言气泡** — 出题/判分/分析等卡片前先出一句自然语言，更有聊天手感
- 话题参数（"复习数据库索引"）先识别、暂不落地；单测 13 意图 + 16 工具层/循环 = 29 条

### 12.4 V2 — 还没做

| 版本 | 内容 | 状态 |
| --- | --- | --- |
| **V2** | LangGraph supervisor 主图：interrupt 上移到主图边界，多子图显式编排 | ⏳ 规划（见 [ROADMAP](ROADMAP.md)） |

V1（当前） vs V2（规划）：

| 维度 | V1.3（当前） | V2（规划） |
| --- | --- | --- |
| 路由 | LLM 工具循环动态决定（`run_agent` + `build_tools`） | LangGraph 主图按意图分发 + interrupt |
| 工具 | 7 个（子 Agent 能力 + 通用搜索） | 搜索 + 计算器 + 文档查询 + ... |
| 图 | 两张独立子图（review/import） | 一个主图调度多个子图 |
| Agent | 主 Agent（LLM 调度）+ 子 Agent（工具化） | 真正一张 LangGraph 主图调度 |

**V1.3 的架构设计为 V2 预留了什么：**

- 每个子 Agent 已是独立模块 + turn 接口，V2 主图可以直接把它们挂成节点 / Tool
- `run_agent()` 的多轮工具循环已把「决策 → 执行」能力验证跑通，V2 只需把它显式画成图上节点
- `tools/` 目录可以继续收纳所有工具（搜索、计算、文档查询等）
- `nlu/` 的快路 / `build_tools` 的工具化在 V2 主图里可直接复用
- Session 隔离机制在多 Agent 场景下同样适用

---

## 附录 A：关键设计决策索引

详细决策分析见 `decisions/`：

| 文档 | 核心结论 |
| --- | --- |
| [BM25 手写教程](decisions/2026-07-13-BM25手写教程.md) | jieba 分词 + rank-bm25 库，40 行实现 |
| [为什么用 BM25 做 RAG](decisions/2026-07-13-为什么用BM25做RAG.md) | 关键词密度高，BM25 比向量更准；零依赖部署 |
| [知识点去重策略](decisions/2026-07-13-知识点去重策略.md) | LLM 初步去重 + BM25 二次过滤 > 0.8 跳过 |
| [双车道调度策略](decisions/2026-07-14-双车道调度策略.md) | 2:1 交错，防饿死 |
| [RAG 升级混合检索](decisions/2026-07-24-RAG升级混合检索.md) | BM25 0.7 + 向量 0.3 加权，bge-small-zh |
| [数据库并发连接策略](decisions/2026-07-24-数据库并发连接策略.md) | pool_size=5, pool_pre_ping, scoped_session |

---

## 附录 B：代码量统计（近似值）

```
模块          文件数     代码行数    备注
backend/        3        150       FastAPI 路由 + 入口
agent/          7       1000       主 Agent（LLM 大脑）+ tools + 4 子 Agent + state（含图/节点）
nlu/            1         90       规则意图识别（8 类 intent）+ fast_path_intent 快路
core/           1         70       共享 LLM + ReAct + run_agent 工具循环
service/        2        250       无状态读服务 + 缓存
rag/            4        200       BM25 + Hybrid + Vector + Retriever
tools/          2        100       搜索 + ReAct
storage/        2        120       数据库
frontend/       ~14      1400      React 19 + TypeScript + Vite（含 dist/）
其他            4        150       main.py + config.py + .env.example + CLI
合计            ~39      ~3400
```
