# StudyForge 架构文档

> 版本：2.0（Web 版）  
> 最后更新：2026-07-28

---

## 目录

1. [项目定位](#1-项目定位)
2. [整体架构概览](#2-整体架构概览)
3. [三层架构详解](#3-三层架构详解)
   - 3.1 [Controller 层（backend/）](#31-controller-层-backend)
   - 3.2 [Service 层（service/）](#32-service-层-service)
   - 3.3 [Data 层（graph/ + rag/ + storage/ + tools/）](#33-data-层)
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
┌─────────────────────────────────────────────────────────────────────┐
│                       用户浏览器                                    │
│  React 19 + TypeScript + Vite                                      │
│  单页应用（HashRouter），组件化状态管理                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP / JSON
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI 服务（Python）                                              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Controller 层 ── backend/router.py                             │  │
│  │  只做：解析 HTTP 参数 → 调 service → 返回 JSON                  │  │
│  │  不做：调 LangGraph、查数据库、管业务状态                         │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                             │ 调用                                   │
│  ┌──────────────────────────▼────────────────────────────────────┐  │
│  │  Service 层 ── service/                                       │  │
│  │                                                                │  │
│  │  session_service  import_service  review_service  stats_service│  │
│  │  封装：LangGraph 驱动、状态管理、缓存、数据分析                  │  │
│  └───────┬──────────┬──────────┬──────────┬──────────────────────┘  │
│          │          │          │          │                         │
│          ▼          ▼          ▼          ▼                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Data 层                                                      │  │
│  │                                                                │  │
│  │  graph/ ── LangGraph 节点 + 图定义 + 分析器                    │  │
│  │  rag/   ── BM25 + 向量混合检索                                 │  │
│  │  tools/ ── 搜索工具（DuckDuckGo + Bing） + ReAct 循环           │  │
│  │  storage/ ── MySQL + SQLAlchemy ORM                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
           │                   │
           ▼                   ▼
     ┌──────────┐      ┌──────────────┐
     │  MySQL   │      │  DeepSeek    │
     │  6 张表  │      │  API         │
     └──────────┘      └──────────────┘
```

### 设计原则

1. **三层分离** — Controller 不碰业务，Service 不碰 HTTP，Data 层专注底层能力
2. **图内/图外分离** — LangGraph 节点内只关注"该做什么事"，不关心 HTTP、不关心存储
3. **隔离优先** — 会话（Session）间数据全隔离，没有跨会话引用
4. **降级设计** — 向量检索失败 → 降级 BM25；搜索 API 失败 → 跳过搜索；embedding 失败 → 不阻塞导入
5. **缓存不是必须的** — 所有缓存都是内存级，服务重启后重建，不会丢业务数据

---

## 3. 三层架构详解

### 3.1 Controller 层（backend/）

#### backend/main.py

```python
# 职责：创建 FastAPI 实例、挂中间件、挂路由、挂静态文件
app = FastAPI()
app.add_middleware(CORSMiddleware, ...)  # 跨域
app.include_router(router)               # API 路由
app.mount("/", StaticFiles(...))         # 前端静态文件
```

只有 39 行，不含任何业务逻辑。启动事件里调 `init_db()` 做建表和迁移。

#### backend/router.py

```python
# 每个端点只做三件事：
@router.get("/api/stats")
def get_stats(session_id: Optional[int] = Query(None)):
    sid = session_service.resolve_session_id(session_id)  # 1. 拿 session_id
    return stats_service.get_dashboard_stats(sid)          # 2. 调 service
                                                           # 3. 返回（隐式 JSON）
```

- 不直接调 `db.query()`、不调 `graph.invoke()`
- 不维护任何状态（`_active_reviews` 已经在 service 层）
- 异常统一转 HTTP 错误码
- session_id 由前端 localStorage 持有，通过 `?session_id=X` 传过来

**为什么不让 router 直接调 graph？**  
为了可测试性。如果 router 里嵌了 LangGraph 调用，写单元测试就得 mock HTTP 请求。现在只需要测 service 方法就行。

---

### 3.2 Service 层（service/）

| 文件 | 职责 | 关键函数 |
| --- | --- | --- |
| `session_service.py` | 会话的 CRUD、ID 解析 | `resolve_session_id()`, `create_session()`, `list_sessions()` |
| `import_service.py` | 资料导入 → 调 import_graph | `import_content(session_id, content, title)` |
| `review_service.py` | 复习图的启动/恢复/退出 | `start()`, `submit_answer()`, `get_next()`, `exit()` |
| `stats_service.py` | 统计查询 + 薄弱分析 + 缓存 | `get_dashboard_stats()`, `list_knowledge_points()`, `analyze()` |

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

#### review_service.py 详解

```python
class ReviewService:
    """复习服务 — 管理每个 thread 的复习会话
    
    复习图是一个循环图，但被拆到多个 HTTP 端点里分步调用。
    
    start():
        → 生成 thread_id (uuid)
        → review_graph.invoke(state, config)
        → 图跑到 wait_input 的 interrupt() 暂停
        → 返回题目
    
    submit_answer(thread_id, answer):
        → Command(resume=answer)
        → 图被唤醒，继续：judge → scheduler → question_gen → wait_input
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

- **单例**：`review_service = ReviewService()` 在模块级别实例化，所有请求共用。因为要维护 `_active_reviews` 字典。
- **thread_id**：每次 `start()` 生成一个 uuid，作为复习会话的唯一标识。存在内存里，服务重启后丢失。
- **不持久化活跃复习**：如果有未完成的复习，服务重启后用户需要重新 start。在设计上接受了这个限制——一次复习通常只需要几分钟。

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

分析结果缓存 15 分钟。如果期间有新的答题记录（`max_record_id` 变化），立即重新计算。`submit_answer` 成功后会调 `_clear_caches(session_id)` 显式清除缓存。

---

### 3.3 Data 层

#### graph/ — LangGraph 核心

| 文件 | 内容 |
| --- | --- |
| `state.py` | `AgentState` TypedDict — 图中所有节点共享的数据结构 |
| `node.py` | 5 个节点函数 + 条件边函数 + 辅助函数（LLM、检索、Memory） |
| `graph.py` | `StateGraph` 编译 — import_graph（一次性）+ review_graph（循环） |
| `analyzer.py` | 薄弱分析：聚合统计 + 去重 + LLM 报告生成 |

#### rag/ — 混合检索

| 文件 | 内容 |
| --- | --- |
| `bm25.py` | 手写 BM25 索引（jieba 分词 + 逆文档频率） |
| `hybrid.py` | Hybrid Search（BM25 0.7 + 向量 0.3 加权融合） |
| `vector.py` | 向量索引（bge-small-zh-v1.5, 512 维） |

#### tools/ — 工具箱

| 文件 | 内容 |
| --- | --- |
| `engine.py` | 搜索引擎（DuckDuckGo 优先 → Bing 后备） |
| `tools.py` | `search_web` tool 定义 + `react_call()` ReAct 循环 |

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
import_service.import_content(session_id, content, title)
  │ 1. 原始文档存 documents 表
  │ 2. 构造 AgentState
  │ 3. import_graph.invoke(state)
  ▼
graph/graph.py 的 import_graph（一次性图）
  │
  └── planner 节点（graph/node.py）
        ├── 查已有知识点 → 构建 BM25 去重索引
        ├── 调 LLM 提取新知识点（Prompt：从内容中提取 title + content）
        ├── BM25 去重（相似度 > 0.8 跳过）
        ├── 批量存入 knowledge_points 表
        ├── 清空检索缓存 _retriever_caches
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
- `_retriever_caches` 在导入后清除，下次复习时重建含新知识点的索引

### 4.2 复习流程

```
┌─────────────────────────────────────────────────────────────────┐
│  POST /api/review/start                                         │
│                                                                 │
│  review_service.start(session_id)                               │
│    → 生成 uuid thread_id                                        │
│    → review_graph.invoke(initial_state, {thread_id})            │
│                                                                 │
│  scheduler 节点：                                                │
│    → 查该会话所有知识点                                          │
│    → 查 review_records 找薄弱 KPs（平均分 < 60）                │
│    → 构建双车道出题队列                                          │
│       ├── 快车道：薄弱 KPs + 混合检索扩散关联 KPs               │
│       └── 慢车道：剩余正常 KPs                                  │
│    → 取第一个知识点                                              │
│                                                                 │
│  question_gen 节点：                                             │
│    → 查 questions 表是否有缓存题目（use_count 最少的）           │
│    → 有缓存 → 直接返回，零 LLM 调用                              │
│    → 无缓存 → 调 LLM 出题                                       │
│       ├── 混合检索相关知识点作为上下文                            │
│       ├── 查历史答题记录（Memory）→ 针对薄弱点出题               │
│       └── ReAct：LLM 可自主搜索网络补全知识                      │
│    → 新题目存入 questions 表供后续复用                            │
│                                                                 │
│  wait_input 节点：                                               │
│    → interrupt("请输入你的回答")                                 │
│    → 图暂停！返回题目给前端                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  POST /api/review/{thread_id}/answer  {"answer": "..."}         │
│                                                                 │
│  review_service.submit_answer(thread_id, answer)                │
│    → Command(resume=answer) → review_graph.invoke()             │
│    → 从暂停处继续执行                                            │
│                                                                 │
│  judge 节点：                                                    │
│    → 查该知识点的历史答题记录（Memory）                          │
│    → 调 LLM 判分：score (0-100) + comment + strengths          │
│                  + weaknesses + missing_kps                     │
│    → 存入 review_records 表                                     │
│    → 清除 stats_cache（缓存失效）                                │
│                                                                 │
│  条件边 judge_should_continue：                                  │
│    ├── exit_review=True → END（复习结束）                        │
│    └── 否则 → 继续到 scheduler（下一题）                        │
│                                                                 │
│  循环：scheduler → question_gen → wait_input（再次暂停）        │
│  → 返回评价 + 下一题                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  GET /api/review/{thread_id}/next                               │
│  review_service.get_next(thread_id)                             │
│    → review_graph.get_state(config)  ← 不调 invoke             │
│    → 从 MemorySaver 读当前 state                                │
│    → 返回缓存的 current_question                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  POST /api/review/{thread_id}/exit                              │
│  review_service.exit(thread_id)                                 │
│    → Command(resume="__exit__")                                 │
│    → wait_input 收到 __exit__，设 exit_review=True              │
│    → 条件边走 end → 图正常结束                                  │
│    → 清除 _active_reviews 中的记录                              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 分析流程

```
前端请求 GET /api/analyze
  │
  router.analyze()
  │ 从 session_service 拿 session_id
  ▼
  stats_service.analyze(session_id, llm_report=True)
  │
  ├─ 检查缓存 _analyze_cache[session_id]
  │  ├─ 缓存有效（TTL 内 + 无新答题记录）→ 直接返回
  │  └─ 缓存失效 → 继续
  │
  ▼
  graph/analyzer.py 的 analyze(session_id)
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
  返回结果 + 写入缓存 _analyze_cache[session_id]
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
- embedding 在 planner 节点中生成，失败时不阻塞

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
| `_llm` | Python 对象 | `graph/node.py` | 进程全局 | 永久 | 服务重启 |
| `_retriever_caches` | HybridRetriever 实例 | `graph/node.py` | 按 session_id | 永久 | 导入新知识点 |
| `_analyze_cache` | 分析结果 dict | `service/stats_service.py` | 按 session_id | 15 分钟 | 新答题记录 |
| `_dashboard_cache` | Dashboard 数据 dict | `service/stats_service.py` | 按 session_id | 15 分钟 | 新答题记录 |
| `_active_reviews` | thread 映射 dict | `service/review_service.py` | 进程全局 | 复习期间 | 复习结束/服务重启 |
| MemorySaver | AgentState 快照 | `graph/graph.py` | 按 thread_id | 进程生命周期 | 服务重启 |

### 6.2 为什么不用 Redis？

当前项目规模（单机、单人/小团队使用）下：

- 不需要跨进程共享缓存
- 不需要缓存持久化（重建代价很低）
- 不需要缓存淘汰算法（数据量小，内存占用可忽略）

Python 进程内 dict 足够。如果未来需要多进程部署，`_active_reviews` 和 MemorySaver 需要换 Redis/PostgreSQL。

### 6.3 TTL 设计

```
分析缓存 TTL = 15 分钟
  原因：用户做一次复习、切个页面、再回来通常不会超过 15 分钟
  覆盖：即使 TTL 没到，有新答题记录也会失效（max_record_id 检查）
  手动清除：submit_answer 成功后调 _clear_caches()

检索器缓存 TTL = 永久（显式清除）
  原因：知识点导入是低频操作，导入时手动清除
```

---

## 7. LangGraph 图详解

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
- `messages` 用了 `add_messages` reducer（保留历史消息），但目前没有用到多轮对话，是预留
- 导入图和复习图共享同一个 `AgentState` 定义，但只用各自需要的字段

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
    ├── App.tsx            # Layout（Topbar + Sidebar + Content）+ Routes
    ├── api.ts             # fetch 封装 + 全部 TypeScript 类型定义
    ├── hooks/
    │   ├── useSession.ts  # localStorage session_id 读写 + 初始化
    │   └── useReview.ts   # 复习状态机 useReducer（6 阶段）
    └── pages/
        ├── Dashboard.tsx  # 概览：统计卡片 + 快捷操作 + 入门指引
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
    ├── <nav.sidebar>            ← 5× NavLink + 主题切换
    └── <main.main-content>      ← Routes
          ├── #/ → <Dashboard />           // 概览
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
| 会话 | `#/sessions` | `/api/sessions` CRUD | 创建、切换当前会话 |
| 导入 | `#/import` | `/api/import`, `/api/import/file` | 上传文件 / 粘贴文本，4 阶段视图 |
| 复习 | `#/review` | `/api/review/start/answer/next/exit` | 6 阶段状态机，最复杂的交互 |
| 分析 | `#/analysis` | `/api/analyze` | 薄弱排行榜 + 高频词频 + LLM 报告 |

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

所有 15 个 API 端点都有 TypeScript 类型定义（Session, Stats, ReviewStartData, Evaluation 等）。

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

项目有 V2 规划，核心变化：

| 维度 | V1（当前） | V2（规划） |
| --- | --- | --- |
| 路由 | 简单 HTTP -> graph | 意图识别节点 + 多图编排 |
| 工具 | 仅有搜索 | 搜索 + 计算器 + 文档查询 + ... |
| 图 | 两个独立图 | 一个主图调度多个子图 |
| Agent | 单 Agent | Multi-Agent 协作 |

**V1 的架构设计为 V2 预留了什么：**

- `tools/` 目录可以收纳所有工具（搜索、计算、文档查询等）
- `service/` 层可扩展，V2 的意图识别节点就是一个新的 service
- Session 隔离机制在多 Agent 场景下同样适用
- 三层架构让 V2 新增功能时只需要加/改 service 层

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
service/        4        200       业务逻辑层
graph/          4        500       LangGraph 节点 + 图 + 分析器
rag/            3        130       BM25 + Hybrid + Vector
tools/          2        100       搜索 + ReAct
storage/        2        120       数据库
frontend/       ~12      1200      React 19 + TypeScript + Vite（含 dist/）
其他            4        150       main.py + config.py + .env.example + CLI
合计            ~35      ~2600
```
