# StudyForge — AI 自适应复习系统

导入学习资料 → 自动提取知识点 → AI 出题 → 自由回答 → AI 评分 + 指出缺失 → 薄弱分析 → 针对性回炉

不是刷题工具，是 **AI 面试官 + 学习教练**。你像聊天一样回答，它像老师一样打分、写评语、告诉你哪里弱、给你找资料补。

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 构建前端（开发模式可跳过，FastAPI 会 fallback 到 frontend/）
cd frontend && npm run build && cd ..

# 配置 .env
cp .env.example .env
# 编辑 .env 填入 MYSQL_PASSWORD 和 LLM_API_KEY

# 启动 Web 服务（自动建表 + 迁移）
python main.py
```

打开 `http://localhost:8000` 即可使用。

### 前置要求

- Python 3.10+
- MySQL（studyforge 数据库需提前创建）
- DeepSeek API Key
- Node.js 18+（前端构建）

---

## 界面截图

| 页面 | 功能 |
| --- | --- |
| **概览** | Dashboard：会话名称、知识点总数、答题记录、平均分、薄弱数 |
| **会话管理** | 创建/切换会话，数据隔离互不干扰 |
| **导入资料** | 上传 `.md`/`.txt` 文件或粘贴文本，AI 自动提取知识点 |
| **复习** | AI 出题 → 输入回答 → AI 评分 + 评语 + 优缺点 + 缺失知识点 |
| **薄弱分析** | 薄弱排行榜 + 高频缺失 + AI 分析报告 |

---

## 项目结构

```
StudyForge/
│
├── main.py                 # 服务入口：uvicorn.run("backend.main:app")
├── config.py               # 配置（MySQL / DeepSeek / .env）
│
├── backend/                # HTTP 层（Controller）
│   ├── main.py             # FastAPI app 定义 + CORS + 静态文件挂载
│   └── router.py           # API 路由（只做 HTTP 解析，不碰业务）
│
├── service/                # 业务逻辑层（Service）
│   ├── session_service.py  # 会话 CRUD + session_id 解析
│   ├── import_service.py   # 资料导入：调 import_graph 提取知识点
│   ├── review_service.py   # 复习管理：驱动 review_graph 暂停/恢复
│   └── stats_service.py    # 统计分析：Dashboard、知识点列表、薄弱分析
│
├── graph/                  # LangGraph 核心
│   ├── state.py            # Agent 状态定义
│   ├── node.py             # 节点函数（调度/出题/判题/提取知识点）
│   ├── graph.py            # 图定义（节点连线 + 条件边）
│   └── analyzer.py         # 薄弱分析（统计 + LLM 报告）
│
├── tools/                  # 工具箱
│   ├── engine.py           # 搜索引擎（DuckDuckGo → Bing 双引擎后备）
│   └── tools.py            # LangChain Tool + ReAct 循环
│
├── rag/                    # 混合检索（BM25 + 向量）
│   ├── bm25.py             # BM25 索引（手写，jieba 分词）
│   ├── hybrid.py           # Hybrid Search（BM25 0.7 + 向量 0.3 加权融合）
│   └── vector.py           # 向量索引（bge-small-zh-v1.5, 512维）
│
├── storage/                # MySQL 存储
│   ├── db.py               # 引擎 + scoped_session + 迁移
│   └── schemas.py          # ORM 模型（6 张表）
│
├── frontend/               # Web 前端（React 19 + TypeScript + Vite）
│   ├── package.json        # 依赖：react 19, react-router-dom v7, vite 6
│   ├── vite.config.ts      # /api proxy → localhost:8000
│   ├── tsconfig.json       # TypeScript 配置
│   ├── index.html          # 入口：<div id="root">
│   ├── style.css           # 样式（576行，亮色/暗色模式，保留原样）
│   ├── src/
│   │   ├── main.tsx        # ReactDOM.createRoot + HashRouter + 主题初始化
│   │   ├── App.tsx         # Layout（Topbar + Sidebar + Content）+ 5 条路由
│   │   ├── api.ts          # fetch 封装 + 全部 TypeScript 类型定义
│   │   ├── hooks/
│   │   │   ├── useSession.ts  # localStorage session_id 管理
│   │   │   └── useReview.ts   # 复习状态机（useReducer，6 阶段）
│   │   └── pages/
│   │       ├── Dashboard.tsx  # 概览页：统计卡片 + 快捷操作
│   │       ├── Sessions.tsx   # 会话管理：创建 / 切换
│   │       ├── Import.tsx     # 资料导入：上传文件 / 粘贴文本
│   │       ├── Review.tsx     # 复习：6 阶段状态机渲染
│   │       └── Analysis.tsx   # 薄弱分析：排行榜 + 词频 + LLM 报告
│   └── dist/               # 构建产物（npm run build）
│
├── benchmark/              # Benchmark 评测套件
│   ├── run.py              # 评测入口
│   ├── cases/              # 5 大主题 93 道主观题
│   └── README.md           # 评测三维度说明
│
└── docs/                    # 公开文档
    ├── ROADMAP.md           # 版本路线（V1 现状 + V2 规划）
    ├── ARCHITECTURE.md      # 整体架构
    └── decisions/           # 设计决策记录（6 篇）

---

## 架构

### 三层架构

```
┌───────────────────────────────────────────────┐
│  backend/router.py          ← HTTP 层         │
│  只做：解析请求参数 + 返回 JSON                   │
│  不碰：LangGraph、DB、业务逻辑                    │
└──────────────────────┬────────────────────────┘
                       ↓ 调用
┌───────────────────────────────────────────────┐
│  service/*                    ← 业务逻辑层      │
│  session_service / import_service              │
│  review_service / stats_service                │
│  封装：LangGraph 图驱动、状态管理、数据分析        │
└──────────────────────┬────────────────────────┘
                       ↓ 调用
┌───────────────────────────────────────────────┐
│  graph/    │  rag/     │  storage/            │
│  tools/    │           │                      │
│  ← LangGraph 核心                               │
│  ← 混合检索 (BM25+向量)                         │
│  ← MySQL + SQLAlchemy ORM                      │
└───────────────────────────────────────────────┘
```

### 复习流程

```text
POST /review/start
    → 启动 review_graph，跑到 wait_input（interrupt，图暂停）
    → 返回第一题

POST /review/{thread_id}/answer  {answer}
    → Command(resume) 唤醒图
    → judge 判分 → question_gen 下一题 → wait_input 再暂停
    → 返回评价 + 下一题

GET  /review/{thread_id}/next
    → 不跑图，读 checkpointer 缓存的当前状态

POST /review/{thread_id}/exit
    → Command(resume="__exit__") 让图正常结束
```

### 数据表

| 表 | 用途 | 关键字段 |
| --- | --- | --- |
| `sessions` | 会话隔离（多主题互不干扰） | name |
| `documents` | 用户导入的原始资料 | title, content, session_id |
| `knowledge_points` | 提取的知识点（出题素材） | title, content, session_id, document_id, embedding(JSON) |
| `questions` | 预生成题目缓存 | kp_id, question_text, use_count |
| `review_records` | 答题记录（分析的原料） | kp_id, ai_score, ai_strengths/weaknesses/missing_kps |

### 缓存策略

| 缓存 | 位置 | 有效期 | 失效条件 |
| --- | --- | --- | --- |
| `_llm` 实例 | 进程全局变量 | 永久 | 服务重启 |
| 检索器 `_retriever_caches` | 进程全局变量 | 永久 | 导入新知识点时清除 |
| 分析报告缓存 | 进程全局 dict | 15 分钟 | 有新答题记录时清除 |
| LangGraph checkpointer | 内存 MemorySaver | 服务运行期间 | 服务重启后丢失 |

---

## 六大核心特性

### 1. 双车道智能调度

从 `review_records` 找出平均分 < 60 的薄弱 KPs，以它们为"种子"：
- **快车道**：薄弱 KPs 本身 + BM25 + 向量混合检索扩散找到的关联 KPs
- **慢车道**：剩余正常 KPs
- **合并规则**：2 快 + 1 慢交错，保证即使薄弱点再多，也有 1/3 出题来自正常范围

### 2. 混合检索（BM25 + 向量）

BM25 和向量检索加权融合，平衡关键词命中与语义相似度：
- BM25 权重 0.7，向量权重 0.3（benchmark/run_rag.py 扫描，召回优先）
- 本地 embedding 模型 `bge-small-zh-v1.5`（33MB，512 维）
- 一方失败自动降级到纯 BM25 或纯向量

### 3. 轻量 ReAct（节点内循环）

`question_gen` 节点内部嵌入 ReAct 循环——LLM 绑定 `search_web` 工具，可自主决定搜索网络获取最新资料。不依赖 LangGraph 的 ToolNode，不改变图结构。`bind_tools` 标准协议，支持 DeepSeek / OpenAI / Claude。

### 4. Agent Memory

每次出题和判分时，自动注入该知识点的历史答题记录——发现"漏了什么"、"进步了没有"。不新增表、不新增依赖，从 `review_records` 查。

### 5. 网络搜索（双引擎后备）

```
tools/engine.py
  web_search("Go GMP work stealing 详解")
    ├── DuckDuckGo（国际优先）→ 失败静默切到
    └── Bing（正则解析 HTML）
```

无需 API Key，免费使用。

### 6. Benchmark 评测套件

5 大主题（数据结构/操作系统/网络/数据库/Golang）、93 道主观题，三位一体评测：

| 维度 | 指标 | 说明 |
|---|---|---|
| 一致性 | 标准差 | 同一题跑多次，分数稳不稳 |
| 准确性 | MAE | AI 评分跟人工标注差多远 |
| 统一度 | Spearman + 分层 | 评分尺子是不是一把 |

详情见 [Benchmark 文档](benchmark/README.md)。

---

## API 端点一览

| 方法 | 路径 | 说明 | session_id |
|---|---|---|---|
| GET | `/api/sessions` | 会话列表 | — |
| POST | `/api/sessions` | 创建会话 | — |
| POST | `/api/sessions/{id}/switch` | 切换（查会话信息） | - |
| GET | `/api/sessions/current` | 当前会话信息 | 可选 |
| POST | `/api/import` | 粘贴文本导入 | session_id |
| POST | `/api/import/file` | 上传文件导入 | session_id |
| GET | `/api/knowledge-points` | 知识点列表（含统计） | session_id |
| POST | `/api/review/start` | 开始复习 | session_id |
| POST | `/api/review/{id}/answer` | 提交回答 | - |
| GET | `/api/review/{id}/next` | 获取下一题 | - |
| POST | `/api/review/{id}/exit` | 结束复习 | - |
| GET | `/api/analyze` | 薄弱分析 | session_id |
| GET | `/api/stats` | Dashboard 统计 | session_id |

**关于 session_id**：前端通过 localStorage 持有会话 ID，调用数据 API 时自动通过 `?session_id=X` 传递。后端无状态，不维护"当前会话"。

---

## 设计决策

| 决策 | 理由 |
|---|---|
| **三层架构** | Controller 只接 HTTP，Service 封装业务，底层专心管数据和图 |
| **节点内 ReAct** | 不加 ToolNode 撑胖图结构，每个节点自闭环 |
| **双车道调度** | 防饿死——薄弱点再多也能保证正常知识点不会被无限推迟 |
| **Hybrid Search** | BM25 高频召回 + 向量语义召回，0.7/0.3 加权融合 |
| **Session 隔离** | 多主题互不干扰，切换如切换上下文 |
| **内存缓存 + TTL** | 分析结果缓存 15 分钟，有新答题记录自动失效，不额外依赖 Redis |

详细决策见 `docs/decisions/`：
- [BM25 手写教程](docs/decisions/2026-07-13-BM25手写教程.md)
- [为什么用 BM25 做 RAG](docs/decisions/2026-07-13-为什么用BM25做RAG.md)
- [知识点去重策略](docs/decisions/2026-07-13-知识点去重策略.md)
- [双车道调度策略](docs/decisions/2026-07-14-双车道调度策略.md)
- [RAG 升级混合检索](docs/decisions/2026-07-24-RAG升级混合检索.md)
- [数据库并发连接策略](docs/decisions/2026-07-24-数据库并发连接策略.md)

---

## 面试能讲什么

| 面试官问 | 怎么说 |
|---|---|
| "LangGraph 怎么用的？" | 条件边驱动循环，interrupt 处理用户输入，MemorySaver checkpointer 保存暂停状态 |
| "RAG 怎么做的？" | BM25（手写，jieba 分词，k1/b 参数可讲）+ 向量检索（bge-small-zh，512 维）加权融合 0.7/0.3，benchmark/run_rag.py 扫描实测 |
| "三层架构怎么分的？" | Controller 只接 HTTP，Service 封装 LangGraph 驱动，底层管 DB + 图节点 |
| "判题逻辑？" | 不是字符串比较，是 LLM 根据标准答案 + 用户回答综合打分 |
| "Tool Calling 呢？" | question_gen 节点内嵌轻量 ReAct，LLM 可自主决定搜网络增强出题 |
| "Agent Memory？" | 出题/判分时注入历史答题记录——发现"漏了什么"、"进步了没有" |
| "薄弱分析怎么做的？" | 从 review_records 聚合，按 KP 统计均分/趋势 + 全局词频 + LLM 报告，结果缓存 15 分钟 |
| "怎么保证不重复考同一块？" | 双车道 2:1 交错 + 一轮一次约束，防止薄弱扩散导致饿死 |
| "向量检索为什么用本地的？" | bge-small-zh 仅 33MB，512 维本地 encode，不依赖外部向量数据库，部署即用 |

---

## 依赖

```text
fastapi / uvicorn
openai / langchain-core / langchain-openai
langgraph
sentence-transformers
sqlalchemy / pymysql
jieba / rank-bm25
ddgs / requests
python-dotenv
```

---

## 迭代路线

### 已完成

- [x] Web UI（FastAPI + React 19 + TypeScript + Vite，SPA HashRouter）
- [x] 三层架构重构（Controller → Service → Data）
- [x] 资料导入 → LLM 提取知识点（BM25 语义去重）
- [x] 混合检索（BM25 + bge-small-zh 向量，0.7/0.3 加权融合）
- [x] AI 主观题出题 + 判分 + 评语 + 优缺点 + 缺失知识点
- [x] LangGraph 条件边复习循环（interrupt + checkpointer）
- [x] 会话隔离（多主题互不干扰）
- [x] 答题记录持久化
- [x] 双车道薄弱感知调度（混合检索扩散）
- [x] 轻量 ReAct — LLM 自主搜索网络出题
- [x] Agent Memory — 历史答题注入上下文
- [x] 薄弱分析报告（统计 + LLM 报告 + 缓存）
- [x] Benchmark 评测套件（93 题 × 三维度）
- [x] 亮色/暗色主题切换
- [x] 开源准备（.gitignore、LICENSE、.env.example）

### 后续想法

- 多文档混合复习
- 遗忘曲线间隔重复
- V2 版本：意图识别 + 自然语言对话 + 多 Agent 编排（详见 [ROADMAP](docs/ROADMAP.md)）
