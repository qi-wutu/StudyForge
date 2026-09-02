# Benchmark 评测套件

两个评测目标，一套哲学（自包含数据 + 可重复跑 + 自动出报告）：

| 脚本         | 评测对象             | 说明                              |
|--------------|----------------------|-----------------------------------|
| `run.py`     | AI 判分（judge 节点） | 一致性 / 准确性 / 统一度三维度     |
| `run_rag.py` | 检索（retrieve）     | Hit@k / MRR，BM25 vs 向量 vs 混合 |

下方第 1 部分讲判分评测，第 2 部分讲 RAG 检索评测。

---

## 1. 判分评测（run.py）

评测 AI 判题系统（judge 节点）的**一致性**、**准确性**和**统一度**，三位一体。

---

## 评测三维度

### 1. 一致性 — 同题同答案跑多次，分数稳不稳？

一道题 + 一个 user_answer，让 AI 评多次（`temperature=0.3`），看分数有没有大幅波动。

**指标**：标准差（stddev）

- stddev ≤ 3  → ★ 稳定
- 3 < stddev ≤ 8 → ◆ 一般（有波动但可接受）
- stddev > 8 → ✗ 不稳定（需要调 prompt/temperature）

一致性差说明：同一个回答，AI 自己都拿不准该打几分。

### 2. 准确性 — AI 评分跟人工标注差多远？

每道题有人工标注的 `expected_score`，AI 给出的分数和它对比。

**指标**：MAE（平均绝对偏差）

- MAE ≤ 10 → ★ 准确
- 10 < MAE ≤ 20 → ◆ 偏大（趋势对但不够准）
- MAE > 20 → ✗ 偏差大（需要调 prompt）

### 3. 统一度 — 评分尺子是不是一把？

这是"评分体系"层面的统合检验，包含三个子维度：

| 子维度 | 方法 | 说明 |
|--------|------|------|
| 排序一致性 | Spearman 秩相关 | 人工给分高的回答 → AI 也应给高分。如果 A 回答预期 80、B 预期 50，但 AI 给了 A 60、B 70，排序就反了 |
| 分层偏差 | 按预期分数分层（低/中/高） | 看低分层（<60）、中分层（60-75）、高分层（>75）的 AI 评分是否各自聚合在合理区间 |
| 系统性偏差 | 各主题平均偏差 | 是不是 AI 对数据库题目总是给高 5 分，对操作系统总是给低 3 分 |

统一度不消耗额外 API 调用，直接用准确性的数据计算。

---

## 目录结构

```
benchmark/
├── README.md            # 本文件
├── run.py               # 评测入口（完全独立，不依赖主项目代码）
├── __init__.py
├── cases/               # 测试用例（JSON）
│   ├── data_structures.json
│   ├── operating_systems.json
│   ├── networking.json
│   ├── databases.json
│   └── golang.json
└── reports/             # 评测报告（自动生成）
```

## 用例分布

| 文件 | 题数 | 覆盖主题 |
|------|------|----------|
| data_structures.json | 19 | 数组、链表、哈希表、栈队列、树、排序、二分查找、DP、LRU、图、红黑树、堆、KMP、布隆过滤器、跳表、回溯、Trie、拓扑排序、滑动窗口 |
| operating_systems.json | 18 | 进程线程、死锁、虚拟内存、调度、页面置换、中断、同步、IO模型、内存分配、文件系统、IPC、协程、用户态内核态、DMA、零拷贝、epoll、自旋锁 |
| networking.json | 19 | TCP三次握手/四次挥手、TCP vs UDP、HTTP、HTTPS、DNS、拥塞控制、状态码、滑动窗口、HTTP/2、粘包、OSI模型、ARP、缓存、WebSocket、负载均衡、超时重传、HTTP/1.0 vs 1.1、CDN |
| databases.json | 18 | ACID、索引、隔离级别、MVCC、锁、EXPLAIN、Redis数据结构/持久化/淘汰、主从复制、哨兵vs集群、B+树、缓存穿透击穿雪崩、死锁、事务对比、慢查询、分布式锁、分库分表 |
| golang.json | 19 | GMP、channel、slice、defer、map、interface、goroutine泄漏、GC、select、panic/recover、反射、并发模式、内存对齐、context、string、WaitGroup、指针vs值接收者、data race、mutex饥饿模式 |
| **总计** | **93** | 覆盖 5 大方向，93 道主观题 |

## 每道题的数据结构

```json
{
  "id": "ds-001",
  "topic": "数组 vs 链表",
  "question": "请对比数组和链表...",
  "standard_answer": "数组是连续内存空间...",
  "user_answer": "数组是连续的内存...",
  "expected_score": 80,
  "expected_comment": "回答基本正确..."
}
```

设计原则：
- 分数覆盖 **40-90** 区间，不是全在 80-100
- user_answer 包含各种类型：基本正确、略懂皮毛、深度缺失、偏题
- 同一主题下分数有梯度，能测出排序一致性

---

## 用法

```bash
# 安装依赖
pip install openai python-dotenv

# 确保 .env 中有 LLM_API_KEY

# 只测准确性（全部跑 1 次，省额度）
python benchmark/run.py

# 分拆模式：抽 10 题跑 5 次（一致性），其余跑 1 次（准确性）
python benchmark/run.py --multi 10 --runs 5

# 快速验证一致性（抽 3 题跑 3 次）
python benchmark/run.py --multi 3 --runs 3 --cases golang

# 指定分组
python benchmark/run.py --cases databases
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--multi N` | 0 | 抽取 N 道题跑多次（一致性测试），其余跑 1 次（准确性测试） |
| `--runs N` | 5 | 一致性样本的重复次数 |
| `--cases X` | 全部 | 只跑某分组，如 `--cases golang` |
| `--output P` | 自动 | 指定报告路径 |

### 模式速查

| 想干嘛 | 命令 |
|--------|------|
| 只想看准确性 | `python benchmark/run.py` |
| 顺带看一致性 | `python benchmark/run.py --multi 10 --runs 5` |
| 开发中快速验证 | `python benchmark/run.py --multi 3 --runs 3 --cases golang` |
| 调完 prompt 看效果 | `python benchmark/run.py --cases databases --multi 5 --runs 5` |

---

## 报告结构

生成的报告包含五个章节：

```
一、一致性评分       ← 多轮样本的标准差分布 + 详情表
二、准确性评分       ← MAE + 偏差最大 TOP5
三、统一度分析       ← Spearman + 分层偏差 + 主题偏差
四、各主题分组统计    ← 分组平均分
五、各题目详细结果    ← 每道题的分值、均值、标准差
```

## 注意事项

1. 运行前在 `.env` 中配置 `LLM_API_KEY`
2. 评测脚本**完全不 import 主项目**（`config.py`、`graph/`、`storage/`等），只读 `.env`
3. 一致性样本用 `temperature=0.3` 制造波动，准确性样本用 `temperature=0` 保证确定
4. 报告自动保存到 `benchmark/reports/` 目录
5. 全量跑 93 题 × 1 次 ≈ 93 次 API 调用；加 --multi 10 --runs 5 ≈ 93+40=133 次

---

## 2. RAG 检索评测（run_rag.py）

评测**检索器**（不是 LLM 判分）——一条查询能不能召回到正确的知识点。

### 评测对象

`rag/` 下的三个真实检索器，不重新实现：

| 检索器 | 原理 |
|--------|------|
| BM25Index | jieba 分词 + BM25，词面精确命中 |
| VectorIndex | bge-small-zh-v1.5 向量 + 余弦相似度，语义召回 |
| HybridRetriever | 0.7 BM25 + 0.3 向量加权融合 |

### 评测集

- **corpus**：`cases/*.json` 里 93 道题的标准答案 → 93 个"知识点"（与生产的 `kp.content` 对齐，只存正文不拼标题）
- **查询分两类**：
  - 词面查询（93 条）：每道题的原问题，关键词重叠高 → 测词面检索
  - 语义查询（15 条）：`cases_rag/hard_queries.json` 手工改写，避开目标标题关键词 → 测语义检索

### 指标

- **Hit@k / Recall@k**：相关知识点在不在前 k 个结果里
- **MRR@k**：首个相关知识点位置倒数的均值（越靠前越高）
- **分层**：词面 / 语义分开统计
- **权重扫描**：bm25_weight 从 0 扫到 1，验证默认 0.7/0.3 是否最优

### 评测用法

```bash
python benchmark/run_rag.py                   # 全量（93+15 条查询）
python benchmark/run_rag.py --cases golang    # 只跑某个主题
python benchmark/run_rag.py --no-sweep        # 跳过权重扫描
python benchmark/run_rag.py --topk 5          # 只看 top-5
```

本地跑，不调 LLM API、不依赖数据库，零成本可重复。

### 评测目录

```text
benchmark/
├── run_rag.py               # 检索评测入口
└── cases_rag/
    └── hard_queries.json    # 语义改写查询（手工标注相关知识点）
```
