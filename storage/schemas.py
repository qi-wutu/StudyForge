"""SQLAlchemy ORM 模型 — 一张表 = 一个 Python 类

ORM 的意思是"对象关系映射"：
  不用写 SQL，用 Python 对象操作数据库。

例子：
  doc = Document(title="xxx", content="yyy")   ← 创建一行
  db.add(doc)                                    ← 插入
  db.commit()                                    ← 提交

为什么用 ORM 而不是直接写 SQL？
  1. 代码更干净，没有字符串拼接的 SQL
  2. 改表结构只用改这个文件，不用到处找 SQL
  3. 面试时可以说"我用过 SQLAlchemy"，很常见
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, func
from sqlalchemy.orm import DeclarativeBase


# Base = 所有模型的基类
# SQLAlchemy 通过扫描 Base 的子类来自动建表
class Base(DeclarativeBase):
    pass


class Session(Base):
    """会话表 — 不同学习主题互相隔离

    每个 session 对应一个学习主题（如"Go 八股"、"Redis 八股"），
    有自己的资料、知识点、答题记录。
    切换 session 就像切换"上下文"，数据互不干扰。
    """
    __tablename__ = "sessions"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(255), nullable=False, unique=True)  # 会话名，如 "golang"
    created_at = Column(DateTime, server_default=func.now())


class Document(Base):
    """原始文档表 — 用户导入的每一份资料

    你执行 python main.py import xxx.md 时，
    文件内容先存到这里，拿到 id 后 planner 节点
    再把知识点关联到这个文档。
    """
    __tablename__ = "documents"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False, default=0)  # 属于哪个会话
    title      = Column(String(255), nullable=False)   # 文件名
    content    = Column(Text, nullable=False)           # 完整内容
    created_at = Column(DateTime, server_default=func.now())


class KnowledgePoint(Base):
    """知识点表 — 从资料里提取出来的知识点

    每一条知识点对应一个"考点"或"主题"。
    出题时 scheduler 从这张表取知识点给 question_gen。
    """
    __tablename__ = "knowledge_points"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer, nullable=False, default=0)  # 属于哪个会话
    document_id = Column(Integer, nullable=False)  # 关联到哪篇文档
    title       = Column(String(255), nullable=False)  # 知识点名称，如"GMP 模型"
    content     = Column(Text, nullable=False)         # 知识点内容（标准答案）
    embedding   = Column(JSON, nullable=True)          # 384维向量，Hybrid Search
    created_at  = Column(DateTime, server_default=func.now())


class Question(Base):
    """预生成的题目表 — 减少 LLM 调用

    每次 question_gen 生成的题目存下来。
    下次同知识点出题时优先从库里取，不用再调 LLM。
    """
    __tablename__ = "questions"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, nullable=False, default=0)
    kp_id        = Column(Integer, nullable=False)   # 关联到哪个知识点
    title        = Column(String(255), nullable=False)  # 题目标题
    question_text = Column(Text, nullable=False)        # 题目内容
    use_count    = Column(Integer, default=0)           # 被用过几次（用于负载均衡）
    created_at   = Column(DateTime, server_default=func.now())


class ReviewRecord(Base):
    """答题记录表 — 每次用户答题 + 判题的全量记录

    存下来的目的：
      - 查历史正确率（analyzer 分析薄弱点时用）
      - 用户可以看到"我之前答过什么"
      - 后续可以根据历史调整复习策略

    JSON 字段（ai_strengths 等）：
      MySQL 直接存 JSON 字符串，
      SQLAlchemy 自动做 Python list ↔ JSON 的互转。
    """
    __tablename__ = "review_records"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(Integer, nullable=False, default=0)  # 属于哪个会话
    kp_id          = Column(Integer, nullable=False)   # 关联到哪个知识点
    question       = Column(Text, nullable=False)      # 当时出的题
    user_answer    = Column(Text, nullable=False)      # 用户怎么答的
    ai_score       = Column(Integer)                   # DeepSeek 给的分数（0-100）
    ai_comment     = Column(Text)                      # 评语
    ai_strengths   = Column(JSON)                      # 优点列表
    ai_weaknesses  = Column(JSON)                      # 不足列表
    ai_missing_kps = Column(JSON)                      # 缺失知识点列表（反哺复习）
    created_at     = Column(DateTime, server_default=func.now())
