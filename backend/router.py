"""API 路由定义

只做 HTTP 解析和 JSON 返回，业务逻辑委托给 service 层。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile
from pydantic import BaseModel

from service import session_service, stats_service
from agent.import_agent import import_content
from agent.review_agent import review_agent
from agent.supervisor import supervisor

router = APIRouter()


# ========================================
# 请求格式说明书（Pydantic 模型）
#
# 每个请求体长什么样写在这里，FastAPI 自动校验。
# 请求不合规自动返回 422，不用自己写 try。
# ========================================

class CreateSessionRequest(BaseModel):
    name: str


class ImportRequest(BaseModel):
    content: str
    title: Optional[str] = None


class AnswerRequest(BaseModel):
    answer: str


class ChatRequest(BaseModel):
    message: str


# ========================================
# 会话管理
#
# 会话 ID 由前端 localStorage 持有，通过 ?session_id=X 传过来。
# 后端不存"当前会话"，只做 CRUD。
# ========================================

@router.get("/api/sessions")
def list_sessions():
    """获取所有会话列表"""
    return session_service.list_sessions()


@router.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    """创建新会话（前端拿到 id 后自己存 localStorage）"""
    try:
        return session_service.create_session(req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/sessions/current")
def current_session(session_id: Optional[int] = Query(None)):
    """获取当前会话信息

    前端传 session_id 就查指定会话，
    没传就返回/创建 default 会话。
    """
    sid = session_service.resolve_session_id(session_id)
    try:
        return session_service.get_session(sid)
    except ValueError:
        return {"id": sid, "name": "default"}


@router.post("/api/sessions/{session_id}/switch")
def switch_session(session_id: int):
    """切换会话（前端自己存 localStorage）"""
    try:
        info = session_service.get_session(session_id)
        # 后台静默预生成题目，不阻塞切换
        review_agent.prewarm(session_id)
        return info
    except ValueError as e:
        raise HTTPException(404, str(e))


# ========================================
# 资料导入
#
# 用户传 Markdown/文本 → service 层调 LangGraph：
#   import_graph → planner 节点 → LLM 提取知识点 → 写入 MySQL
#
# 整个流程是同步的——请求发出去到拿回结果是等着的。
# ========================================

@router.post("/api/import")
def import_document(req: ImportRequest, session_id: Optional[int] = Query(None)):
    """粘贴文本导入，AI 提取知识点"""
    sid = session_service.resolve_session_id(session_id)
    title = req.title or "未命名文档"
    return import_content(sid, req.content, title)


@router.post("/api/import/file")
async def import_file(file: UploadFile, session_id: Optional[int] = Query(None)):
    """上传文件导入（.md / .txt），AI 提取知识点"""
    sid = session_service.resolve_session_id(session_id)
    content = (await file.read()).decode("utf-8")
    title = file.filename or "未命名文档"
    return import_content(sid, content, title)


# ========================================
# 知识点查询
#
# 每个知识点附带统计：平均分、答题次数
# 数据来自两张表：knowledge_points + review_records
# ========================================

@router.get("/api/knowledge-points")
def list_knowledge_points(session_id: Optional[int] = Query(None)):
    """列出知识点，含历史答题统计"""
    sid = session_service.resolve_session_id(session_id)
    return stats_service.list_knowledge_points(sid)


# ========================================
# 复习
#
# 核心流程——复习图（review_graph）被拆到多个端点里分步调用：
#
#   POST /review/start
#       → 启动图，跑到 wait_input（interrupt，图暂停）
#       → 返回第一道题
#
#   POST /review/{id}/answer  {answer}
#       → Command(resume) 唤醒图
#       → 图跑完 judge → 继续跑到 wait_input 再暂停
#       → 返回评价 + 下一题
#
#   GET  /review/{id}/next
#       → 不跑图，光读 checkpointer 里存的当前状态
#       → 用于继续之前中断的复习
#
#   POST /review/{id}/exit
#       → Command(resume="__exit__") 唤醒图让它正常结束
#
# 复习状态通过 thread_id 串联，存在内存里（_active_reviews）。
# 服务重启后未完成的复习会丢失，需要重新 start。
# ========================================

@router.post("/api/review/start")
def start_review(session_id: Optional[int] = Query(None)):
    """开始一次复习——出第一道题"""
    sid = session_service.resolve_session_id(session_id)
    try:
        return review_agent.start(sid)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/api/review/{thread_id}/answer")
def submit_answer(thread_id: str, req: AnswerRequest):
    """提交回答 → AI 判分 → 返回评价"""
    return review_agent.submit_answer(thread_id, req.answer)


@router.get("/api/review/{thread_id}/next")
def get_next_question(thread_id: str):
    """获取当前 thread 的下一题（不跑图，只读缓存状态）"""
    try:
        return review_agent.get_next(thread_id)
    except ValueError as e:
        msg = str(e)
        raise HTTPException(404 if "已过期" in msg else 400, msg)


@router.post("/api/review/{thread_id}/exit")
def exit_review(thread_id: str):
    """主动结束复习"""
    return review_agent.exit(thread_id)


@router.get("/api/review/active")
def list_active_reviews():
    """查看当前正在进行的复习会话"""
    return {"active": review_agent.list_active()}


# ========================================
# 自然语言对话（V1.1 入口）
#
# 把「自然语言输入 → 意图识别 → 分发」做成一个入口。
# Supervisor 内部调度 复习/问答/导入/分析 四个子 Agent（agent/）。
# 现有各功能端点保留，后续可用对话逐步替代（见 ROADMAP）。
# ========================================

@router.post("/api/chat")
def chat(req: ChatRequest, session_id: Optional[int] = Query(None)):
    """自然语言交流入口——识别意图并分发到对应能力"""
    sid = session_service.resolve_session_id(session_id)
    return supervisor.chat(sid, req.message)


# ========================================
# 薄弱分析
#
# 两个层次：
#   1. 数据层——答题记录的统计聚合（平均分、高频缺失）
#   2. LLM 层——让 DeepSeek 写一段分析报告
#
# no_llm=true 时跳过 LLM，光出统计数据。
# ========================================

@router.get("/api/analyze")
def analyze(no_llm: bool = False, session_id: Optional[int] = Query(None)):
    """生成薄弱分析报告"""
    sid = session_service.resolve_session_id(session_id)
    result = stats_service.analyze(sid, llm_report=(not no_llm))
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# ========================================
# Dashboard 统计
#
# 首页概览用的汇总数据，一次查多张表拼起来。
# ========================================

@router.get("/api/stats")
def get_stats(session_id: Optional[int] = Query(None)):
    """获取 Dashboard 概览统计"""
    sid = session_service.resolve_session_id(session_id)
    return stats_service.get_dashboard_stats(sid)
