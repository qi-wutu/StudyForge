"""简易意图识别

V1.1：规则优先 —— 高效、可预测、不额外消耗 LLM。
后续可在「规则拿不准」时接 LLM 兜底，升级成多 Agent supervisor 的 intent node。

识别前需要知道当前是否在复习中（review_active），用来消歧：
  - 复习进行中：非命令、非明确提问的普通消息 → 视为「回答当前题」(answer)
  - 否则：按提问/闲聊处理

返回一个字符串 intent：
  exit_review / start_review / next / analyze / import / qa / answer / smalltalk
"""

import re

# 强命令（无条件优先，命中即生效）
_EXIT = re.compile(r"(退出|结束|不考了?|停(一下|止)?|算了|下次再|先不(考|学)|stop|quit|exit)", re.I)
_START = re.compile(r"^(开始.?复习|复习(一下|一遍|了吧|一下)?$|考(我|考)?|来考|出题|测测|测试|考考)", re.I)
_NEXT = re.compile(r"^(下一题|下一道|继续|再来一题|换一题|再来一道|next|下一题吧)", re.I)
_ANALYZE = re.compile(r"(薄弱|哪里(不|没)行|我(哪|哪里)弱|分析(一下)?|短板|报告|没掌握|没学会|当前水平|哪些薄弱)", re.I)
_IMPORT = re.compile(r"^(导入|收录|添加资料|存一下|记一下|导入一下)", re.I)

# 提问判定：结尾问号，或以疑问词开头
_QUESTION_END = re.compile(r"[？?]$")
_QUESTION_START = re.compile(
    r"^(什么是|是什么|啥是|啥叫|怎么|为什么|为何|怎样|如何|讲讲|解释|区别|说一下|"
    r"what|how|why|could|can|tell me)",
    re.I,
)


def _looks_like_question(msg: str) -> bool:
    if _QUESTION_END.search(msg):
        return True
    if _QUESTION_START.search(msg):
        return True
    return False


def classify_intent(message: str, *, review_active: bool = False) -> str:
    """对用户一条消息做意图识别。

    Args:
        message: 用户原始输入
        review_active: 当前是否处于「复习中」（有未答完的题）

    Returns:
        见模块 docstring 的 intent 列表
    """
    msg = message.strip()
    if not msg:
        return "smalltalk"

    # 1. 强命令——无条件最高优先
    if _EXIT.search(msg):
        return "exit_review"
    if _START.match(msg):
        return "start_review"
    if _NEXT.match(msg):
        return "next"
    if _ANALYZE.search(msg):
        return "analyze"
    if _IMPORT.match(msg):
        return "import"

    # 2. 复习中：非命令、非明确提问 → 当作对当前题的回答
    if review_active:
        if _looks_like_question(msg):
            return "qa"
        return "answer"

    # 3. 非复习：提问 → qa，否则兜底闲聊
    if _looks_like_question(msg):
        return "qa"
    return "smalltalk"


# ===== 导入内容的提取（V1.1 简单前缀式） =====

def extract_import_content(message: str) -> str | None:
    """从「导入：xxx / 导入 xxx」里取出真正的内容。

    没有内容（太短）返回 None，由上层引导用户。
    """
    msg = message.strip()
    for prefix in ("导入：", "导入:", "导入 ", "收录："):
        if msg.startswith(prefix):
            content = msg[len(prefix):].strip()
            if len(content) > 20:
                return content
    return None
