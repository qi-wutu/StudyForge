"""意图识别单测（V1.1 规则版）

把 8 类 intent 的关键样例 + 复习语境消歧锁定下来。
将来 V1.3 加 LLM 兜底时，靠这组用例防回归。
"""

from nlu.intent import classify_intent, extract_import_content


# ===== 强命令：无条件命中 =====

def test_强命令_退出():
    assert classify_intent("退出") == "exit_review"
    assert classify_intent("不考了，下次再学") == "exit_review"


def test_强命令_开始复习():
    for msg in ("开始复习", "复习一下", "考我", "出题", "测测数据库"):
        assert classify_intent(msg, review_active=False) == "start_review", msg


def test_强命令_下一题():
    for msg in ("下一题", "下一道", "继续", "换一题"):
        assert classify_intent(msg, review_active=True) == "next", msg


def test_强命令_分析():
    for msg in ("我哪里薄弱", "分析一下", "我的短板是什么", "哪些知识点没掌握"):
        assert classify_intent(msg, review_active=False) == "analyze", msg


def test_强命令_导入():
    for msg in ("导入：一些学习资料内容用于测试", "导入:"):
        assert classify_intent(msg, review_active=False) == "import", msg


# ===== 提问 / 复习中作答的消歧 =====

def test_非复习_提问_走qa():
    for msg in ("什么是 GMP 模型", "GMP 是什么？", "怎么理解工作窃取", "解释一下 P 的作用"):
        assert classify_intent(msg, review_active=False) == "qa", msg


def test_非复习_普通闲聊_走兜底():
    assert classify_intent("你好呀", review_active=False) == "smalltalk"
    assert classify_intent("谢谢", review_active=False) == "smalltalk"


def test_复习中_普通话_当作答():
    # 复习进行中，非命令、非明确提问 → 视为对当前题的回答
    for msg in ("G是goroutine，M负责调度", "答案大概是调度器", "我不知道"):
        assert classify_intent(msg, review_active=True) == "answer", msg


def test_复习中_明确提问_仍可插话qa():
    # 复习中也能插话提问 → 不算作答
    for msg in ("这个和协程有什么区别？", "什么是work stealing？"):
        assert classify_intent(msg, review_active=True) == "qa", msg


def test_复习中_强命令优先于作答():
    # 即使复习中，强命令也不该被当成答案
    assert classify_intent("退出", review_active=True) == "exit_review"
    assert classify_intent("下一题", review_active=True) == "next"


# ===== 空输入 =====

def test_空输入_兜底():
    assert classify_intent("", review_active=False) == "smalltalk"
    assert classify_intent("   ", review_active=True) == "smalltalk"


# ===== 导入内容提取 =====

def test_导入提取():
    content = "Go语言中GMP模型的三要素是Goroutine、M、P，三者配合实现高并发调度。"
    assert extract_import_content(f"导入：{content}") == content
    assert extract_import_content(f"导入 {content}") == content


def test_导入内容太短_返回None():
    assert extract_import_content("导入") is None
    assert extract_import_content("导入：太短") is None
