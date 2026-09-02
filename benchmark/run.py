"""Benchmark 评测脚本 — 综合评估 LLM 判题能力

完全独立于主项目代码，不 import main 项目的任何模块。
直接调用 DeepSeek API，模拟 judge 节点的评分逻辑。

三大评测维度：
  1. 一致性（Consistency）——同一道题 × N 次，分数波动越小越好
  2. 准确性（Accuracy）——AI 评分 vs 人工标注期望分数，偏差越小越好
  3. 统一度（Uniformity）——跨题评分尺度是否统一：
     - 预期分数低的回答 → AI 给低分，预期高的 → AI 给高分（单调性）
     - 相同质量层次（按预期分组）的答案，AI 给分聚集在合理区间
     - 排名一致性：AI 排序 vs 人工排序的 Spearman 秩相关

用法：
  cd StudyForge
  python benchmark/run.py                    # 跑全部 case，每道题跑 3 次
  python benchmark/run.py --runs 5           # 每道题跑 5 次
  python benchmark/run.py --cases golang     # 只跑某个 file
  python benchmark/run.py --fast             # 快速模式：每题只跑 1 次
"""

import json
import math
import os
import random
import statistics
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("需要安装 openai 包：pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("需要安装 python-dotenv 包：pip install python-dotenv")
    sys.exit(1)

# ============================================================
# 配置
# ============================================================

load_dotenv()  # 从 .env 读取 LLM_API_KEY 等

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = Path(__file__).resolve().parent / "cases"

# ============================================================
# 评测 prompt（与 graph/node.py 中的 judge 节点保持一致）
# ============================================================

JUDGE_SYSTEM_PROMPT = """你是一个专业的面试官。请根据标准答案评判用户的回答。

要求：
1. 给出 0-100 的分数
2. 写出总体评语
3. 列出优点和不足
4. 列出用户没答出来的知识点（missing_kps），用于后续复习

返回 JSON 格式，示例：
{{
    "score": 72,
    "comment": "基本正确，但对 P 的作用理解不够深入",
    "strengths": ["正确说明了 G 是 goroutine"],
    "weaknesses": ["没有提到工作窃取机制"],
    "missing_kps": ["GMP 中 P 的本地运行队列", "work stealing"]
}}

只输出 JSON，不要输出其他内容。"""


def build_judge_prompt(question: str, standard_answer: str, user_answer: str) -> str:
    """构建 judge 的 prompt"""
    return f"""标准答案：{standard_answer}

题目：{question}

用户回答：{user_answer}"""


# ============================================================
# LLM 调用封装
# ============================================================

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def call_judge(question: str, standard_answer: str, user_answer: str,
               temperature: float = 0.3,
               max_retries: int = 2) -> dict[str, Any]:
    """调用 LLM 执行一次评分，返回评 JSON

    Args:
        temperature: 控制随机性。0 表示确定性的，越高越随机。
    Returns:
        {"score": int, "comment": str, "strengths": [...], "weaknesses": [...], "missing_kps": [...]}
    """
    prompt = build_judge_prompt(question, standard_answer, user_answer)
    last_error = ""

    for attempt in range(1 + max_retries):
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                # 让 LLM 倾向于输出 JSON 格式
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response")

            result = json.loads(content)

            # 验证必要字段
            if "score" not in result:
                raise ValueError(f"Missing 'score' field: {content[:200]}")

            # 确保 score 是数字且在合理范围
            result["score"] = int(result["score"])
            result["score"] = max(0, min(100, result["score"]))

            return result

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(1)  # 等待后重试
                continue

    return {
        "score": -1,
        "comment": f"调用失败：{last_error}",
        "strengths": [],
        "weaknesses": [],
        "missing_kps": [],
    }


# ============================================================
# 数据加载
# ============================================================

def load_all_cases() -> list[dict]:
    """从 cases 目录加载所有测试用例，返回打平的列表"""
    all_cases = []
    case_files = sorted(CASES_DIR.glob("*.json"))

    for fpath in case_files:
        topic_group = fpath.stem  # 文件名作为主题分组
        with open(fpath, "r", encoding="utf-8") as f:
            cases = json.load(f)
        for case in cases:
            case["_group"] = topic_group
            case["_file"] = fpath.name
        all_cases.extend(cases)

    return all_cases


def load_cases_by_group(group: str) -> list[dict]:
    """加载指定 group 的测试用例"""
    fpath = CASES_DIR / f"{group}.json"
    if not fpath.exists():
        print(f"  错误：未找到 {fpath}")
        print(f"  可选：{[p.stem for p in CASES_DIR.glob('*.json')]}")
        sys.exit(1)

    with open(fpath, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for case in cases:
        case["_group"] = fpath.stem
        case["_file"] = fpath.name
    return cases


# ============================================================
# 评分统计
# ============================================================

def compute_stats(scores: list[int]) -> dict:
    """计算分数列表的统计指标"""
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": 0, "median": 0, "stddev": 0, "min": 0, "max": 0}

    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stddev = statistics.stdev(scores) if n > 1 else 0

    return {
        "n": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "stddev": round(stddev, 2),
        "min": min(scores),
        "max": max(scores),
    }


# ============================================================
# 主流程
# ============================================================


def _pick_consistency_samples(cases: list[dict], n: int) -> set[str]:
    """从各主题均衡抽取 N 道题作为一致性测试样本"""
    if n >= len(cases):
        return {c["id"] for c in cases}

    groups: dict[str, list[str]] = {}
    for c in cases:
        groups.setdefault(c["_group"], []).append(c["id"])

    selected = set()
    total = len(cases)
    group_order = sorted(groups.keys(), key=lambda g: -len(groups[g]))

    for g in group_order:
        g_ids = groups[g]
        share = max(1, round(n * len(g_ids) / total))
        take = min(share, len(g_ids))
        for gid in g_ids:
            if len(selected) >= n:
                break
            if gid not in selected:
                selected.add(gid)
    return selected


def run_benchmark(cases: list[dict],
                  consistency_ids: set[str] | None = None,
                  consistency_runs: int = 5,
                  verbose: bool = True) -> tuple[list[dict], float]:
    """运行 benchmark 双模式

    Args:
        cases: 测试用例列表
        consistency_ids: 需要跑多次的样本 ID 集合；None 则全部跑多次
        consistency_runs: 一致性样本的重复次数
        verbose: 是否打印

    Returns:
        (all_results, elapsed_seconds)
    """
    if not API_KEY:
        print("  错误：未设置 LLM_API_KEY，请在 .env 中配置")
        sys.exit(1)

    all_results = []
    total = len(cases)
    multi_count = len(consistency_ids) if consistency_ids else total
    single_count = total - multi_count

    print(f"\n  开始 Benchmark 评测")
    print(f"  模型：{MODEL}")
    print(f"  用例总数：{total}")
    if consistency_ids is not None:
        print(f"  模式：{multi_count} 题跑 {consistency_runs} 次 (一致性) + {single_count} 题跑 1 次 (准确性)")
    else:
        print(f"  每题运行次数：{consistency_runs}")
    print(f"  {'='*50}\n")

    start_time = time.time()

    for idx, case in enumerate(cases, 1):
        case_id = case["id"]
        topic = case["topic"]
        question = case["question"]
        standard_answer = case["standard_answer"]
        user_answer = case["user_answer"]

        is_multi = consistency_ids is not None and case_id in consistency_ids
        per_case_runs = consistency_runs if is_multi else 1
        temperature = 0.3 if is_multi else 0.0

        scores: list[int] = []
        comments: list[str] = []

        for run_i in range(per_case_runs):
            result = call_judge(question, standard_answer, user_answer,
                                temperature=temperature)
            score = result.get("score", -1)
            if score >= 0:
                scores.append(score)
                comments.append(result.get("comment", ""))

            if verbose:
                tag = " [一致]" if is_multi else ""
                print(f"   第 {run_i+1}/{per_case_runs} 次 -> 分数 {score}{tag}")

        stats = compute_stats(scores)

        if verbose:
            expected = case.get("expected_score")
            expected_str = f" (预期 {expected})" if expected else ""
            tag_str = " *多轮" if is_multi else ""
            print(f"  [{idx}/{total}] {case_id} {topic}{tag_str}")
            print(f"    分数列表：{scores}")
            print(f"    均值 {stats['mean']}  标准差 {stats['stddev']}{expected_str}")
            print()

        all_results.append({
            "case": case,
            "scores": scores,
            "comments": comments,
            "stats": stats,
            "_is_consistency": is_multi,
        })

    elapsed = time.time() - start_time
    print(f"\n  评测完成！总用时：{elapsed:.1f}s")
    print(f"  平均每题用时：{elapsed / total:.1f}s")

    return all_results, elapsed


def generate_report(all_results: list[dict], case_count: int,
                    consistency_runs: int, elapsed: float) -> str:
    """生成评测报告 — 一致性 + 准确性 + 统一度"""
    lines = []
    lines.append("# Benchmark 评测报告")
    lines.append(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型：{MODEL}")

    consistency_results = [r for r in all_results if r["_is_consistency"]]

    cons_stddevs = []
    for r in consistency_results:
        s = r["stats"]
        if s["n"] > 1 and s["mean"] > 0:
            cons_stddevs.append(s["stddev"])

    # 准确性数据（全部题目）
    acc_diffs = []
    for r in all_results:
        c = r["case"]
        exp = c.get("expected_score")
        if exp is not None and r["stats"]["mean"] > 0:
            acc_diffs.append(abs(r["stats"]["mean"] - exp))

    multi_count = len(consistency_results)
    single_count = case_count - multi_count
    lines.append(f"\n配置：")
    lines.append(f"- 一致性测试：{multi_count} 题 * {consistency_runs} 次")
    lines.append(f"- 准确性测试：{case_count} 题")
    lines.append(f"- 用时：{elapsed:.1f}s")

    # === 一、一致性 ===
    lines.append("\n---\n## 一、一致性评分\n")
    lines.append("同题同答案多次评分，标准差越小越稳定。\n")

    if cons_stddevs:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|---|---|")
        lines.append(f"| 样本数 | {len(cons_stddevs)} |")
        lines.append(f"| 平均标准差 | {statistics.mean(cons_stddevs):.2f} |")
        lines.append(f"| 最大标准差（最不稳定） | {max(cons_stddevs):.2f} |")
        lines.append(f"| 最小标准差（最稳定） | {min(cons_stddevs):.2f} |")
        lines.append(f"| 中位数标准差 | {statistics.median(cons_stddevs):.2f} |")
        lines.append("")

        stable = sum(1 for s in cons_stddevs if s <= 3)
        moderate = sum(1 for s in cons_stddevs if 3 < s <= 8)
        unstable = sum(1 for s in cons_stddevs if s > 8)
        total_cons = len(cons_stddevs)
        lines.append(f"| 等级 | stddev 范围 | 题数 | 占比 |")
        lines.append(f"|---|---|---|---|")
        lines.append(f"| * 稳定 | <= 3 | {stable} | {stable/total_cons*100:.0f}% |")
        lines.append(f"| ~ 一般 | 3~8 | {moderate} | {moderate/total_cons*100:.0f}% |")
        lines.append(f"| x 不稳定 | > 8 | {unstable} | {unstable/total_cons*100:.0f}% |")
        lines.append("")

        lines.append("### 一致性详情\n")
        lines.append(f"| 题号 | 主题 | 分值列表 | 均值 | 标准差 |")
        lines.append(f"|---|---|---|---|---|")
        for r in consistency_results:
            c = r["case"]
            s = r["stats"]
            score_str = "/".join(str(v) for v in r["scores"])
            lines.append(f"| {c['id']} | {c['topic']} | {score_str} | {s['mean']} | {s['stddev']} |")
        lines.append("")
    else:
        lines.append("（本次未配置多轮一致性测试）\n")

    # === 二、准确性 ===
    lines.append("---\n## 二、准确性评分\n")
    lines.append("AI 评分 vs 人工期望分数。MAE = 平均绝对偏差。\n")

    if acc_diffs:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|---|---|")
        lines.append(f"| 有效样本数 | {len(acc_diffs)} |")
        lines.append(f"| MAE | {statistics.mean(acc_diffs):.2f} |")
        lines.append(f"| 中位数偏差 | {statistics.median(acc_diffs):.2f} |")
        lines.append(f"| 最大偏差 | {max(acc_diffs):.2f} |")
        lines.append(f"| 最小偏差 | {min(acc_diffs):.2f} |")
        lines.append("")

        close = sum(1 for d in acc_diffs if d <= 10)
        off = sum(1 for d in acc_diffs if 10 < d <= 20)
        far = sum(1 for d in acc_diffs if d > 20)
        lines.append(f"| 等级 | 偏差范围 | 题数 | 占比 |")
        lines.append(f"|---|---|---|---|")
        lines.append(f"| * 准确 | <= 10 | {close} | {close/len(acc_diffs)*100:.0f}% |")
        lines.append(f"| ~ 偏大 | 10~20 | {off} | {off/len(acc_diffs)*100:.0f}% |")
        lines.append(f"| x 偏差大 | > 20 | {far} | {far/len(acc_diffs)*100:.0f}% |")
        lines.append("")

        # 偏差 TOP5
        sorted_by_diff = sorted(
            all_results,
            key=lambda r: abs(r["stats"]["mean"] - r["case"].get("expected_score", 999))
            if r["case"].get("expected_score") else 999,
            reverse=True,
        )
        lines.append("### 偏差最大 TOP 5\n")
        lines.append(f"| 题号 | 主题 | AI 均值 | 预期 | 偏差 |")
        lines.append(f"|---|---|---|---|---|")
        shown = 0
        for r in sorted_by_diff:
            c = r["case"]
            exp = c.get("expected_score")
            if exp is None or r["stats"]["mean"] <= 0:
                continue
            diff = abs(r["stats"]["mean"] - exp)
            lines.append(f"| {c['id']} | {c['topic']} | {r['stats']['mean']} | {exp} | {diff:.1f} |")
            shown += 1
            if shown >= 5:
                break
        lines.append("")

    # === 三、统一度 ===
    lines.append("---\n## 三、统一度分析\n")
    lines.append("衡量 AI 评分跨题尺度的统一性：")
    lines.append("- 排名一致：人工预期高的回答，AI 也应给高分")
    lines.append("- 层次分明：不同质量等级应有显著差距")
    lines.append("- 无系统性偏差：不倾向某些主题整体偏高/偏低\n")

    # 3-1. Spearman
    pairs = [(c.get("expected_score"), r["stats"]["mean"])
             for r in all_results if (c := r["case"]).get("expected_score") and r["stats"]["mean"] > 0]
    n_pairs = len(pairs)

    lines.append("### 3.1 排序一致性 (Spearman 秩相关)\n")
    if n_pairs >= 10:
        expected_vals = sorted(set(p[0] for p in pairs))
        ai_vals = sorted(set(p[1] for p in pairs))
        expected_rank = {v: i+1 for i, v in enumerate(expected_vals)}
        ai_rank = {v: i+1 for i, v in enumerate(ai_vals)}

        rank_diffs_sq = sum((expected_rank[e] - ai_rank[a])**2 for e, a in pairs)
        spearman = 1 - (6 * rank_diffs_sq) / (n_pairs * (n_pairs**2 - 1))

        lines.append(f"- **Spearman rho = {spearman:.3f}**")
        if spearman > 0.7:
            lines.append("- 解读：AI 排序与人工高度一致")
        elif spearman > 0.4:
            lines.append("- 解读：AI 排序与人工中等一致")
        else:
            lines.append("- 解读：AI 排序与人工一致性低")
        lines.append(f"- 样本量：{n_pairs} 对")
        lines.append("")
    else:
        lines.append("（样本量不足，需 >= 10）\n")

    # 3-2. 按预期分数分层
    lines.append("### 3.2 按质量层次分层\n")
    tiers = {"低 (<60)": [], "中 (60-75)": [], "高 (>75)": []}
    tier_ranges = {"低 (<60)": (0, 60), "中 (60-75)": (60, 75), "高 (>75)": (75, 100)}
    tier_devs = {"低 (<60)": [], "中 (60-75)": [], "高 (>75)": []}

    for r in all_results:
        c = r["case"]
        exp = c.get("expected_score")
        if exp is None or r["stats"]["mean"] <= 0:
            continue
        for tname, (lo, hi) in tier_ranges.items():
            if lo <= exp < hi:
                tiers[tname].append(r["stats"]["mean"])
                tier_devs[tname].append(abs(r["stats"]["mean"] - exp))
                break

    lines.append(f"| 质量层 | 区间 | 题数 | AI 平均分 | 中位偏差 |")
    lines.append(f"|---|---|---|---|---|")
    for tname in ["低 (<60)", "中 (60-75)", "高 (>75)"]:
        ai_scores = tiers[tname]
        devs = tier_devs[tname]
        lo, hi = tier_ranges[tname]
        if not ai_scores:
            lines.append(f"| {tname} | {lo}-{hi} | 0 | - | - |")
        else:
            ai_mean = statistics.mean(ai_scores)
            med_dev = statistics.median(devs) if devs else 0
            lines.append(f"| {tname} | {lo}-{hi} | {len(ai_scores)} | {ai_mean:.1f} | {med_dev:.1f} |")
    lines.append("")

    # 3-3. 各主题偏差
    lines.append("### 3.3 各主题系统性偏差\n")
    groups: dict[str, list[float]] = {}
    for r in all_results:
        c = r["case"]
        exp = c.get("expected_score")
        if exp is None or r["stats"]["mean"] <= 0:
            continue
        groups.setdefault(c["_group"], []).append(r["stats"]["mean"] - exp)

    if groups:
        lines.append(f"| 主题 | 题数 | 平均偏差 | 趋势 |")
        lines.append(f"|---|---|---|---|")
        for gname, diffs in sorted(groups.items()):
            avg_dev = statistics.mean(diffs)
            if abs(avg_dev) < 3:
                trend = "~ 正常"
            elif avg_dev > 0:
                trend = "↑ 偏高"
            else:
                trend = "↓ 偏低"
            lines.append(f"| {gname} | {len(diffs)} | {avg_dev:+.1f} | {trend} |")
        lines.append("")

    # === 四、各主题分组 ===
    lines.append("---\n## 四、各主题分组统计\n")
    for gname in sorted(groups.keys()):
        g_results = [r for r in all_results if r["case"]["_group"] == gname and r["stats"]["mean"] > 0]
        if not g_results:
            continue
        g_scores = [r["stats"]["mean"] for r in g_results]
        lines.append(f"### {gname}\n")
        lines.append(f"- 题数：{len(g_results)}")
        lines.append(f"- 平均分：{statistics.mean(g_scores):.2f}")
        lines.append("")

    # === 五、各题目详情 ===
    lines.append("---\n## 五、各题目详细结果\n")
    for r in all_results:
        c = r["case"]
        s = r["stats"]
        scores = r["scores"]
        tag = " [一致性样本]" if r["_is_consistency"] else ""
        diff_text = ""
        if c.get("expected_score") and s["mean"] > 0:
            diff = round(s["mean"] - c["expected_score"], 1)
            diff_text = f"偏差 {diff:+.1f}"

        lines.append(f"### [{c['id']}] {c['topic']}{tag}\n")
        lines.append(f"- **题目**：{c['question'][:80]}...")
        lines.append(f"- **分数**：{scores} (共 {s['n']} 次)")
        lines.append(f"- **均值/中位数**：{s['mean']} / {s['median']}")
        lines.append(f"- **标准差**：{s['stddev']}  {diff_text}")
        if c.get("expected_score"):
            lines.append(f"- **预期分数**：{c['expected_score']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = ArgumentParser(description="StudyForge Benchmark 评测工具")
    parser.add_argument("--multi", type=int, default=0,
                        help="一致性样本量：抽 N 题跑多次看方差（默认 0 = 全部只跑 1 次）")
    parser.add_argument("--runs", type=int, default=5,
                        help="一致性样本的重复次数（默认 5）")
    parser.add_argument("--cases", type=str, default="",
                        help="指定分组（如 golang，不指定则全部）")
    parser.add_argument("--output", type=str, default="",
                        help="报告输出路径")
    args = parser.parse_args()

    # 加载用例
    if args.cases:
        print(f"\n  加载用例：{args.cases}")
        cases = load_cases_by_group(args.cases)
    else:
        cases = load_all_cases()

    print(f"  共加载 {len(cases)} 道题目")

    # 决定哪些题跑多次
    if args.multi > 0:
        consistency_ids = _pick_consistency_samples(cases, args.multi)
        print(f"  一致性样本（跑 {args.runs} 次）：{len(consistency_ids)} 题")
        print(f"  其余 {len(cases) - len(consistency_ids)} 题跑 1 次")
    else:
        consistency_ids = None  # 全部只跑 1 次

    # 运行
    all_results, elapsed = run_benchmark(cases,
                                         consistency_ids=consistency_ids,
                                         consistency_runs=args.runs,
                                         verbose=True)

    # 报告
    report = generate_report(all_results, len(cases), args.runs, elapsed)

    reports_dir = PROJECT_ROOT / "benchmark" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = args.cases if args.cases else "all"
    report_path = reports_dir / f"report_{report_name}_{timestamp}.md"

    if args.output:
        report_path = Path(args.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(report, encoding="utf-8")
    print(f"\n  报告已保存：{report_path}")

    # 快速汇总
    print("\n  " + "=" * 50)
    print("  快速汇总")
    print("  " + "=" * 50)

    cons_stddevs = [r["stats"]["stddev"] for r in all_results
                    if r["_is_consistency"] and r["stats"]["n"] > 1 and r["stats"]["mean"] > 0]
    if cons_stddevs:
        print(f"  一致性（stddev）：平均 {statistics.mean(cons_stddevs):.2f}")
        print(f"  最大 {max(cons_stddevs):.2f}  最小 {min(cons_stddevs):.2f}")

    all_diffs = []
    for r in all_results:
        c = r["case"]
        exp = c.get("expected_score")
        if exp and r["stats"]["mean"] > 0:
            all_diffs.append(abs(r["stats"]["mean"] - exp))

    if all_diffs:
        print(f"  准确性（MAE）：平均偏差 {statistics.mean(all_diffs):.2f}")
        print(f"  最大偏差 {max(all_diffs):.2f}")

    print(f"  用时：{elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
