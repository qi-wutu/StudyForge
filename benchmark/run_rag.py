"""RAG 检索评测 — 评估混合检索器（BM25 + 向量）的召回质量

和 benchmark/run.py 同一套评测哲学，但评测对象不同：
  - run.py     → 评测 LLM 判分（judge）准不准
  - run_rag.py → 评测检索（retrieve）准不准

评测对象是 rag/ 下的真实检索器（不是重新实现）：
  - BM25Index       纯词面检索（jieba 分词 + BM25）
  - VectorIndex     纯语义检索（bge-small-zh-v1.5 + 余弦）
  - HybridRetriever 混合检索（默认 0.7 BM25 + 0.3 向量）

评测集（自包含，不依赖数据库，本地跑零成本）：
  - corpus：benchmark/cases/*.json 里 93 道题的标准答案 → 当作 93 个"知识点"
  - 查询分两类：
      A) 93 道题的原问题          → kind=lexical（关键词重叠高，测词面检索）
      B) cases_rag/hard_queries.json 的语义改写 → kind=semantic（关键词重叠低，测语义检索）

指标（top-k = 1/3/5）：
  - Hit@k / Recall@k  相关知识点是否被召回到前 k
  - MRR@k             首个相关知识点位置倒数的均值
  - 按 kind 分层 + 三路检索器对比
  - 额外：bm25_weight 全扫 0→1，验证默认 0.7/0.3 是不是最优

用法：
  cd StudyForge
  python benchmark/run_rag.py                   # 全量
  python benchmark/run_rag.py --cases golang    # 只跑某个主题
  python benchmark/run_rag.py --no-sweep        # 跳过权重扫描
  python benchmark/run_rag.py --topk 5          # 只看 top-5（可多个：--topk 1 3 5）
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 让 rag/ 可以直接 import

from rag.bm25 import BM25Index
from rag.vector import VectorIndex
from rag.hybrid import HybridRetriever

CASES_DIR = ROOT / "benchmark" / "cases"
RAG_CASES_DIR = ROOT / "benchmark" / "cases_rag"
REPORTS_DIR = ROOT / "benchmark" / "reports"

# 生产默认权重（rag/hybrid.py 的默认值），报告用它跟扫描结果对比
PROD_BM25_WEIGHT = 0.7


# ============================================================
# 数据加载
# ============================================================

def load_corpus(case_stems: list[str]) -> list[dict]:
    """从 cases/*.json 构建语料库。

    一个"知识点" = 一道题的 standard_answer（完整作答，覆盖该主题核心内容）。
    与生产一致，检索索引只存 content，不拼 title。
    """
    corpus: list[dict] = []
    seen: set[str] = set()
    for path in sorted(CASES_DIR.glob("*.json")):
        if case_stems and path.stem not in case_stems:
            continue
        cases = json.loads(path.read_text(encoding="utf-8"))
        for c in cases:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            corpus.append({
                "id": c["id"],
                "topic": c["topic"],
                "content": c["standard_answer"],
            })
    return corpus


def load_queries(corpus: list[dict], case_stems: list[str]) -> list[dict]:
    """构造查询集：
    A) 每道题的原问题 → kind=lexical，相关知识点 = 自己
    B) hard_queries.json 的语义改写 → kind=semantic，相关知识点 = 手工标注
    """
    corpus_ids = {d["id"] for d in corpus}
    queries: list[dict] = []

    for path in sorted(CASES_DIR.glob("*.json")):
        if case_stems and path.stem not in case_stems:
            continue
        cases = json.loads(path.read_text(encoding="utf-8"))
        for c in cases:
            queries.append({
                "id": c["id"],
                "topic": c["topic"],
                "kind": "lexical",
                "query": c["question"],
                "relevant_ids": [c["id"]],
            })

    hard_path = RAG_CASES_DIR / "hard_queries.json"
    if hard_path.exists():
        for hq in json.loads(hard_path.read_text(encoding="utf-8")):
            if case_stems and hq.get("topic") not in case_stems:
                continue
            bad = [r for r in hq.get("relevant_ids", []) if r not in corpus_ids]
            if bad:
                print(f"  [跳过] hard query {hq['id']} 标注的知识点 {bad} 不在本次语料中")
                continue
            queries.append({
                "id": hq["id"],
                "topic": hq.get("topic", ""),
                "kind": "semantic",
                "query": hq["query"],
                "relevant_ids": hq["relevant_ids"],
            })
    return queries


# ============================================================
# 索引构建
# ============================================================

def build_indexes(docs: list[str]):
    """构建三路检索器。

    向量只编码一次，混合检索器复用同一份 embedding，避免反复编码。
    向量模型缺失时降级：返回 vector_ok=False，只评测 BM25。
    """
    bm25 = BM25Index()
    bm25.build(docs)

    vec = VectorIndex()
    try:
        vec.build(docs)  # 只编码一次（bge-small-zh-v1.5）
    except Exception as e:
        print(f"  [警告] 向量模型不可用（{e.__class__.__name__}），降级为纯 BM25 评测")
        print(f"         请确认模型已缓存：BAAI/bge-small-zh-v1.5")
        return bm25, None, None, None, False

    embeddings = vec.get_embeddings()
    hybrid = HybridRetriever(bm25_weight=PROD_BM25_WEIGHT)
    hybrid.build(docs, embeddings)

    return bm25, vec, hybrid, embeddings, True


# ============================================================
# 指标计算
# ============================================================

def evaluate(search, queries: list[dict], doc_index: dict, topk_list: list[int]):
    """跑一遍检索，返回每个查询的召回结果。

    search: callable(query, top_k) -> [(idx, doc, score)]
    doc_index: {doc_id: idx}
    """
    max_k = max(topk_list)
    rows = []
    for q in queries:
        rel = {doc_index[r] for r in q["relevant_ids"] if r in doc_index}
        results = search(q["query"], max_k)
        retrieved = [idx for idx, _doc, _score in results]
        rows.append({
            "id": q["id"],
            "kind": q["kind"],
            "query": q["query"],
            "rel": rel,
            "retrieved": retrieved,
        })
    return rows


def compute_metrics(rows: list[dict], topk_list: list[int]):
    """汇总指标：Hit@k（相关在不在前 k）+ MRR@k（首个相关位置倒数）。

    Returns:
        (overall, stratified)：
          overall     = {k: {"hit": float, "mrr": float}}
          stratified  = {kind: {k: {"hit": float, "mrr": float, "n": int}}}
    """
    n = len(rows)
    overall = {k: {"hit": 0.0, "mrr": 0.0} for k in topk_list}
    kinds = sorted({r["kind"] for r in rows})
    stratified = {kind: {k: {"hit": 0.0, "mrr": 0.0, "n": 0} for k in topk_list}
                  for kind in kinds}

    for r in rows:
        rel = r["rel"]
        if not rel:
            continue
        for k in topk_list:
            top = r["retrieved"][:k]
            hit = 1.0 if rel & set(top) else 0.0
            mrr = 0.0
            for pos, idx in enumerate(top, start=1):
                if idx in rel:
                    mrr = 1.0 / pos
                    break
            overall[k]["hit"] += hit
            overall[k]["mrr"] += mrr
            sk = stratified[r["kind"]][k]
            sk["hit"] += hit
            sk["mrr"] += mrr
            sk["n"] += 1

    for k in topk_list:
        overall[k]["hit"] /= n
        overall[k]["mrr"] /= n
    for kind in kinds:
        for k in topk_list:
            if stratified[kind][k]["n"]:
                stratified[kind][k]["hit"] /= stratified[kind][k]["n"]
                stratified[kind][k]["mrr"] /= stratified[kind][k]["n"]

    return overall, stratified


def sweep(docs: list[str], embeddings, queries: list[dict],
          doc_index: dict, top_k: int) -> tuple[list[dict], dict, dict]:
    """扫描 bm25_weight ∈ [0, 1]，看哪个权重最优。

    复用同一份 embedding，只重建 BM25（jieba 分词很快），
    验证默认 0.7/0.3 是不是真的最优。
    """
    results = []
    for i in range(11):
        w = i / 10.0
        retriever = HybridRetriever(bm25_weight=w)
        retriever.build(docs, embeddings)
        rows = evaluate(retriever.search, queries, doc_index, [top_k])
        overall, _ = compute_metrics(rows, [top_k])
        results.append({
            "w": w,
            "hit": overall[top_k]["hit"],
            "mrr": overall[top_k]["mrr"],
        })
    best_hit = max(results, key=lambda r: r["hit"])
    best_mrr = max(results, key=lambda r: r["mrr"])
    return results, best_hit, best_mrr


# ============================================================
# 报告
# ============================================================

def generate_report(data: dict) -> str:
    c = data["corpus"]
    q = data["queries"]
    topk_list = data["topk_list"]
    L = [f"# RAG 检索评测报告\n"]
    L.append(f"> 评测对象：StudyForge 混合检索器（BM25 {PROD_BM25_WEIGHT} + 向量 {1 - PROD_BM25_WEIGHT}）")
    L.append(f"> 数据集：{len(c)} 个知识点语料，{len(q)} 条查询（词面 {data['n_lex']} + 语义 {data['n_sem']}）")
    L.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 一、概况
    L.append("## 一、评测概况\n")
    L.append("| 项目 | 数据 |")
    L.append("|---|---|")
    L.append(f"| 语料知识点 | {len(c)} |")
    L.append(f"| 查询总数 | {len(q)}（词面 {data['n_lex']} + 语义 {data['n_sem']}） |")
    L.append("| 检索器 | BM25 / 向量 / 混合(0.7) |")
    L.append("| 向量模型 | BAAI/bge-small-zh-v1.5 |")
    L.append(f"| top-k | {' / '.join(map(str, topk_list))} |")
    L.append(f"| 评测用时 | {data['elapsed']:.1f}s |\n")

    # 二、三路对比
    maxk = max(topk_list)
    L.append("## 二、三路检索器对比（全体查询）\n")
    hdr = "| 检索器 |" + "".join(f" Hit@{k} |" for k in topk_list) + f" MRR@{maxk} |"
    sep = "|---|" + "---|" * len(topk_list) + "---|"
    L.append(hdr)
    L.append(sep)
    for name, res in data["results"].items():
        cells = "|".join(f" {res['overall'][k]['hit']:.3f} " for k in topk_list)
        L.append(f"| {name} | {cells} | {res['overall'][maxk]['mrr']:.3f} |")
    L.append("")

    # 三、按类型分层
    L.append("## 三、按查询类型分层\n")
    kind_names = {"lexical": "词面查询（原问题，关键词重叠高）", "semantic": "语义查询（改写题，关键词重叠低）"}
    # 各检索器的 kind 集合一致，用第一个的即可
    kinds = sorted(data["results"][list(data["results"].keys())[0]]["stratified"].keys())
    for kind in kinds:
        name = kind_names.get(kind, kind)
        L.append(f"### {name}（{data['strat_n'][kind]} 条）\n")
        L.append(hdr)
        L.append(sep)
        for rname, res in data["results"].items():
            sk = res["stratified"][kind]
            cells = "|".join(f" {sk[k]['hit']:.3f} " for k in topk_list)
            L.append(f"| {rname} | {cells} | {sk[maxk]['mrr']:.3f} |")
        L.append("")

    # 四、权重扫描
    if data["sweep"] is not None:
        L.append("## 四、bm25_weight 权重扫描（top-k={}）\n".format(data["sweep_k"]))
        L.append(f"| bm25_weight | Hit@{data['sweep_k']} | MRR@{data['sweep_k']} |")
        L.append("|---|--:|--:|")
        for r in data["sweep"]:
            mark = ""
            if r["w"] == data["best_hit"]["w"]:
                mark = "  ← 最优 Hit"
            elif r["w"] == data["best_mrr"]["w"]:
                mark = "  ← 最优 MRR"
            L.append(f"| {r['w']:.1f} | {r['hit']:.3f} | {r['mrr']:.3f} |{mark}")
        L.append("")
        b = data["best_hit"]
        L.append(f"**最优权重**：bm25_weight = {b['w']:.1f}（Hit@{data['sweep_k']} = {b['hit']:.3f}）\n")
        if abs(b["w"] - PROD_BM25_WEIGHT) < 0.05:
            L.append(f"> 结论：默认权重 {PROD_BM25_WEIGHT} 正好落在最优区间，生产配置无需调整。\n")
        else:
            L.append(f"> 结论：实测最优是 {b['w']:.1f}，与生产默认 {PROD_BM25_WEIGHT} 不同，建议按需调整。\n")

    # 五、最难查询
    sec = 4
    missed = data["missed"]
    if missed:
        sec += 1
        L.append(f"## {sec}、混合检索 top-{maxk} 召不回的查询（{len(missed)} 条）\n")
        L.append("| 查询 | 相关知识点 | 类型 |")
        L.append("|---|---|---|")
        for r in missed[:10]:
            rel_ids = ", ".join(str(x) for x in sorted(r["rel"]))
            L.append(f"| {r['query'][:60]}... | {rel_ids} | {r['kind']} |")
        L.append("")

    # 六、结论
    sec += 1
    L.append(f"## {sec}、结论\n")
    names = list(data["results"].keys())
    maxk = max(topk_list)
    k_comp = 3 if 3 in topk_list else min(topk_list)
    best_name = max(names, key=lambda nm: data["results"][nm]["overall"][maxk]["hit"])
    L.append(f"- 全体查询里召回最好的是 **{best_name}**（Hit@{maxk} = {data['results'][best_name]['overall'][maxk]['hit']:.3f}）。")
    if "混合(0.7)" in names and "BM25" in names:
        hy = data["results"]["混合(0.7)"]["overall"][maxk]["hit"]
        bm = data["results"]["BM25"]["overall"][maxk]["hit"]
        if hy >= bm:
            L.append(f"- 混合检索整体 Hit@{maxk} = {hy:.3f}，不输纯 BM25 的 {bm:.3f}，且词面/语义两类的具体表现见分层表。")
        else:
            L.append(f"- 混合检索整体 Hit@{maxk} = {hy:.3f}，略低于纯 BM25 的 {bm:.3f}，说明混合的权重还可以再调。")
    if "向量" in names:
        vec_sem = data["results"]["向量"]["stratified"]["semantic"][k_comp]["hit"]
        bm_sem = data["results"]["BM25"]["stratified"]["semantic"][k_comp]["hit"]
        if vec_sem > bm_sem:
            L.append(f"- 语义查询（换说法）上，向量 Hit@{k_comp} = {vec_sem:.3f} vs BM25 {bm_sem:.3f}——纯词面检索在改写的场景会掉，语义检索是必要补充。")
        else:
            L.append(f"- 语义查询（换说法）上，向量 Hit@{k_comp} = {vec_sem:.3f} vs BM25 {bm_sem:.3f}——当前语义集 BM25 足够，说明改写查询仍保留了词面线索。")
    L.append("")
    return "\n".join(L)


# ============================================================
# 入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="RAG 检索评测")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="只跑指定主题（如 golang databases）")
    ap.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5],
                    help="评估的 top-k 集合（可多个）")
    ap.add_argument("--no-sweep", action="store_true",
                    help="跳过 bm25_weight 权重扫描")
    args = ap.parse_args()

    topk_list = sorted(set(args.topk))
    case_stems = args.cases

    t0 = time.time()

    corpus = load_corpus(case_stems)
    if not corpus:
        print("  错误：没有加载到任何知识点，检查 benchmark/cases/*.json")
        sys.exit(1)
    queries = load_queries(corpus, case_stems)
    if not queries:
        print("  错误：没有加载到任何查询")
        sys.exit(1)

    n_lex = sum(1 for q in queries if q["kind"] == "lexical")
    n_sem = sum(1 for q in queries if q["kind"] == "semantic")

    docs = [d["content"] for d in corpus]
    doc_index = {d["id"]: i for i, d in enumerate(corpus)}

    print(f"\n  RAG 检索评测")
    print(f"  语料：{len(corpus)} 个知识点")
    print(f"  查询：{len(queries)} 条（词面 {n_lex} + 语义 {n_sem}）")
    print(f"  top-k：{' / '.join(map(str, topk_list))}")
    print(f"  {'=' * 50}\n")

    bm25, vec, hybrid, embeddings, vector_ok = build_indexes(docs)

    retrievers = [("BM25", bm25.search)]
    if vector_ok:
        retrievers.append(("向量", vec.search))
        retrievers.append(("混合(0.7)", hybrid.search))

    results = {}
    maxk = max(topk_list)
    for name, search in retrievers:
        rows = evaluate(search, queries, doc_index, topk_list)
        overall, stratified = compute_metrics(rows, topk_list)
        results[name] = {"rows": rows, "overall": overall, "stratified": stratified}
        line = f"  {name.ljust(8)}"
        for k in topk_list:
            line += f"  Hit@{k} {overall[k]['hit']:.3f}"
        line += f"  MRR@{maxk} {overall[maxk]['mrr']:.3f}"
        print(line)

    # 权重扫描（用列表里的 k，优先 3）
    sweep_res = best_hit = best_mrr = None
    sweep_k = 3 if 3 in topk_list else topk_list[0]
    if not args.no_sweep and vector_ok:
        print(f"\n  权重扫描中（bm25_weight 0.0 -> 1.0，top-k={sweep_k}）...")
        sweep_res, best_hit, best_mrr = sweep(docs, embeddings, queries, doc_index, sweep_k)
        print(f"  最优 bm25_weight = {best_hit['w']:.1f}（Hit@{sweep_k} = {best_hit['hit']:.3f}）")

    # 最难查询：混合检索 top-maxk 召不回
    missed = []
    if "混合(0.7)" in results:
        missed = [r for r in results["混合(0.7)"]["rows"]
                  if r["rel"] and not (r["rel"] & set(r["retrieved"][:maxk]))]

    elapsed = time.time() - t0

    data = {
        "corpus": corpus, "queries": queries,
        "n_lex": n_lex, "n_sem": n_sem,
        "topk_list": topk_list, "elapsed": elapsed,
        "results": results,
        "strat_n": {kind: sum(1 for q in queries if q["kind"] == kind)
                    for kind in ("lexical", "semantic")},
        "sweep": sweep_res, "sweep_k": sweep_k,
        "best_hit": best_hit, "best_mrr": best_mrr,
        "missed": missed,
    }
    report = generate_report(data)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "rag_" + ("_".join(case_stems) if case_stems else "all")
    report_path = REPORTS_DIR / f"{name}_{ts}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n  评测完成！总用时：{elapsed:.1f}s")
    print(f"  报告已保存：{report_path}")
    print()


if __name__ == "__main__":
    main()
