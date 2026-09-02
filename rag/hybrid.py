"""混合检索 — BM25 + 向量检索加权融合

BM25 保底：高频专有名词（GMP、defer）精确命中。
向量补语义：同义表达（goroutine调度 → GMP模型）召回。
权重：0.7 BM25 + 0.3 向量（benchmark/run_rag.py 权重扫描实测：Hit@3 最优区间 0.6~0.8）。

降级策略：向量检索异常时自动回退到纯 BM25。
"""

from rag.bm25 import BM25Index
from rag.vector import VectorIndex


def _minmax(seq: list[float]) -> list[float]:
    """Min-max 归一化到 [0, 1]"""
    if not seq:
        return seq
    mn, mx = min(seq), max(seq)
    if mx == mn:
        return [1.0] * len(seq)
    return [(v - mn) / (mx - mn) for v in seq]


def _normalize(results: list[tuple[int, str, float]]
               ) -> list[tuple[int, str, float]]:
    scores = _minmax([r[2] for r in results])
    return [(r[0], r[1], scores[i]) for i, r in enumerate(results)]


class HybridRetriever:
    """混合检索器 — BM25 保底 + 向量补语义"""

    def __init__(self, bm25_weight: float = 0.7):
        self.bm25 = BM25Index()
        self.vector = VectorIndex()
        self.bm25_weight = bm25_weight
        self.docs: list[str] = []
        self._built = False

    def build(self, docs: list[str],
              embeddings: list[list[float]] | None = None):
        """同时构建 BM25 和向量索引"""
        self.docs = docs
        if docs:
            self.bm25.build(docs)
            self.vector.build(docs, embeddings)
        self._built = True

    def search(self, query: str, top_k: int = 5
               ) -> list[tuple[int, str, float]]:
        """混合检索

        1. BM25 搜 top_k × 2（扩大候选池）
        2. 向量搜 top_k × 2
        3. 分别 min-max 归一化
        4. 加权融合 → 取 top_k
        """
        if not self._built or not self.docs:
            return []

        # 扩大候选池
        bm25_res = self.bm25.search(query, top_k * 2)
        vec_res = self.vector.search(query, top_k * 2)

        # 归一化
        bm25_norm = _normalize(bm25_res) if bm25_res else []
        vec_norm = _normalize(vec_res) if vec_res else []

        # 纯向量或纯 BM25 降级
        if not bm25_norm and not vec_norm:
            return []
        if not bm25_norm:
            return _normalize(vec_res)[:top_k]
        if not vec_norm:
            return _normalize(bm25_res)[:top_k]

        # 加权融合
        w_bm25 = self.bm25_weight
        w_vec = 1 - w_bm25
        fused: dict[int, float] = {}
        doc_map: dict[int, str] = {}

        for idx, doc, score in bm25_norm:
            fused[idx] = score * w_bm25
            doc_map[idx] = doc
        for idx, doc, score in vec_norm:
            fused[idx] = fused.get(idx, 0) + score * w_vec
            if idx not in doc_map:
                doc_map[idx] = doc

        sorted_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [(idx, doc_map[idx], score)
                for idx, score in sorted_items[:top_k]]
