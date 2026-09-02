"""BM25 检索 — 基于 rank_bm25 库

BM25 公式由库实现，我们了解原理就行。
这里只做封装：分词（jieba）+ 调 BM25Okapi。
"""

import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return jieba.lcut(text.lower())


class BM25Index:
    """BM25 索引 — 构建一次，多次检索"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[str] = []
        self._index: BM25Okapi | None = None

    def build(self, docs: list[str]):
        """构建索引"""
        self.docs = docs
        tokenized = [_tokenize(d) for d in docs]
        self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, str, float]]:
        """检索，返回 [(文档索引, 文档内容, 分数), ...]"""
        if not self._index or not self.docs:
            return []

        query_tokens = _tokenize(query)
        scores = self._index.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append((i, self.docs[i], scores[i]))
        return results
