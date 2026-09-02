"""向量检索 — 基于 sentence-transformers + numpy 余弦相似度

数据量 < 200 条/session，numpy 暴力搜索足够。
不上向量数据库，不上 FAISS——当前量级不需要。

用法：
    index = VectorIndex()
    index.build(docs)                          # 自动编码
    results = index.search("goroutine 调度")    # 语义检索
"""

import numpy as np

# 全局缓存 embedding 模型（只加载一次）
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            "BAAI/bge-small-zh-v1.5",
            local_files_only=True,  # 强制离线——模型已缓存，不连 huggingface.co
        )
    return _model


class VectorIndex:
    """向量索引 — 构建一次，多次检索"""

    def __init__(self):
        self.docs: list[str] = []
        self._embeddings: np.ndarray | None = None

    def build(self, docs: list[str],
              embeddings: list[list[float]] | None = None):
        """构建索引

        Args:
            docs: 文档内容列表
            embeddings: 可选，预计算向量（从 DB 读出）。
                        为 None 时自动用模型编码。
        """
        self.docs = docs
        if embeddings is not None:
            self._embeddings = np.array(embeddings, dtype=np.float32)
        else:
            model = _get_model()
            self._embeddings = model.encode(
                docs, normalize_embeddings=True, show_progress_bar=False
            )

    def get_embeddings(self) -> np.ndarray | None:
        """获取当前索引中的所有向量（用于持久化回 DB）"""
        return self._embeddings

    def search(self, query: str, top_k: int = 5
               ) -> list[tuple[int, str, float]]:
        """检索，返回 [(文档索引, 文档内容, 余弦相似度), ...]"""
        if self._embeddings is None or not self.docs:
            return []

        model = _get_model()
        q_vec = model.encode(query, normalize_embeddings=True,
                             show_progress_bar=False)
        scores = np.dot(self._embeddings, q_vec)  # 归一化后点积=余弦
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(i), self.docs[int(i)], float(scores[i]))
            for i in top_indices if scores[i] > 0
        ]
