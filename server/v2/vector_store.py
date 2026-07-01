"""Vector store (Qdrant) client and retriever — supports journal-rank re-ranking."""

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from config import QDRANT_PATH, QDRANT_COLLECTION, QDRANT_TOP_K_DEFAULT
from models import embed_model

# ── Qdrant client (local mode) ──
client = QdrantClient(path=QDRANT_PATH)

# ── Vector store ──
vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embed_model,
)

# ── 期刊排名权重（rank 1~7，数字越小优先级越高） ──
JOURNAL_RANK_WEIGHTS = {
    1: 1.5,   # Nature
    2: 1.4,   # Nature Energy
    3: 1.3,   # Nature Materials
    4: 1.2,   # Nature Photonics
    5: 1.15,  # Nature Nanotechnology
    6: 1.05,  # Nature Communications
    7: 1.0,   # Other / arXiv
}


# 文件名前缀 → journal_rank 映射
JOURNAL_PREFIX_RANK = {
    "Nature_": 1,
    "NatEnergy_": 2,
    "NatMater_": 3,
    "NatPhoton_": 4,
    "NatNanotech_": 5,
    "NatComm_": 6,
}
_DEFAULT_RANK = 7


def _get_journal_rank_from_source(source: str) -> int:
    """从 source 文件名前缀解析 journal_rank。"""
    for prefix, rank in JOURNAL_PREFIX_RANK.items():
        if source.startswith(prefix):
            return rank
    return _DEFAULT_RANK


def _rerank_by_journal(docs, top_k: int):
    """对检索结果按期刊权重排序。
    
    权重公式: final_score = semantic_score * journal_weight
    
    同权重的文档保持原有语义排序，不跨权重打乱过多。
    """
    if not docs:
        return docs

    scored = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        rank = _get_journal_rank_from_source(source)
        weight = JOURNAL_RANK_WEIGHTS.get(rank, 1.0)
        
        # 如果 doc 有 score 属性（Qdrant 返回的 relevance score）
        score = getattr(doc, "score", None)
        if score is not None:
            weighted_score = score * weight
        else:
            weighted_score = weight  # 降级：仅按期刊权重排序
        
        scored.append((doc, weighted_score))
    
    # 按加权分数降序排列
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# ── Retriever factory (dynamic top_k) ──
def get_retriever(top_k: int = QDRANT_TOP_K_DEFAULT, rank_boost: bool = True):
    """Get a retriever that fetches `top_k` results.
    
    Args:
        top_k: 检索数量（0 = 不检索）
        rank_boost: 是否按期刊优先级重排序（默认开启）
    """
    if top_k <= 0:
        return _NoRetriever()
    
    k_retrieve = max(top_k, 20) if rank_boost else top_k  # 多取一些用于重排序
    
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": k_retrieve}
    )

    if rank_boost:
        return _RankBoostedRetriever(base_retriever, target_k=top_k)
    return base_retriever


class _NoRetriever:
    """空检索器：top_k=0 时使用，永远返回空列表。"""
    def invoke(self, query):
        return []

    def get_relevant_documents(self, query):
        return []


class _RankBoostedRetriever:
    """带期刊排名的检索器包装。"""
    
    def __init__(self, base_retriever, target_k: int = 10):
        self._base = base_retriever
        self._target_k = target_k
    
    def invoke(self, query):
        docs = self._base.invoke(query)
        return _rerank_by_journal(docs, self._target_k)
    
    def get_relevant_documents(self, query):
        return self.invoke(query)
