"""Vector store (Qdrant) client and retriever — v3: MMR diversity + multi-query merge."""

import math
from typing import List
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import (
    QDRANT_PATH, QDRANT_COLLECTION, QDRANT_TOP_K_DEFAULT,
    MMR_ENABLED, MMR_LAMBDA, MMR_CANDIDATES,
    MAX_CONTEXT_SOURCES,
)
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
    for prefix, rank in JOURNAL_PREFIX_RANK.items():
        if source.startswith(prefix):
            return rank
    return _DEFAULT_RANK


def _rerank_by_journal(docs, top_k: int):
    if not docs:
        return docs
    scored = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        rank = _get_journal_rank_from_source(source)
        weight = JOURNAL_RANK_WEIGHTS.get(rank, 1.0)
        score = getattr(doc, "score", None)
        if score is not None:
            weighted_score = score * weight
        else:
            weighted_score = weight
        scored.append((doc, weighted_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# ── MMR (Maximal Marginal Relevance) ──

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mmr_rerank(docs, query_embedding: List[float], top_k: int,
               lambda_: float = 0.6, candidates: int = 30) -> List:
    """Maximal Marginal Relevance: balance relevance + diversity.
    
    MMR = argmax [ λ * sim(query, doc) - (1-λ) * max sim(doc, selected) ]
    
    λ close to 1.0 = relevance-focused, λ close to 0.3 = diversity-focused.
    """
    if not docs:
        return docs
    
    # Score docs by relevance (from Qdrant score or compute)
    scored = []
    for doc in docs:
        score = getattr(doc, "score", 0.5)
        if score is None:
            score = 0.5
        scored.append((doc, score))
    
    # Sort by relevance, take top N candidates
    scored.sort(key=lambda x: x[1], reverse=True)
    candidates_list = [doc for doc, _ in scored[:candidates]]
    
    if len(candidates_list) <= top_k:
        # If fewer candidates than needed, skip MMR
        return candidates_list
    
    selected = []
    remaining = list(candidates_list)
    
    # Start with the most relevant doc
    selected.append(remaining.pop(0))
    
    while len(selected) < top_k and remaining:
        # Compute MMR score for each remaining doc
        mmr_scores = []
        for doc in remaining:
            # Relevance: use the document score
            relevance = getattr(doc, "score", 0.5) or 0.5
            
            # Diversity: max similarity to already selected docs
            doc_vec = getattr(doc, "embedding", None)
            if doc_vec is None:
                # Fallback if no embedding stored in the doc
                max_sim = 0
            else:
                max_sim = max(
                    cosine_similarity(doc_vec, getattr(s, "embedding", [0.0]))
                    for s in selected
                )
            
            mmr = lambda_ * relevance - (1 - lambda_) * max_sim
            mmr_scores.append(mmr)
        
        # Pick the one with highest MMR
        best_idx = max(range(len(remaining)), key=lambda i: mmr_scores[i])
        selected.append(remaining.pop(best_idx))
    
    return selected


def merge_and_deduplicate(doc_lists: List[List]) -> List:
    """Merge multiple result lists, deduplicate by source text hash."""
    seen = set()
    merged = []
    for docs in doc_lists:
        for doc in docs:
            text = doc.page_content[:200]  # compare first 200 chars
            h = hash(text)
            if h not in seen:
                seen.add(h)
                merged.append(doc)
    return merged


# ── Retriever factory ──

def get_retriever(top_k: int = QDRANT_TOP_K_DEFAULT, rank_boost: bool = True,
                  use_mmr: bool = False):
    """Get a retriever.
    
    Args:
        top_k: 检索数量（0 = 不检索）
        rank_boost: 是否按期刊优先级重排序
        use_mmr: 是否启用 MMR 多样性重排
    """
    if top_k <= 0:
        return _NoRetriever()
    
    k_retrieve = max(top_k, MMR_CANDIDATES) if use_mmr else max(top_k, 20)
    
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": k_retrieve}
    )

    if use_mmr:
        return _MMRRetriever(base_retriever, target_k=top_k, lambda_=MMR_LAMBDA)
    if rank_boost:
        return _RankBoostedRetriever(base_retriever, target_k=top_k)
    return base_retriever


# ── Multi-query retriever ──

def multi_query_retrieve(queries: List[str], top_k_per_query: int = 8,
                         use_mmr: bool = True) -> List:
    """Run multiple queries in parallel and merge results.
    
    Args:
        queries: list of query strings
        top_k_per_query: how many results per query
        use_mmr: apply MMR to the merged result set
        
    Returns:
        merged, deduplicated, ranked list of docs
    """
    all_results = []
    for q in queries:
        retriever = get_retriever(top_k=top_k_per_query, rank_boost=True, use_mmr=False)
        docs = retriever.invoke(q)
        all_results.append(docs)
    
    merged = merge_and_deduplicate(all_results)
    
    if use_mmr and len(merged) > MAX_CONTEXT_SOURCES:
        # We don't have query_embedding here, fallback to journal ranking
        merged = _rerank_by_journal(merged, MAX_CONTEXT_SOURCES)
    elif len(merged) > MAX_CONTEXT_SOURCES:
        merged = merged[:MAX_CONTEXT_SOURCES]
    
    return merged


class _NoRetriever:
    def invoke(self, query):
        return []
    def get_relevant_documents(self, query):
        return []


class _RankBoostedRetriever:
    def __init__(self, base_retriever, target_k: int = 10):
        self._base = base_retriever
        self._target_k = target_k
    def invoke(self, query):
        docs = self._base.invoke(query)
        return _rerank_by_journal(docs, self._target_k)
    def get_relevant_documents(self, query):
        return self.invoke(query)


class _MMRRetriever:
    """MMR 多样性重排检索器。"""
    def __init__(self, base_retriever, target_k: int = 10, lambda_: float = 0.6):
        self._base = base_retriever
        self._target_k = target_k
        self._lambda = lambda_
    def invoke(self, query):
        docs = self._base.invoke(query)
        # Use query embedding for MMR
        query_vec = embed_model.embed_query(query)
        return mmr_rerank(docs, query_vec, self._target_k, self._lambda)
    def get_relevant_documents(self, query):
        return self.invoke(query)
