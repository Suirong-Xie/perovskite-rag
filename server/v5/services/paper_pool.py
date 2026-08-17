"""
Session-level Paper Pool — 跨轮复用的论文元数据池。

每篇论文在池中只存一次，跨轮累积。支持：
  - 合并搜索结果（按 source 去重）
  - 标记已读/无全文
  - 启发式选择代表性论文（不调 LLM）
  - 生成供 prompt 注入的摘要
  - 话题漂移检测
  - JSON 序列化（persist 到 sessions/{sid}/paper_pool.json）
"""

from __future__ import annotations

import re
import time
from typing import Optional

# ── 论文分类关键词（与 agent_sm.py 保持一致）──

_REVIEW_TITLE_KW = [
    "review", "progress", "advances", "survey", "overview",
    "comprehensive", "perspective", "roadmap", "retrospect",
    "tutorial", "state of the art", "state-of-the-art",
    "critical review", "mini review", "recent progress",
    "recent advances", "current status", "this review",
    "we review", "summarizes recent", "overview of",
    "综述", "进展", "回顾", "概述", "研究进展", "研究现状",
]

# 期刊排名（越高越权威）
_JOURNAL_RANK = {
    "Nature": 10, "Science": 10,
    "Nature Energy": 9, "Nature Materials": 9, "Nature Photonics": 9,
    "Nature Nanotechnology": 9, "Nature Chemistry": 9, "Nature Physics": 9,
    "Nature Communications": 8, "Science Advances": 8,
    "Joule": 8, "Energy & Environmental Science": 8,
    "Advanced Materials": 7, "Advanced Energy Materials": 7,
    "Advanced Functional Materials": 7,
    "ACS Energy Letters": 7, "Nano Letters": 7, "ACS Nano": 7,
    "Angewandte Chemie": 7, "Journal of the American Chemical Society": 7,
    "Nano Energy": 6, "Chemistry of Materials": 6,
    "Journal of Materials Chemistry A": 6,
    "ACS Applied Materials & Interfaces": 5,
    "Journal of Physical Chemistry": 5,
    "Solar RRL": 5, "Advanced Science": 6,
    "Small": 5, "Nanoscale": 5,
    "RSC Advances": 3, "Scientific Reports": 4,
    "Other": 3,
}


def _classify_paper_type(source: str, meta: dict) -> str:
    """判断综述还是实验论文。"""
    title = (meta.get("title", "") or "").lower()
    content = (meta.get("content_preview", "") or "").lower()
    source_lower = (source or "").lower()
    for kw in _REVIEW_TITLE_KW:
        kw_lower = kw.lower()
        if kw_lower in title or kw_lower in content or kw_lower in source_lower:
            return "review"
    return "experimental"


def _journal_score(journal: str) -> int:
    """返回期刊权威度分数（0-10）。"""
    if not journal:
        return 3
    # 精确匹配
    if journal in _JOURNAL_RANK:
        return _JOURNAL_RANK[journal]
    # 模糊匹配
    for name, rank in _JOURNAL_RANK.items():
        if name.lower() in journal.lower():
            return rank
    return 3


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """简单的 token 重叠度评分（0-1）。"""
    if not text_a or not text_b:
        return 0.0
    # 提取英文单词 + 中文字符
    tokens_a = set(re.findall(r'[a-zA-Z]{3,}|[一-鿿]+', text_a.lower()))
    tokens_b = set(re.findall(r'[a-zA-Z]{3,}|[一-鿿]+', text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    return len(overlap) / min(len(tokens_a), len(tokens_b))


# 测量/数据相关关键词 — 用于数据查询类问题的论文选择
_DATA_CONTENT_KW = [
    "thickness", "profilomet", "cross-section", "cross section",
    "SEM", "TEM", "AFM", "XRD", "XPS", "UV-Vis", "ellipsomet",
    "measured", "measurement", "nm thick", "µm thick", "nm,",
    "wt%", "at%", "concentration", "doping ratio",
    "PCE of", "efficiency of", "achieved", "obtained",
    "reported", "demonstrat", "exhibited", "showed a",
    "V_oc", "J_sc", "FF", "EQE",
    "厚度", "测量", "表征", "截面", "轮廓仪",
]


def _data_content_score(paper: dict) -> float:
    """评分论文是否包含测量数据/量化信息（0-1）。

    检查 content_preview、title 中是否出现测量相关关键词。
    用于数据查询类问题的论文优选。
    """
    text = (
        (paper.get("content_preview", "") or "")
        + " " + (paper.get("title", "") or "")
    ).lower()
    if not text.strip():
        return 0.0
    score = sum(1 for kw in _DATA_CONTENT_KW if kw.lower() in text)
    # 归一化：最高可能 ~10 个匹配 → 0-1 范围
    return min(score / 10.0, 1.0)


class SessionPaperPool:
    """会话级论文池。"""

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.papers: dict[str, dict] = {}       # source → paper record
        self.last_query: str = ""               # 最近一次搜索查询（用于话题漂移检测）
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    # ── 增删改查 ──

    def add_from_search(self, raw_results: list[dict], query: str = ""):
        """合并搜索结果到池中。"""
        if query:
            self.last_query = query
        for r in raw_results:
            src = r.get("source", "")
            if not src:
                continue
            if src in self.papers:
                # 更新已有记录（可能从不同搜索源获得更多元数据）
                existing = self.papers[src]
                if not existing.get("title") and r.get("title"):
                    existing["title"] = r.get("title", "")
                if not existing.get("year") and r.get("year"):
                    existing["year"] = str(r.get("year", ""))
                continue
            self.papers[src] = {
                "source": src,
                "title": r.get("title", ""),
                "journal": r.get("journal_name", "") or r.get("venue", ""),
                "year": str(r.get("year", "") or r.get("_s2_year", "")),
                "content_preview": (
                    r.get("content", "")
                    or r.get("abstract", "")
                    or r.get("summary", "")
                    or ""
                )[:500],
                "doi": r.get("_s2_doi", ""),
                "has_fulltext": r.get("has_pdf", True),  # 默认假设有全文
                "is_read": False,
                "read_content": None,
                "paper_type": _classify_paper_type(src, r),
                "journal_score": _journal_score(
                    r.get("journal_name", "") or r.get("venue", "")
                ),
            }
        self.updated_at = time.time()

    def mark_read(self, source: str, content: str = ""):
        """标记论文为已读。"""
        if source in self.papers:
            self.papers[source]["is_read"] = True
            self.papers[source]["read_content"] = content[:5000] if content else None
            self.papers[source]["has_fulltext"] = True
            self.updated_at = time.time()

    def mark_nofulltext(self, source: str):
        """标记论文无全文。"""
        if source in self.papers:
            self.papers[source]["has_fulltext"] = False
            self.papers[source]["is_read"] = True  # 不可读，标记为已处理
            self.updated_at = time.time()

    def get(self, source: str) -> dict | None:
        return self.papers.get(source)

    def get_unread(self) -> list[dict]:
        """返回所有未读且有全文的论文。"""
        return [
            p for p in self.papers.values()
            if not p["is_read"] and p["has_fulltext"]
        ]

    def get_read(self) -> list[dict]:
        """返回所有已读且有全文的论文。"""
        return [
            p for p in self.papers.values()
            if p["is_read"] and p["has_fulltext"]
        ]

    def get_nofulltext(self) -> list[dict]:
        """返回所有无全文的论文。"""
        return [
            p for p in self.papers.values()
            if not p["has_fulltext"]
        ]

    def get_by_type(self, paper_type: str) -> list[dict]:
        """按类型筛选（review / experimental）。"""
        return [
            p for p in self.papers.values()
            if p.get("paper_type") == paper_type
        ]

    # ── 代表性论文选择 ──

    def select_representative(
        self, n: int = 3, question: str = "", question_type: str = "broad",
        response_style: str = "",
    ) -> list[dict]:
        """选择 n 篇最有代表性的未读论文。

        Args:
            n: 最多选择篇数
            question: 用户原始问题（用于关键词重叠评分）
            question_type: "broad"（优先综述）或 "specific"（优先相关实验论文）
            response_style: 回答风格（data_lookup/how_to/compare/overview）

        Returns:
            按优先级排序的论文列表
        """
        unread = self.get_unread()
        if not unread:
            return []

        # 数据查询类：额外加分给含测量数据的论文
        is_data_lookup = (response_style == "data_lookup")

        if question_type == "broad":
            # 优先综述（最多 2 篇），再选实验论文
            reviews = [p for p in unread if p.get("paper_type") == "review"]
            experiments = [p for p in unread if p.get("paper_type") != "review"]
            reviews.sort(key=lambda p: p.get("journal_score", 0), reverse=True)
            selected = reviews[:2]  # 综述最多 2 篇
            # 剩余名额给实验论文
            remaining = n - len(selected)
            if remaining > 0:
                experiments.sort(
                    key=lambda p: (
                        _keyword_overlap(question, p.get("content_preview", "")) * 5
                        + p.get("journal_score", 0) * 0.5
                        + (_data_content_score(p) * 2 if is_data_lookup else 0)
                    ),
                    reverse=True,
                )
                seen_journals = {p.get("journal", "") for p in selected}
                for p in experiments:
                    if len(selected) >= n:
                        break
                    j = p.get("journal", "")
                    if j not in seen_journals or len(selected) >= n - 1:
                        selected.append(p)
                        seen_journals.add(j)
        else:
            # specific: 按关键词重叠度排序，跳过综述
            candidates = [p for p in unread if p.get("paper_type") != "review"]
            if len(candidates) < n:
                candidates = unread  # 如果实验论文不够，也包含综述
            candidates.sort(
                key=lambda p: (
                    _keyword_overlap(question, p.get("content_preview", "")) * 5
                    + p.get("journal_score", 0) * 0.3
                    + (_data_content_score(p) * 3 if is_data_lookup else 0)
                ),
                reverse=True,
            )
            # 期刊多样性
            selected = []
            seen_journals = set()
            for p in candidates:
                if len(selected) >= n:
                    break
                j = p.get("journal", "")
                if j not in seen_journals or len(selected) >= n - 1:
                    selected.append(p)
                    seen_journals.add(j)

        return selected[:n]

    def select_for_followup(
        self, n: int = 4, followup_question: str = "",
        response_style: str = "",
    ) -> list[dict]:
        """根据跟进问题选择最相关的论文。"""
        unread = self.get_unread()
        if not unread:
            return []

        is_data_lookup = (response_style == "data_lookup")
        unread.sort(
            key=lambda p: (
                _keyword_overlap(
                    followup_question, p.get("content_preview", "")
                ) * 5
                + (_data_content_score(p) * 3 if is_data_lookup else 0)
            ),
            reverse=True,
        )
        # 期刊多样性
        selected = []
        seen_journals = set()
        for p in unread:
            if len(selected) >= n:
                break
            j = p.get("journal", "")
            if j not in seen_journals or len(selected) >= n - 1:
                selected.append(p)
                seen_journals.add(j)
        return selected[:n]

    # ── 摘要生成 ──

    def summary_for_prompt(self, max_papers: int = 20) -> str:
        """生成供 Agent prompt 使用的论文摘要。"""
        unread = self.get_unread()
        read = self.get_read()
        noft = self.get_nofulltext()

        lines = [
            f"论文池: 共 {len(self.papers)} 篇",
            f"  未读全文: {len(unread)} 篇",
            f"  已读: {len(read)} 篇",
            f"  无全文: {len(noft)} 篇",
            "",
        ]

        if unread:
            # 按类型分组
            reviews = [p for p in unread if p.get("paper_type") == "review"]
            expts = [p for p in unread if p.get("paper_type") != "review"]

            if reviews:
                lines.append("### 📝 综述论文（优先阅读）")
                for p in reviews[:max_papers]:
                    lines.append(self._fmt_paper(p))
                lines.append("")

            if expts:
                lines.append("### 🔬 实验论文")
                for p in expts[:max_papers]:
                    lines.append(self._fmt_paper(p))
                lines.append("")

        if noft:
            lines.append("### 🔗 无全文（跳过）")
            for p in noft[:10]:
                lines.append(f"  - {p['source']}")
            lines.append("")

        return "\n".join(lines)

    def unread_summary_for_suggestions(self, max_papers: int = 10) -> str:
        """生成供建议生成的未读论文摘要。"""
        unread = self.get_unread()
        lines = []
        for p in unread[:max_papers]:
            lines.append(self._fmt_paper(p))
        return "\n".join(lines) if lines else "(无)"

    @staticmethod
    def _fmt_paper(p: dict) -> str:
        """格式化单篇论文条目。"""
        j = p.get("journal", "")
        y = p.get("year", "")
        pre = (p.get("content_preview", "") or "")[:120]
        yr_str = f"{y} | " if y else ""
        extra = f" — {yr_str}{j}" if j else ""
        line = f"  - `{p['source']}`{extra}"
        if pre:
            line += f"\n    _{pre}..._"
        return line

    # ── 话题漂移检测 ──

    def detect_topic_drift(self, new_query: str, threshold: float = 0.1) -> bool:
        """检测新查询是否与论文池话题漂移。

        如果漂移了，应该清空池重新搜索。
        """
        if not self.last_query or not self.papers:
            return False  # 空池，不算漂移
        overlap = _keyword_overlap(self.last_query, new_query)
        return overlap < threshold

    # ── 序列化 ──

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "papers": self.papers,
            "last_query": self.last_query,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionPaperPool":
        pool = cls(session_id=data.get("session_id", ""))
        pool.papers = data.get("papers", {})
        pool.last_query = data.get("last_query", "")
        pool.created_at = data.get("created_at", time.time())
        pool.updated_at = data.get("updated_at", time.time())
        return pool

    @property
    def total_count(self) -> int:
        return len(self.papers)

    @property
    def unread_count(self) -> int:
        return len(self.get_unread())

    @property
    def read_count(self) -> int:
        return len(self.get_read())
