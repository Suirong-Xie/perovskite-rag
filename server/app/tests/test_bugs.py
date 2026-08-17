"""
Test Suite G: Bug Regression Tests
Verifies that bugs discovered during structured audit have been fixed.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestBug001FixKeywordSubstringMatching:
    """BUG-001 (FIXED): 'vs' should NOT match 'perovskite' as substring.

    Fix: word-boundary matching (\b) for short ASCII keywords (<=4 chars).
    """

    def test_vs_no_longer_matches_perovskite(self):
        """'perovskite' should now be classified as 'broad', not 'specific'."""
        from app.services.agent_sm import _classify_question
        assert _classify_question("perovskite") == "broad", \
            "BUG-001 regression: 'vs' still matches 'perov(s)kite'"

    def test_efficiency_of_perovskite_is_broad(self):
        from app.services.agent_sm import _classify_question
        assert _classify_question("efficiency of perovskite") == "broad", \
            "BUG-001 regression: 'vs' still matches 'perov(s)kite'"

    def test_vs_still_matches_intentionally(self):
        """'vs' as a standalone word should still trigger 'specific'."""
        from app.services.agent_sm import _classify_question
        assert _classify_question("n-i-p vs p-i-n structure efficiency") == "specific", \
            "Standalone 'vs' should still be detected"

    def test_other_short_kw_use_word_boundaries(self):
        """Short keywords should not cause substring false positives."""
        from app.services.agent_sm import _classify_question
        # "which" should not be confused; "optimization" should work
        result = _classify_question("perovskite solar cell efficiency optimization")
        # "optimization" is a specific keyword
        assert result == "specific", "'optimization' should be detected"


class TestBug002FixFindPdfFastEmptySource:
    """BUG-002 (FIXED): find_pdf_fast("") now returns None.

    Fix: guard at function entry: if not source or not source.strip(): return None.
    """

    def test_empty_source_returns_none(self):
        from app.services.tools.paper_utils import find_pdf_fast
        result = find_pdf_fast("")
        assert result is None, "BUG-002 regression: empty source should return None"

    def test_whitespace_source_returns_none(self):
        from app.services.tools.paper_utils import find_pdf_fast
        assert find_pdf_fast("   ") is None
        assert find_pdf_fast("\t") is None

    def test_nonexistent_source_still_returns_none(self):
        from app.services.tools.paper_utils import find_pdf_fast
        result = find_pdf_fast("this_paper_does_not_exist_xyz123.pdf")
        assert result is None


class TestBug003FixSynonymExpansion:
    """BUG-003 (FIXED): Synonym expansion now interleaves across keywords.

    Fixes:
      - Interleaved allocation: each keyword gets 1 variant before round 2.
      - Multi-word synonym keys (e.g., "lead free") matched via n-gram sliding window.
      - Synonym dict keys normalized to lowercase (fixes "2D" vs "2d" mismatch).
    """

    def test_2d_synonyms_no_longer_blocked_by_stability(self):
        """'2D perovskite stability' should produce 2d synonyms."""
        from app.services.retrieval import _expand_queries
        queries = _expand_queries("2D perovskite stability")
        has_2d_syn = any(
            "two-dimensional" in q or "ruddlesden-popper" in q or "layered" in q
            for q in queries
        )
        assert has_2d_syn, (
            f"BUG-003 regression: 2D synonyms not found in {queries}"
        )

    def test_lead_free_multi_word_matched(self):
        """'lead free perovskite' should produce tin/sn-based variants."""
        from app.services.retrieval import _expand_queries
        queries = _expand_queries("lead free perovskite solar cells")
        has_lf_syn = any("tin" in q or "sn-based" in q for q in queries)
        assert has_lf_syn, (
            f"BUG-003 regression: 'lead free' multi-word not matched in {queries}"
        )

    def test_large_area_multi_word_matched(self):
        """'large area perovskite' should produce scalable/scale-up synonyms."""
        from app.services.retrieval import _expand_queries
        queries = _expand_queries("large area perovskite module")
        has_syn = any("scalable" in q or "scale-up" in q for q in queries)
        assert has_syn, (
            f"BUG-003 regression: 'large area' multi-word not matched in {queries}"
        )


class TestBug004FixTieBreaking:
    """BUG-004 (FIXED): Classifier now breaks ties in favor of specific keywords.

    Fix: strong specific keywords ("vs", "difference", "compare", etc.) receive
    2x weight so they break ties with broad keywords.
    """

    def test_difference_vs_what_is_tie_now_specific(self):
        """'what is the difference between...' should be specific now."""
        from app.services.agent_sm import _classify_question
        result = _classify_question(
            "what is the difference between organic and inorganic HTLs?"
        )
        assert result == "specific", (
            f"BUG-004 regression: tie should break to specific, got '{result}'"
        )

    def test_compare_in_broad_context_still_specific(self):
        """'tell me about compare...' should be specific (compare has 2x weight)."""
        from app.services.agent_sm import _classify_question
        result = _classify_question("overview and compare perovskite stability")
        assert result == "specific", (
            f"BUG-004 regression: 'compare' should break tie, got '{result}'"
        )

    def test_which_in_broad_context_still_specific(self):
        """'which' should be detected as strong specific keyword."""
        from app.services.agent_sm import _classify_question
        # "which" is a strong specific keyword (2x weight)
        result = _classify_question("summarize which method is best for perovskite")
        assert result == "specific", (
            f"BUG-004 regression: 'which' should break tie, got '{result}'"
        )
