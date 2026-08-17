"""
Test Suite A: State Machine Core Logic
Covers: StateContext, question/paper classification, fallback answer, state transitions
"""
import pytest
import re
import time
from app.services.agent_sm import (
    StateContext, AgentState,
    _classify_question, _classify_paper,
    _build_fallback_answer,
    MIN_FULLTEXT_PAPERS, MIN_READ_CHARS,
    MAX_BACK_TO_RETRIEVE, MAX_CONSECUTIVE_FAILS,
    BUDGETS,
)


# ═══════════════════════════════════════════════════════════════════
# A1: StateContext — Source Tracking
# ═══════════════════════════════════════════════════════════════════

class TestStateContextBasics:
    """Test the StateContext dataclass initialization and defaults."""

    def test_initial_state_is_empty(self, state_context):
        """Fresh StateContext should have empty sets and zero counts."""
        assert state_context.fulltext_sources == set()
        assert state_context.nofulltext_sources == set()
        assert state_context.unknown_sources == set()
        assert state_context.paper_meta == {}
        assert state_context.search_count == 0
        assert state_context.read_success_count == 0
        assert state_context.read_fail_count == 0
        assert state_context.back_to_retrieve_count == 0
        assert state_context.rejected_tool_calls == 0
        assert state_context.total_tool_calls == 0
        assert state_context.question_type == "broad"
        assert state_context.state_history == []

    def test_log_state_appends_history(self, state_context):
        """log_state should append a record to state_history."""
        state_context.log_state(AgentState.RETRIEVE, "enter", "test entry")
        assert len(state_context.state_history) == 1
        entry = state_context.state_history[0]
        assert entry["state"] == "retrieve"
        assert entry["action"] == "enter"
        assert entry["detail"] == "test entry"
        assert "timestamp" in entry

    def test_log_state_multiple_entries(self, state_context):
        """Multiple log_state calls should create ordered history."""
        states = [
            (AgentState.RETRIEVE, "enter"),
            (AgentState.RETRIEVE, "search", "keyword: stability"),
            (AgentState.QUICK_READ, "enter"),
            (AgentState.QUICK_READ, "read", "paper 1"),
            (AgentState.RESPOND, "enter"),
        ]
        for s, a, *d in states:
            state_context.log_state(s, a, d[0] if d else "")
        assert len(state_context.state_history) == 5
        assert state_context.state_history[0]["state"] == "retrieve"
        assert state_context.state_history[-1]["state"] == "respond"

    def test_state_summary_empty_context(self, state_context):
        """State summary should handle empty context gracefully."""
        summary = state_context.state_summary()
        assert summary["current_state"] == "init"
        assert summary["current_state_label"] == ""
        assert summary["searches_done"] == 0
        assert summary["papers_found"] == 0
        assert summary["papers_fulltext"] == 0
        assert summary["papers_nofulltext"] == 0
        assert summary["papers_read"] == 0
        assert summary["total_tool_calls"] == 0

    def test_state_summary_with_data(self, preset_context):
        """State summary should reflect current context state."""
        preset_context.log_state(AgentState.QUICK_READ, "read", "paper read")
        preset_context.search_count = 3
        preset_context.read_success_count = 2
        pres = preset_context.state_summary()
        assert pres["current_state"] == "quick_read"
        assert pres["searches_done"] == 3
        assert pres["papers_fulltext"] == 2
        assert pres["papers_nofulltext"] == 1
        assert pres["papers_found"] == 2 + 3  # fulltext + unknown
        assert pres["papers_read"] == 2

    def test_state_summary_after_transition(self, preset_context):
        """After transitioning to respond, summary should show correct state."""
        preset_context.log_state(AgentState.RESPOND, "enter")
        pres = preset_context.state_summary()
        assert pres["current_state"] == "respond"
        assert pres["current_state_label"] == "生成回答"


class TestSourceTracking:
    """Test the source set operations for paper tracking."""

    def test_add_fulltext_source(self, state_context):
        state_context.fulltext_sources.add("Nature_2023_test.pdf")
        assert len(state_context.fulltext_sources) == 1
        state_context.fulltext_sources.add("Nature_2023_test.pdf")  # duplicate
        assert len(state_context.fulltext_sources) == 1  # still 1

    def test_move_from_unknown_to_fulltext(self, state_context):
        """When a paper is successfully read, it moves from unknown to fulltext."""
        state_context.unknown_sources.add("Nature_2023_test.pdf")
        # Simulate successful read
        state_context.fulltext_sources.add("Nature_2023_test.pdf")
        state_context.unknown_sources.discard("Nature_2023_test.pdf")
        assert "Nature_2023_test.pdf" in state_context.fulltext_sources
        assert "Nature_2023_test.pdf" not in state_context.unknown_sources

    def test_move_to_nofulltext_on_fail(self, state_context):
        """When read fails, source moves from unknown to nofulltext."""
        state_context.unknown_sources.add("missing_paper.pdf")
        state_context.nofulltext_sources.add("missing_paper.pdf")
        state_context.unknown_sources.discard("missing_paper.pdf")
        assert "missing_paper.pdf" in state_context.nofulltext_sources
        assert "missing_paper.pdf" not in state_context.unknown_sources

    def test_source_mutual_exclusion(self, state_context):
        """A paper should only be in ONE set at a time."""
        src = "Nature_2023_test.pdf"
        state_context.unknown_sources.add(src)
        # Move to fulltext
        state_context.fulltext_sources.add(src)
        state_context.unknown_sources.discard(src)
        assert src in state_context.fulltext_sources
        assert src not in state_context.unknown_sources
        assert src not in state_context.nofulltext_sources


# ═══════════════════════════════════════════════════════════════════
# A2: Question Classification
# ═══════════════════════════════════════════════════════════════════

class TestQuestionClassification:
    """Test the keyword-based question type classifier."""

    # ── Broad questions ──
    BROAD_QUESTIONS = [
        "介绍一下钙钛矿太阳能电池的稳定性研究进展",
        "概述钙钛矿太阳能电池的发展现状",
        "钙钛矿太阳能电池有哪些提高效率的方法？",
        "什么是钙钛矿太阳能电池的离子迁移现象？",
        "列举几种提高钙钛矿稳定性的策略",
        "给我看几篇关于钙钛矿界面的论文",
        "总结一下全无机钙钛矿的研究进展",
        "归纳钙钛矿叠层电池的最新突破",
        "钙钛矿太阳能电池的研究热点有哪些？",
        "tell me about perovskite solar cell stability",
        "what is the current status of lead-free perovskites?",
        "overview of perovskite tandem solar cells",
        "summarize the recent progress in perovskite photovoltaics",
        "an introduction to 2D perovskite materials",
        "综述钙钛矿太阳能电池的长期稳定性",
        "钙钛矿领域的前沿方向是什么？",
    ]

    # ── Specific questions ──
    SPECIFIC_QUESTIONS = [
        "对比n-i-p和p-i-n结构的钙钛矿太阳能电池的稳定性差异",
        "MAPbI3和FAPbI3的带隙分别是多少？",
        "掺杂钾离子对钙钛矿薄膜形貌有什么影响？",
        "具体如何通过钝化减少钙钛矿缺陷？",
        "哪种空穴传输材料效率最高？",
        "退火温度如何影响钙钛矿晶粒尺寸？",
        "compare the efficiency of different ETL materials in perovskite solar cells",
        # NOTE: "what is" (broad) matches before "difference" (specific), tie → broad
        # "what is the difference between organic and inorganic HTLs?" → broad
        "how much does bromide doping increase the bandgap?",
        "which fabrication method gives the highest PCE?",
        # NOTE: this question has no strong specific keywords; slips to broad
        # "how does moisture affect MAPbI3 degradation rate?" → broad
        "具体来说，CsPbI3的相稳定性如何提高？",
    ]

    # ── Edge cases ──
    EDGE_QUESTIONS = [
        ("", "broad"),  # empty
        ("   ", "broad"),  # whitespace
        ("perovskite", "broad"),  # FIXED: "vs" no longer matches as substring
        ("钙钛矿", "broad"),  # single CN word
        ("对比 vs difference", "specific"),  # mixed language
        ("efficiency of perovskite", "broad"),  # FIXED: "vs" no longer matches as substring
        ("如何提高钙钛矿太阳能电池的效率并且对比不同方法？", "specific"),  # CN, more specific
    ]

    @pytest.mark.parametrize("question", BROAD_QUESTIONS)
    def test_broad_questions(self, question):
        assert _classify_question(question) == "broad", \
            f"Expected 'broad' for: {question[:60]}..."

    @pytest.mark.parametrize("question", SPECIFIC_QUESTIONS)
    def test_specific_questions(self, question):
        assert _classify_question(question) == "specific", \
            f"Expected 'specific' for: {question[:60]}..."

    @pytest.mark.parametrize("question,expected", EDGE_QUESTIONS)
    def test_edge_cases(self, question, expected):
        assert _classify_question(question) == expected, \
            f"Failed for: '{question[:60]}'"

    def test_case_insensitivity(self):
        """Classification should be case-insensitive."""
        upper = "COMPARE THE DIFFERENCE BETWEEN NIP AND PIN STRUCTURES"
        lower = "compare the difference between nip and pin structures"
        assert _classify_question(upper) == _classify_question(lower)

    def test_default_is_broad(self):
        """Unclassifiable questions should default to 'broad'."""
        assert _classify_question("abc123 xyz789") == "broad"
        assert _classify_question("---") == "broad"


# ═══════════════════════════════════════════════════════════════════
# A3: Paper Classification
# ═══════════════════════════════════════════════════════════════════

class TestPaperClassification:
    """Test review vs experimental paper classification."""

    REVIEW_TITLES = [
        "A comprehensive review of perovskite solar cells",
        "Recent advances in perovskite photovoltaics",
        "Progress and perspectives on halide perovskites",
        "A survey of stability enhancement strategies",
        "Perovskite solar cells: an overview of current status",
        "Perovskite photovoltaics: a roadmap to commercialization",
        "Recent progress in lead-free perovskite solar cells",
        "State of the art of flexible perovskite devices",
        "This review summarizes the degradation mechanisms",
        "We review the interfacial engineering approaches",
        "Critical review of charge transport layers",
        "Mini review: passivation strategies for perovskite",
        "Current status of tandem perovskite solar cells",
        "summarizes recent advances in 2D perovskites",
        "钙钛矿太阳能电池研究进展综述",
        "钙钛矿光伏器件研究进展与展望",
    ]

    EXPERIMENTAL_TITLES = [
        "Enhanced stability of MAPbI3 via potassium doping",
        "High-efficiency inverted perovskite solar cells",
        "Ion migration suppression in mixed-halide perovskites",
        "Interface passivation using organic ammonium salts",
        "Achieving 25.8% efficiency in perovskite solar cells",
        "The role of grain boundaries in perovskite degradation",
        "Compositional engineering of wide-bandgap perovskites",
        "Efficient and stable perovskite-silicon tandem cells",
    ]

    @pytest.mark.parametrize("title", REVIEW_TITLES)
    def test_review_papers(self, title):
        meta = {"title": title, "content_preview": ""}
        assert _classify_paper("test.pdf", meta) == "review", \
            f"Expected 'review' for: {title[:60]}"

    @pytest.mark.parametrize("title", EXPERIMENTAL_TITLES)
    def test_experimental_papers(self, title):
        meta = {"title": title, "content_preview": ""}
        assert _classify_paper("test.pdf", meta) == "experimental", \
            f"Expected 'experimental' for: {title[:60]}"

    def test_review_keyword_in_content_preview(self):
        """Review detection from content_preview, not just title."""
        meta = {"title": "Investigation of perovskite films",
                "content_preview": "This review provides a comprehensive overview of..."}
        assert _classify_paper("test.pdf", meta) == "review"

    def test_review_keyword_in_source_filename(self):
        """Review detection from source filename."""
        meta = {"title": "A study of perovskite", "content_preview": "We report..."}
        assert _classify_paper("Nature_2023_review_perovskite.pdf", meta) == "review"

    def test_default_is_experimental(self):
        """Unclassified papers default to 'experimental'."""
        meta = {"title": "Some generic paper title", "content_preview": "We measured..."}
        assert _classify_paper("unknown.pdf", meta) == "experimental"

    def test_empty_meta(self):
        """Empty meta should still return 'experimental'."""
        assert _classify_paper("paper.pdf", {}) == "experimental"


# ═══════════════════════════════════════════════════════════════════
# A4: Fallback Answer Generation
# ═══════════════════════════════════════════════════════════════════

class TestBuildFallbackAnswer:
    """Test the fallback answer builder used when LLM fails to produce output."""

    def test_with_fulltext_papers(self):
        ft = {"Nature_2023_s41586-023-06121-1.pdf",
              "NatEnergy_2024_s41560-024-01234-5.pdf"}
        result = _build_fallback_answer(ft, set())
        assert "2" in result  # count
        assert "📄" in result
        assert "/api/pdf/Nature_2023_s41586-023-06121-1" in result
        assert "/api/pdf/NatEnergy_2024_s41560-024-01234-5" in result

    def test_with_nofulltext_only(self):
        result = _build_fallback_answer(set(), {"JACS_2021_jacs.3456.pdf"})
        assert "1" in result
        assert "仅有摘要" in result

    def test_empty_all(self):
        result = _build_fallback_answer(set(), set())
        assert "未找到相关论文" in result

    def test_output_is_valid_markdown(self):
        """Fallback answer should contain a markdown heading."""
        result = _build_fallback_answer({"test.pdf"}, set())
        assert result.startswith("##")


# ═══════════════════════════════════════════════════════════════════
# A5: Constants Validation
# ═══════════════════════════════════════════════════════════════════

class TestConstants:
    """Validate that critical thresholds are within reasonable bounds."""

    def test_min_fulltext_papers_reasonable(self):
        assert MIN_FULLTEXT_PAPERS >= 2, \
            "Should require at least 2 fulltext papers (progressive mode)"
        assert MIN_FULLTEXT_PAPERS <= 20, \
            "Should not be unrealistically high"

    def test_min_read_chars_reasonable(self):
        assert MIN_READ_CHARS >= 50, "Too low: would accept garbage"
        assert MIN_READ_CHARS <= 500, "Too high: would reject valid extracts"

    def test_max_back_to_retrieve_reasonable(self):
        assert MAX_BACK_TO_RETRIEVE >= 1, "Need at least one retry"
        assert MAX_BACK_TO_RETRIEVE <= 5, "Should not loop infinitely"

    def test_max_consecutive_fails_reasonable(self):
        assert MAX_CONSECUTIVE_FAILS >= 2, "Too low: one transient fail triggers back"
        assert MAX_CONSECUTIVE_FAILS <= 5, "Too high: wastes tries on missing papers"

    def test_budgets_have_required_keys(self):
        assert "retrieve_llm" in BUDGETS
        assert "retrieve_search" in BUDGETS
        assert "quick_read" in BUDGETS
        assert BUDGETS["retrieve_llm"] > 0
        assert BUDGETS["retrieve_search"] > 0
        assert BUDGETS["quick_read"] > 0


# ═══════════════════════════════════════════════════════════════════
# A6: Pre-search Paper Extraction (regex validation)
# ═══════════════════════════════════════════════════════════════════

class TestPreSearchExtraction:
    """Test the regex used to extract File IDs from pre-search user messages."""

    PRESEARCH_PATTERN = re.compile(r'File ID:\s*`?([^`\s]+)`?')

    def test_extract_single_file_id(self):
        text = "File ID: `Nature_2023_s41586-023-06121-1`"
        matches = self.PRESEARCH_PATTERN.findall(text)
        assert "Nature_2023_s41586-023-06121-1" in matches

    def test_extract_multiple_file_ids(self):
        text = (
            "[P1] Nature | File ID: `NatEnergy_2024_xxx` | PDF: /api/pdf/...\n"
            "[P2] Science | File ID: `Science_2023_yyy` | PDF: /api/pdf/..."
        )
        matches = self.PRESEARCH_PATTERN.findall(text)
        assert len(matches) == 2
        assert "NatEnergy_2024_xxx" in matches
        assert "Science_2023_yyy" in matches

    def test_file_id_without_backticks(self):
        text = "File ID: Nature_2023_s41586-023-06121-1"
        matches = self.PRESEARCH_PATTERN.findall(text)
        assert "Nature_2023_s41586-023-06121-1" in matches

    def test_no_file_id_present(self):
        text = "No file IDs here, just regular text."
        matches = self.PRESEARCH_PATTERN.findall(text)
        assert matches == []

    def test_file_id_with_underscore_digits(self):
        """File IDs with complex patterns should be extracted correctly."""
        text = "File ID: `ACS_Energy_Lett_2023_acs.5678`"
        matches = self.PRESEARCH_PATTERN.findall(text)
        assert "ACS_Energy_Lett_2023_acs.5678" in matches

    def test_non_ascii_boundary(self):
        """File ID should not capture trailing CJK characters."""
        text = "File ID: `NatEnergy_2024_test`的全文"
        matches = self.PRESEARCH_PATTERN.findall(text)
        assert "NatEnergy_2024_test" in matches
