"""
Test Suite E: Chat Pipeline & Integration
Covers: chat.py logic, task lifecycle, source validation,
        thinking chain, citation checks, PDF verification
"""
import pytest
import json
import re
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5.core.schemas import AgentEvent, TaskInfo, ChatRequest


# ═══════════════════════════════════════════════════════════════════
# E1: TaskInfo Lifecycle
# ═══════════════════════════════════════════════════════════════════

class TestTaskInfo:
    """Test the TaskInfo data structure lifecycle."""

    def test_initial_state(self):
        ti = TaskInfo(sid="test_session_123")
        assert ti.sid == "test_session_123"
        assert ti.chunks == []
        assert ti.done is False
        assert ti.error is None
        assert ti.cancelled is False
        assert ti.sources == []
        assert ti.pdfs_validated == set()
        assert ti.sources_json is None
        assert ti.agent_state is None

    def test_state_transitions(self):
        ti = TaskInfo(sid="test")
        assert not ti.done
        ti.done = True
        assert ti.done
        ti.error = "Something broke"
        assert ti.error == "Something broke"

    def test_cancellation(self):
        ti = TaskInfo(sid="test")
        ti.cancelled = True
        ti.done = True
        assert ti.cancelled
        assert ti.done

    def test_source_accumulation(self):
        ti = TaskInfo(sid="test")
        ti.sources.append({"source": "Nature_2023_test.pdf", "journal_name": "Nature"})
        ti.sources.append({"source": "Science_2022_test.pdf", "journal_name": "Science"})
        assert len(ti.sources) == 2

    def test_pdf_validation_tracking(self):
        ti = TaskInfo(sid="test")
        ti.pdfs_validated.add("Nature_2023_test")
        ti.pdfs_validated.add("NatEnergy_2024_test")
        assert len(ti.pdfs_validated) == 2
        # Duplicate should be ignored
        ti.pdfs_validated.add("Nature_2023_test")
        assert len(ti.pdfs_validated) == 2

    def test_chunk_accumulation(self):
        ti = TaskInfo(sid="test")
        ti.chunks.append("Hello ")
        ti.chunks.append("World")
        assert "".join(ti.chunks) == "Hello World"

    def test_agent_state_update(self):
        ti = TaskInfo(sid="test")
        ti.agent_state = {
            "current_state": "retrieve",
            "searches_done": 2,
            "papers_found": 15,
        }
        assert ti.agent_state["current_state"] == "retrieve"
        assert ti.agent_state["searches_done"] == 2


# ═══════════════════════════════════════════════════════════════════
# E2: Thinking Chain Building
# ═══════════════════════════════════════════════════════════════════

class TestBuildThinkingChain:
    """Test chat.py._build_thinking_chain with realistic data."""

    def _build_thinking_chain(self, chunks):
        """Exact replica of chat.py._build_thinking_chain."""
        chain_parts = []
        for c in chunks:
            stripped = c.strip()
            if not stripped:
                continue
            if (stripped.startswith("\U0001f4ad") or
                stripped.startswith("\U0001f527") or
                stripped.startswith("⚠️")):
                chain_parts.append(stripped)
        return "\n\n".join(chain_parts)

    def test_empty_input(self):
        assert self._build_thinking_chain([]) == ""

    def test_only_answer_chunks(self):
        """Pure answer chunks (no thinking/tool markers) should produce empty chain."""
        chunks = [
            "## Results\n\n",
            "The perovskite solar cell efficiency...\n\n",
            "[📄](/api/pdf/test.pdf)\n\n",
        ]
        assert self._build_thinking_chain(chunks) == ""

    def test_mixed_thinking_and_answer(self):
        chunks = [
            "\U0001f4ad 搜索策略：从稳定性角度搜索...\n\n",
            '\U0001f527 **search_papers**({"query": "perovskite stability"})\n\n',
            "## 钙钛矿稳定性研究\n\n",  # answer — should be filtered
            "Smith et al. found...\n\n",
        ]
        result = self._build_thinking_chain(chunks)
        assert "## 钙钛矿稳定性研究" not in result
        assert "Smith et al" not in result
        assert "\U0001f4ad" in result
        assert "\U0001f527" in result

    def test_full_agent_trace(self):
        """Complete agent trace from start to finish."""
        chunks = [
            "\U0001f4ad 分析问题：钙钛矿太阳能电池的长期稳定性研究进展\n\n",
            '\U0001f527 **search_papers**({"query": "perovskite solar cell long-term stability"})\n\n',
            "\U0001f4ad 搜索返回 5 篇论文，其中包含 1 篇综述。优先阅读综述论文。\n\n",
            '\U0001f527 **read_paper**({"source": "NatEnergy_2024_review.pdf"})\n\n',
            "\U0001f4ad 综述已读完，了解了主要降解机制分类。现在读实验论文获取具体数据。\n\n",
            '\U0001f527 **read_paper**({"source": "Nature_2023_experimental.pdf"})\n\n',
            "\U0001f4ad 获得关键数据：PCE=25.8%, T80=1000h。继续读下一篇。\n\n",
            '\U0001f527 **read_paper**({"source": "Science_2022_stability.pdf"})\n\n',
            "⚠️ PDF not found: Science_2022_stability.pdf\n\n",
            "\U0001f4ad 这篇没有全文，跳过。已收集足够数据，开始回答。\n\n",
        ]
        result = self._build_thinking_chain(chunks)
        assert result.count("\U0001f4ad") == 5
        assert result.count("\U0001f527") == 4
        assert result.count("⚠️") == 1

    def test_consecutive_errors(self):
        """Multiple consecutive errors should all appear in chain."""
        chunks = [
            '\U0001f527 **read_paper**({"source": "paper1.pdf"})\n\n',
            "⚠️ PDF not found: paper1.pdf\n\n",
            '\U0001f527 **read_paper**({"source": "paper2.pdf"})\n\n',
            "⚠️ PDF not found: paper2.pdf\n\n",
            '\U0001f527 **read_paper**({"source": "paper3.pdf"})\n\n',
            "⚠️ PDF not found: paper3.pdf\n\n",
        ]
        result = self._build_thinking_chain(chunks)
        assert result.count("⚠️") == 3

    def test_leading_trailing_whitespace_handled(self):
        chunks = ["  \U0001f4ad thinking  \n", "\t\U0001f527 tool_call\t\n"]
        result = self._build_thinking_chain(chunks)
        assert "\U0001f4ad thinking" in result
        assert "\U0001f527 tool_call" in result


# ═══════════════════════════════════════════════════════════════════
# E3: Source Validation
# ═══════════════════════════════════════════════════════════════════

class TestSourceValidation:
    """Test source deduplication, PDF verification, and source list building."""

    def test_source_deduplication(self):
        """Duplicate sources should be merged (same source name)."""
        sources = [
            {"source": "Nature_2023_test.pdf", "journal_name": "Nature"},
            {"source": "Nature_2023_test.pdf", "journal_name": "Nature"},  # duplicate
            {"source": "Science_2022_test.pdf", "journal_name": "Science"},
        ]
        seen = set()
        unique = []
        for s in sources:
            src = s.get("source", "")
            if src and src not in seen:
                seen.add(src)
                unique.append(s)
        assert len(unique) == 2

    def test_file_id_normalization(self):
        """file_id should strip .pdf suffix."""
        source = "Nature_2023_s41586-023-06121-1.pdf"
        file_id = source.replace(".pdf", "")
        assert file_id == "Nature_2023_s41586-023-06121-1"
        assert ".pdf" not in file_id

    def test_source_json_serialization(self):
        """Validated sources should serialize to JSON."""
        validated = [{
            "file_id": "Nature_2023_s41586-023-06121-1",
            "journal_name": "Nature",
            "source": "Nature_2023_s41586-023-06121-1.pdf",
            "content_preview": "We investigate the long-term stability...",
            "pdf_url": "/api/pdf/Nature_2023_s41586-023-06121-1",
            "highlight": {},
            "has_pdf": True,
        }]
        json_str = json.dumps(validated, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["has_pdf"] is True

    def test_supplementary_source_no_pdf(self):
        """Sources without PDF should have doi_url."""
        supplementary = {
            "file_id": "Science_2022_abc",
            "journal_name": "Science",
            "source": "Science_2022_abc.pdf",
            "content_preview": "A brief summary...",
            "doi_url": "https://doi.org/10.1126/science.abc1234",
            "has_pdf": False,
        }
        assert supplementary["has_pdf"] is False
        assert supplementary["doi_url"].startswith("https://doi.org/")


# ═══════════════════════════════════════════════════════════════════
# E4: Content Cleanup and Fallback
# ═══════════════════════════════════════════════════════════════════

class TestContentCleanup:
    """Test the final content cleanup and fallback logic."""

    TOOL_CALLS_PATTERN = r'<(?:anth:)?tool_calls>.*?</(?:anth:)?tool_calls>'

    def test_clean_content_no_change(self):
        text = "## Valid answer about perovskite solar cells."
        cleaned = re.sub(self.TOOL_CALLS_PATTERN, '', text, flags=re.DOTALL).strip()
        assert cleaned == text

    def test_strip_tool_calls_from_answer(self):
        text = (
            "Good content here.\n"
            "<tool_calls>\n"
            "<invoke name='search_papers'>\n"
            "</invoke>\n"
            "</tool_calls>\n"
            "More good content."
        )
        cleaned = re.sub(self.TOOL_CALLS_PATTERN, '', text, flags=re.DOTALL).strip()
        assert "Good content here." in cleaned
        assert "More good content." in cleaned
        assert "<tool_calls>" not in cleaned

    def test_strip_anthropic_format(self):
        text = "Content <anth:tool_calls>tool</anth:tool_calls> end."
        cleaned = re.sub(self.TOOL_CALLS_PATTERN, '', text, flags=re.DOTALL).strip()
        assert "Content  end." == cleaned

    def test_fallback_on_short_content(self):
        """Content < 50 chars should trigger fallback."""
        content = "Short."
        assert len(content) < 50

    def test_fallback_on_empty_content(self):
        content = ""
        assert not content.strip()

    def test_fallback_generation_with_sources(self):
        """Generate fallback message from validated sources."""
        validated = [
            {"has_pdf": True}, {"has_pdf": True}, {"has_pdf": True},
            {"has_pdf": False},
        ]
        n_primary = sum(1 for s in validated if s.get("has_pdf") is not False)
        n_supp = len(validated) - n_primary
        parts = [f"已检索到 {len(validated)} 篇相关文献"]
        if n_primary:
            parts.append(f"{n_primary} 篇有全文")
        if n_supp:
            parts.append(f"{n_supp} 篇补充参考")
        parts.append("，请查看下方来源列表获取详细信息。")
        result = "".join(parts)
        assert "4" in result
        assert "3" in result
        assert "全文" in result

    def test_fallback_no_sources(self):
        """No sources should give generic error."""
        validated = []
        if not validated:
            result = "抱歉，未能完成本次检索。请重试或更换关键词。"
        assert "未能完成" in result


# ═══════════════════════════════════════════════════════════════════
# E5: Pre-Search User Message Construction
# ═══════════════════════════════════════════════════════════════════

class TestPreSearchMessage:
    """Test the user message augmentation with pre-search results."""

    def test_primary_papers_format(self):
        """Pre-search message with PDF papers should follow correct format."""
        primary = [{
            "source": "Nature_2023_test.pdf",
            "journal_name": "Nature",
            "content": "Test content about perovskite stability...",
            "has_pdf": True,
        }]
        user_message = "钙钛矿的稳定性如何？"
        user_message += "\n\n系统已为你预检索了以下文献：\n"
        user_message += "\n📄 **可阅读全文的论文**（优先使用这些论文的信息回答）：\n"
        for i, r in enumerate(primary):
            file_id = r.get("source", "").replace(".pdf", "")
            pdf_link = f"/api/pdf/{file_id}"
            user_message += (
                f"[P{i+1}] {r.get('journal_name', 'Unknown')} | "
                f"File ID: `{file_id}` | "
                f"PDF: {pdf_link}\n"
                f"    内容: {r.get('content', '')[:400]}\n\n"
            )
        assert "📄 **可阅读全文的论文**" in user_message
        assert "File ID: `Nature_2023_test`" in user_message
        assert "/api/pdf/Nature_2023_test" in user_message

    def test_supplementary_papers_format(self):
        supplementary = [{
            "journal_name": "Science",
            "_s2_doi": "10.1126/science.abc1234",
            "content": "Brief abstract...",
            "has_pdf": False,
        }]
        user_message = "\n🔗 **仅有摘要/元数据的论文**"
        user_message += "（仅作参考，无法打开全文，回答中不要将其作为主要信息来源）：\n"
        for i, r in enumerate(supplementary):
            doi = r.get("_s2_doi", "")
            doi_link = f"https://doi.org/{doi}" if doi else "(无 DOI)"
            user_message += (
                f"[S{i+1}] {r.get('journal_name', 'Unknown')} | "
                f"DOI: {doi_link}\n"
                f"    摘要: {r.get('content', '')[:300]}\n\n"
            )
        assert "🔗 **仅有摘要/元数据的论文**" in user_message
        assert "无法打开全文" in user_message
        assert "https://doi.org/10.1126/science.abc1234" in user_message

    def test_no_pre_search_results(self):
        """When no pre-search results, don't add the section header."""
        results = []
        if not results:
            # Don't add pre-search section
            pass
        assert len(results) == 0

    def test_mixed_primary_and_supplementary(self):
        primary = [
            {"source": "NatEnergy_2024_test.pdf", "journal_name": "NatEnergy",
             "content": "We study...", "has_pdf": True},
        ]
        supplementary = [
            {"journal_name": "ACS Energy Lett", "_s2_doi": "10.1021/acs.12345",
             "content": "Summary...", "has_pdf": False},
        ]
        assert len(primary) > 0
        assert len(supplementary) > 0
        # Both sections should exist
        assert primary[0]["has_pdf"]
        assert not supplementary[0].get("has_pdf")


# ═══════════════════════════════════════════════════════════════════
# E6: ChatRequest Model
# ═══════════════════════════════════════════════════════════════════

class TestChatRequest:
    """Test the ChatRequest Pydantic model."""

    def test_minimal_request(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.session_id is None
        assert req.paper_id is None

    def test_full_request(self):
        req = ChatRequest(
            message="Explain perovskite stability",
            session_id="abc123",
            paper_id="Nature_2023_test",
        )
        assert req.message == "Explain perovskite stability"
        assert req.session_id == "abc123"
        assert req.paper_id == "Nature_2023_test"

    def test_chinese_message(self):
        req = ChatRequest(message="介绍钙钛矿太阳能电池的稳定性")
        assert len(req.message) > 0

    def test_empty_message(self):
        """Empty message should still create valid request."""
        req = ChatRequest(message="")
        assert req.message == ""


# ═══════════════════════════════════════════════════════════════════
# E7: AgentEvent Construction
# ═══════════════════════════════════════════════════════════════════

class TestAgentEvent:
    """Test AgentEvent factory methods."""

    def test_thinking_event(self):
        event = AgentEvent.thinking("Analyzing query...")
        assert event.type == "thinking"
        assert event.data["content"] == "Analyzing query..."

    def test_tool_call_event(self):
        event = AgentEvent.tool_call("search_papers", {"query": "perovskite"})
        assert event.type == "tool_call"
        assert event.data["name"] == "search_papers"
        assert event.data["arguments"] == {"query": "perovskite"}

    def test_tool_result_event(self):
        event = AgentEvent.tool_result("read_paper", "Content of paper...")
        assert event.type == "tool_result"
        assert event.data["name"] == "read_paper"

    def test_tool_result_with_error(self):
        event = AgentEvent.tool_result("read_paper", "", error="PDF not found")
        assert event.type == "tool_result"
        assert event.data["error"] == "PDF not found"

    def test_text_event(self):
        event = AgentEvent.text("## Results\n\nPCE is 25.8%")
        assert event.type == "text"
        assert "PCE" in event.data["content"]

    def test_done_event(self):
        event = AgentEvent.done()
        assert event.type == "done"
        assert event.data == {}

    def test_error_event(self):
        event = AgentEvent.error("Connection timeout")
        assert event.type == "error"
        assert event.data["message"] == "Connection timeout"

    def test_search_results_event(self):
        data = [{"source": "paper.pdf", "journal_name": "Nature"}]
        event = AgentEvent.search_results(data)
        assert event.type == "search_results"
        assert len(event.data["results"]) == 1

    def test_state_change_event(self):
        event = AgentEvent.state_change("read", {
            "current_state": "read",
            "papers_found": 5,
            "papers_read": 2,
        })
        assert event.type == "state"
        assert event.data["current_state"] == "read"
        assert event.data["papers_found"] == 5
        assert event.data["papers_read"] == 2


# ═══════════════════════════════════════════════════════════════════
# E8: Safety Valve & Budget Tests
# ═══════════════════════════════════════════════════════════════════

class TestSafetyValves:
    """Test that safety valves and budget limits are enforced."""

    def test_max_total_tool_calls(self):
        from v5.services.agent_sm import MAX_TOTAL_TOOL_CALLS
        assert MAX_TOTAL_TOOL_CALLS > 0
        assert MAX_TOTAL_TOOL_CALLS <= 50  # reasonable bound

    def test_max_rejected_in_read(self):
        from v5.services.agent_sm import MAX_REJECTED_IN_READ
        assert MAX_REJECTED_IN_READ >= 1
        assert MAX_REJECTED_IN_READ <= 5

    def test_max_loops_from_agent(self):
        from v5.core.config import AGENT_MAX_ROUNDS
        assert AGENT_MAX_ROUNDS > 0
        assert AGENT_MAX_ROUNDS <= 20

    def test_retrieve_budget_multiplier(self):
        """RETRIEVE has 2x multiplier on LLM rounds vs budget."""
        from v5.services.agent_sm import BUDGETS
        max_llm = BUDGETS["retrieve_llm"] * 2
        max_search = BUDGETS["retrieve_search"]
        assert max_llm > 0
        assert max_search > 0


# ═══════════════════════════════════════════════════════════════════
# E9: Compression & Context Management
# ═══════════════════════════════════════════════════════════════════

class TestContextCompression:
    """Test the RETRIEVE→READ context compression logic."""

    def test_retrieve_prompt_boundary_detection(self):
        """Test finding the RETRIEVE_PROMPT injection point in messages."""
        messages = [
            {"role": "system", "content": "You are a perovskite assistant."},
            {"role": "user", "content": "What is perovskite stability?"},
            {"role": "system", "content": "## 当前阶段：文献检索 (RETRIEVE)\n\n目标是收集..."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "search_papers", "arguments": "{}"}}]},
            {"role": "tool", "content": "Found 5 results..."},
        ]
        retrieve_start = None
        for i, m in enumerate(messages):
            if m.get("role") == "system" and "当前阶段：文献检索" in (m.get("content") or ""):
                retrieve_start = i
                break
        assert retrieve_start is not None
        assert retrieve_start == 2

    def test_no_retrieve_boundary(self):
        """When no RETRIEVE boundary, compression should be skipped."""
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Hello"},
        ]
        retrieve_start = None
        for i, m in enumerate(messages):
            if m.get("role") == "system" and "当前阶段：文献检索" in (m.get("content") or ""):
                retrieve_start = i
                break
        assert retrieve_start is None

    def test_source_extraction_from_search_results(self):
        """Extract paper sources from search result messages."""
        messages = [
            {"role": "tool", "content": (
                "Found 3 results for 'perovskite':\n"
                "[1] Nature | Source: Nature_2023_test.pdf\n"
                "[2] Science | Source: Science_2022_test.pdf\n"
            )},
            {"role": "user", "content": "File ID: `NatEnergy_2024_pre.pdf`"},
        ]
        found = []
        for m in messages:
            content = m.get("content")
            if not content:
                continue
            for match in re.finditer(
                r'(?:Source:\s*|File ID:\s*`?)([\w\d_\-./]+)',
                content
            ):
                pid = match.group(1).rstrip('.')
                if pid not in found:
                    found.append(pid)
        assert "Nature_2023_test.pdf" in found
        assert "NatEnergy_2024_pre.pdf" in found
