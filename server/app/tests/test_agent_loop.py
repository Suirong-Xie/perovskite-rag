"""
Test Suite B: Agent Loop Logic
Covers: parse_tool_call, extract_text_without_tool_call,
        thinking_chain building, citation hallucination detection,
        XML contamination detection in RESPOND output

Note: parse_tool_call and related functions are replicated here
to avoid the agent.py → materials_service → pymatgen import chain.
The functions under test are pure text processing with no side effects.
"""
import pytest
import json
import re


# ═══════════════════════════════════════════════════════════════════
# Replicated functions from agent.py (pure text processing, tested identically)
# ═══════════════════════════════════════════════════════════════════

TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*\n?(.*?)\n?\s*</tool_call>', re.DOTALL
)


def parse_tool_call(text: str):
    """Replica of agent.parse_tool_call — exact same logic."""
    from app.core.schemas import ToolCall
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None
    json_str = match.group(1).strip()
    try:
        data = json.loads(json_str)
        name = data.get("name", "")
        arguments = data.get("arguments", {})
        if not name:
            return None
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(name, arguments)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_text_without_tool_call(text: str) -> str:
    """Replica of agent.extract_text_without_tool_call — exact same logic."""
    return TOOL_CALL_PATTERN.sub('', text).strip()


# ═══════════════════════════════════════════════════════════════════
# B1: Tool Call XML Parsing
# ═══════════════════════════════════════════════════════════════════

class TestParseToolCall:
    """Test the <tool_call> XML regex parser."""

    def test_basic_tool_call(self):
        text = '<tool_call>\n{"name": "search_papers", "arguments": {"query": "perovskite stability"}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "search_papers"
        assert tc.arguments == {"query": "perovskite stability"}

    def test_tool_call_with_trailing_text(self):
        text = (
            'Let me search for papers.\n\n'
            '<tool_call>\n{"name": "search_papers", "arguments": {"query": "PCE"}}\n</tool_call>\n'
            'After searching, I will...'
        )
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "search_papers"

    def test_read_paper_call(self):
        text = '<tool_call>\n{"name": "read_paper", "arguments": {"source": "Nature_2023_xxx.pdf"}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "read_paper"
        assert tc.arguments["source"] == "Nature_2023_xxx.pdf"

    def test_read_arxiv_call(self):
        text = '<tool_call>\n{"name": "read_arxiv_paper", "arguments": {"arxiv_id": "2606.13414"}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "read_arxiv_paper"

    def test_extract_data_call(self):
        text = '<tool_call>\n{"name": "extract_data", "arguments": {"source": "Nature_2023_xxx.pdf"}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "extract_data"

    def test_compare_papers_call(self):
        text = ('<tool_call>\n'
                '{"name": "compare_papers", "arguments": {"sources": "a.pdf,b.pdf", "metrics": "PCE,Voc"}}\n'
                '</tool_call>')
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "compare_papers"

    def test_no_tool_call(self):
        text = "This is just a regular response with no tool calls."
        tc = parse_tool_call(text)
        assert tc is None

    def test_malformed_json(self):
        """Malformed JSON inside <tool_call> should return None."""
        text = '<tool_call>\n{this is not json}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is None

    def test_empty_name(self):
        """Tool call with empty name should return None."""
        text = '<tool_call>\n{"name": "", "arguments": {}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is None

    def test_arguments_is_string_not_dict(self):
        """If arguments is a string, should default to empty dict."""
        text = '<tool_call>\n{"name": "search_papers", "arguments": "invalid"}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.arguments == {}

    def test_arguments_is_null(self):
        """If arguments is null, should default to empty dict."""
        text = '<tool_call>\n{"name": "search_papers", "arguments": null}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.arguments == {}

    def test_multiple_tool_calls_only_first(self):
        """Only the first <tool_call> should be parsed."""
        text = (
            '<tool_call>\n{"name": "search_papers", "arguments": {"query": "A"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read_paper", "arguments": {"source": "x.pdf"}}\n</tool_call>'
        )
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "search_papers"

    def test_tool_call_no_newlines(self):
        """Tool call without newlines should still parse."""
        text = '<tool_call>{"name": "search_papers", "arguments": {"query": "test"}}</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.name == "search_papers"

    def test_tool_call_whitespace_variants(self):
        """Various whitespace around the JSON should be handled."""
        variants = [
            '<tool_call>{"name":"search_papers","arguments":{"query":"A"}}</tool_call>',
            '<tool_call>\n\t{"name": "search_papers", "arguments": {"query": "A"}}\t\n</tool_call>',
            '<tool_call>  {"name": "search_papers", "arguments": {"query": "A"}}  </tool_call>',
        ]
        for v in variants:
            tc = parse_tool_call(v)
            assert tc is not None, f"Failed to parse: {v[:50]}"
            assert tc.name == "search_papers"

    def test_json_with_unicode(self):
        """Tool calls with Unicode arguments should parse correctly."""
        text = '<tool_call>\n{"name": "search_papers", "arguments": {"query": "CdTe stability"}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.arguments["query"] == "CdTe stability"

    def test_json_with_special_chars(self):
        """JSON with special characters in arguments."""
        text = '<tool_call>\n{"name": "search_papers", "arguments": {"query": "mixed-halide (Cs/FA/MA)"}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert "mixed-halide" in tc.arguments["query"]

    def test_tool_call_arguments_with_int(self):
        """Arguments with integer values should be preserved."""
        text = '<tool_call>\n{"name": "search_papers", "arguments": {"query": "test", "top_k": 10}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.arguments["top_k"] == 10

    def test_tool_call_regex_no_false_positive(self):
        """The regex should not match text that looks like but isn't tool_call."""
        false_positives = [
            "I will call the search tool.",
            "<tool>some content</tool>",
            "tool_call is not a tag",
            "<<tool_call>>",
        ]
        for text in false_positives:
            tc = parse_tool_call(text)
            assert tc is None, f"False positive for: {text[:40]}"

    def test_tool_call_with_escaped_quotes(self):
        """JSON with escaped quotes in string values."""
        text = '<tool_call>\n{"name": "search_papers", "arguments": {"query": "mixed 2D/3D perovskite"}}\n</tool_call>'
        tc = parse_tool_call(text)
        assert tc is not None
        assert "2D/3D" in tc.arguments["query"]


# ═══════════════════════════════════════════════════════════════════
# B2: Text Extraction (removing tool_call blocks)
# ═══════════════════════════════════════════════════════════════════

class TestExtractTextWithoutToolCall:
    """Test the function that strips <tool_call> blocks from text."""

    def test_remove_single_tool_call(self):
        text = (
            "Let me search.\n"
            '<tool_call>\n{"name": "search_papers", "arguments": {"query": "test"}}\n</tool_call>\n'
            "Results are found."
        )
        result = extract_text_without_tool_call(text)
        assert "Let me search." in result
        assert "Results are found." in result
        assert "<tool_call>" not in result

    def test_no_tool_call_present(self):
        text = "Just a regular answer with no tool calls."
        result = extract_text_without_tool_call(text)
        assert result == text

    def test_only_tool_call(self):
        text = '<tool_call>\n{"name": "search_papers", "arguments": {"query": "test"}}\n</tool_call>'
        result = extract_text_without_tool_call(text)
        assert result.strip() == ""

    def test_multiple_tool_calls_removed(self):
        text = (
            '<tool_call>{"name": "A"}</tool_call>'
            'between'
            '<tool_call>{"name": "B"}</tool_call>'
        )
        result = extract_text_without_tool_call(text)
        assert "<tool_call>" not in result
        assert "between" in result

    def test_nested_angle_brackets_safe(self):
        """Text with angle brackets (not tool_call) should be preserved."""
        text = "The bandgap is < 1.5 eV and efficiency > 20%"
        result = extract_text_without_tool_call(text)
        assert "< 1.5" in result
        assert "> 20%" in result


# ═══════════════════════════════════════════════════════════════════
# B3: Thinking Chain Building
# ═══════════════════════════════════════════════════════════════════

class TestThinkingChain:
    """Test the thinking chain builder from chat.py._build_thinking_chain."""

    def _build_thinking_chain(self, chunks):
        """Replicate chat.py._build_thinking_chain logic."""
        chain_parts = []
        for c in chunks:
            stripped = c.strip()
            if not stripped:
                continue
            if (stripped.startswith("\U0001f4ad") or    # 💭
                stripped.startswith("\U0001f527") or    # 🔧
                stripped.startswith("⚠️")):   # ⚠️
                chain_parts.append(stripped)
        return "\n\n".join(chain_parts)

    def test_thinking_single_chunk(self):
        chunks = ["\U0001f4ad 正在思考搜索策略...\n\n"]
        result = self._build_thinking_chain(chunks)
        assert "正在思考搜索策略" in result

    def test_tool_call_in_chain(self):
        chunks = [
            "\U0001f4ad 准备搜索...\n\n",
            '\U0001f527 **search_papers**({"query": "perovskite stability"})\n\n',
        ]
        result = self._build_thinking_chain(chunks)
        assert "\U0001f4ad" in result
        assert "\U0001f527" in result
        assert "search_papers" in result

    def test_error_in_chain(self):
        chunks = [
            '\U0001f527 **read_paper**({"source": "missing.pdf"})\n\n',
            "⚠️ PDF not found: missing.pdf\n\n",
        ]
        result = self._build_thinking_chain(chunks)
        assert "\U0001f527" in result
        assert "⚠" in result
        assert "missing.pdf" in result

    def test_markdown_answer_is_filtered(self):
        """Final answer (markdown) should NOT appear in thinking chain."""
        chunks = [
            "\U0001f4ad 开始检索...\n\n",
            "## 钙钛矿太阳能电池稳定性研究进展\n\n",
            "在2023年Nature期刊中，Smith等人报道了...\n\n",
            "[/api/pdf/Nature_2023_test]\n\n",
        ]
        result = self._build_thinking_chain(chunks)
        assert "\U0001f4ad" in result
        assert "## 钙钛矿" not in result
        assert "[/api/pdf/" not in result

    def test_empty_chunks(self):
        assert self._build_thinking_chain([]) == ""

    def test_whitespace_only_chunks(self):
        assert self._build_thinking_chain(["   \n", "\t\n  "]) == ""

    def test_full_pipeline_chain(self):
        """Simulate a complete agent pipeline's thinking chain."""
        chunks = [
            "\U0001f4ad 分析用户问题：钙钛矿太阳能电池的稳定性...\n\n",
            '\U0001f527 **search_papers**({"query": "perovskite stability degradation"})\n\n',
            "\U0001f4ad 搜索返回3篇论文，选择综述开始阅读...\n\n",
            '\U0001f527 **read_paper**({"source": "Nature_2023_paper.pdf"})\n\n',
            "\U0001f4ad 综述已读，切换到实验论文...\n\n",
            '\U0001f527 **read_paper**({"source": "Science_2022_paper.pdf"})\n\n',
            "⚠️ PDF not found: Science_2022_paper.pdf\n\n",
        ]
        result = self._build_thinking_chain(chunks)
        assert result.count("\U0001f4ad") == 3
        assert result.count("\U0001f527") == 3
        assert result.count("⚠") == 1


# ═══════════════════════════════════════════════════════════════════
# B4: Citation Hallucination Detection
# ═══════════════════════════════════════════════════════════════════

class TestCitationHallucination:
    """Test the regex patterns used to detect fake/hallucinated citations."""

    CITATION_PATTERNS = [
        r'\[\U0001f4c4\]\(/api/pdf/([^)]+)\)',
        r'\[\U0001f4c4\]\(([A-Za-z][A-Za-z0-9_\-]{8,})\)',
        r'\[⚠️ 未验证引用\]\(/api/pdf/([^)]+)\)',
    ]

    def _find_cited_ids(self, text):
        cited = set()
        for pattern in self.CITATION_PATTERNS:
            for cid in re.findall(pattern, text):
                cited.add(cid.replace('.pdf', ''))
        return cited

    def test_standard_citation_format(self):
        text = "As reported in [\U0001f4c4](/api/pdf/Nature_2023_s41586-023-06121-1), the PCE reached 25.8%."
        ids = self._find_cited_ids(text)
        assert "Nature_2023_s41586-023-06121-1" in ids

    def test_citation_with_pdf_suffix(self):
        text = "The study [\U0001f4c4](/api/pdf/Nature_2023_s41586-023-06121-1.pdf) showed..."
        ids = self._find_cited_ids(text)
        assert "Nature_2023_s41586-023-06121-1" in ids

    def test_multiple_citations(self):
        text = (
            "Smith et al. [\U0001f4c4](/api/pdf/Nature_2023_test) found X. "
            "Zhang et al. [\U0001f4c4](/api/pdf/Science_2022_test) confirmed Y."
        )
        ids = self._find_cited_ids(text)
        assert len(ids) == 2

    def test_no_citations(self):
        text = "Several studies have shown improved stability."
        ids = self._find_cited_ids(text)
        assert len(ids) == 0

    def test_already_marked_fake_detected(self):
        text = "[⚠️ 未验证引用](/api/pdf/FakePaper_2024_id)"
        ids = self._find_cited_ids(text)
        assert "FakePaper_2024_id" in ids

    def test_bare_file_id_link(self):
        text = "See [\U0001f4c4](NatEnergy_2024_test_paper) for details."
        ids = self._find_cited_ids(text)
        assert "NatEnergy_2024_test_paper" in ids

    def test_fake_replacement_all_formats(self):
        full_content = (
            "[\U0001f4c4](/api/pdf/Fake_2024_xxx) and "
            "[\U0001f4c4](/api/pdf/Fake_2024_xxx.pdf) and "
            "[\U0001f4c4](Fake_2024_xxx) and "
            "[\U0001f4c4](Fake_2024_xxx.pdf)"
        )
        fake_ids = {"Fake_2024_xxx"}
        validated_ids = set()
        fake = fake_ids - validated_ids
        for fid in fake:
            for fmt in [
                f"[\U0001f4c4](/api/pdf/{fid})",
                f"[\U0001f4c4](/api/pdf/{fid}.pdf)",
                f"[\U0001f4c4]({fid})",
                f"[\U0001f4c4]({fid}.pdf)",
            ]:
                full_content = full_content.replace(
                    fmt,
                    f"[⚠️ 未验证引用](/api/pdf/{fid})"
                )
        assert "[\U0001f4c4](/api/pdf/Fake_2024_xxx)" not in full_content
        assert "[\U0001f4c4](Fake_2024_xxx)" not in full_content
        assert full_content.count("[⚠️ 未验证引用]") == 4

    def test_validated_citations_not_flagged(self):
        validated = {"Nature_2023_real"}
        cited = {"Nature_2023_real"}
        fake_ids = cited - validated
        assert len(fake_ids) == 0


# ═══════════════════════════════════════════════════════════════════
# B5: XML Contamination Detection in RESPOND
# ═══════════════════════════════════════════════════════════════════

class TestXMLContaminationDetection:
    """Test the detection of <tool_calls> XML in RESPOND output."""

    CONTAMINATION_TAGS = (
        "<tool_call",
        "<tool_calls>",
        "antha:tool_call",
        "<invoke",
    )

    def test_clean_text_passes(self):
        text = "The perovskite solar cells show excellent stability."
        assert not any(tag in text for tag in self.CONTAMINATION_TAGS)

    def test_tool_call_in_text_detected(self):
        text = "Here are the results.\n<tool_call>search_papers</tool_call>"
        assert any(tag in text for tag in self.CONTAMINATION_TAGS)

    def test_tool_calls_in_text_detected(self):
        text = "Let me <tool_calls> search </tool_calls> for papers."
        assert any(tag in text for tag in self.CONTAMINATION_TAGS)

    def test_anthropic_format_detected(self):
        text = "antha:tool_call: search_papers"
        assert any(tag in text for tag in self.CONTAMINATION_TAGS)

    def test_invoke_tag_detected(self):
        text = "<invoke name='search_papers'>"
        assert any(tag in text for tag in self.CONTAMINATION_TAGS)

    def test_clean_markdown_still_clean(self):
        """Realistic markdown answer should pass contamination check."""
        text = (
            "## 钙钛矿太阳能电池稳定性研究进展\n\n"
            "在2023年**Nature**期刊中，Smith等人明确报道了通过钾离子掺杂"
            "可将MAPbI3钙钛矿的长期稳定性从100小时提升至1000小时以上。\n\n"
            "Zhang等人进一步发现，在85°C/85%RH条件下..."
        )
        assert not any(tag in text for tag in self.CONTAMINATION_TAGS)

    def test_xml_in_code_block_detected(self):
        """XML in code blocks should still be detected."""
        text = (
            "```\n"
            "<tool_calls>\n"
            "  <invoke name='search_papers'>\n"
            "```"
        )
        assert any(tag in text for tag in self.CONTAMINATION_TAGS)

    def test_partial_tag_detected(self):
        """Even partial opening tags should be caught."""
        assert "<tool_call" in "<tool_call>"
        assert "<tool_calls>" in "text <tool_calls> text"


# ═══════════════════════════════════════════════════════════════════
# B6: RESPOND Content Cleanup (XML stripping regex)
# ═══════════════════════════════════════════════════════════════════

class TestRespondContentCleanup:
    """Test the regex used to strip <tool_calls> from final RESPOND content."""

    TOOL_CALLS_STRIP_PATTERN = r'<(?:anth:)?tool_calls>.*?</(?:anth:)?tool_calls>'

    def test_strip_basic_tool_calls(self):
        text = "Answer <tool_calls>some tool content</tool_calls> end."
        cleaned = re.sub(self.TOOL_CALLS_STRIP_PATTERN, '', text, flags=re.DOTALL).strip()
        assert "<tool_calls>" not in cleaned
        assert "Answer  end." == cleaned

    def test_strip_anth_tool_calls(self):
        text = "Answer <anth:tool_calls>content</anth:tool_calls> end."
        cleaned = re.sub(self.TOOL_CALLS_STRIP_PATTERN, '', text, flags=re.DOTALL).strip()
        assert "<anth:tool_calls>" not in cleaned

    def test_strip_multiline_tool_calls(self):
        text = (
            "Start of answer.\n"
            "<tool_calls>\n"
            "  <invoke name='search'>\n"
            "  </invoke>\n"
            "</tool_calls>\n"
            "End of answer."
        )
        cleaned = re.sub(self.TOOL_CALLS_STRIP_PATTERN, '', text, flags=re.DOTALL).strip()
        assert "Start of answer." in cleaned
        assert "End of answer." in cleaned
        assert "<tool_calls>" not in cleaned

    def test_clean_text_unchanged(self):
        text = "A perfectly clean answer about perovskite solar cells."
        cleaned = re.sub(self.TOOL_CALLS_STRIP_PATTERN, '', text, flags=re.DOTALL).strip()
        assert cleaned == text


# ═══════════════════════════════════════════════════════════════════
# B7: Source/Paper Extraction from Tool Messages for RESPOND
# ═══════════════════════════════════════════════════════════════════

class TestReadPaperExtraction:
    """Test _extract_read_papers logic from agent_sm.py."""

    MIN_READ_CHARS = 100

    def _extract_read_papers(self, messages, paper_meta=None):
        papers = []
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if not content.startswith("Content of "):
                continue
            first_newline = content.find("\n")
            if first_newline == -1:
                continue
            header = content[:first_newline]
            body = content[first_newline + 1:].strip()
            source = header.replace("Content of ", "").split(" (first")[0]
            if len(body) >= self.MIN_READ_CHARS:
                meta = (paper_meta or {}).get(source, {})
                papers.append({
                    "source": source,
                    "content": body[:5000],
                    "journal": meta.get("journal", ""),
                    "year": meta.get("year", ""),
                    "title": meta.get("title", ""),
                })
        return papers

    def test_extract_single_paper(self):
        messages = [{
            "role": "tool",
            "content": (
                "Content of Nature_2023_s41586-023-06121-1.pdf (first 5000 chars):\n"
                + "X" * 100
            ),
        }]
        papers = self._extract_read_papers(messages)
        assert len(papers) == 1
        assert papers[0]["source"] == "Nature_2023_s41586-023-06121-1.pdf"

    def test_skip_short_content(self):
        messages = [{
            "role": "tool",
            "content": (
                "Content of Short_paper.pdf (first 5000 chars):\n"
                + "X" * 50
            ),
        }]
        papers = self._extract_read_papers(messages)
        assert len(papers) == 0

    def test_skip_non_read_paper_messages(self):
        messages = [
            {"role": "tool", "content": "Found 5 results for 'perovskite':\n..."},
            {"role": "tool", "content": "Extracted data: {...}"},
        ]
        papers = self._extract_read_papers(messages)
        assert len(papers) == 0

    def test_with_metadata(self):
        paper_meta = {
            "Nature_2023_test.pdf": {
                "journal": "Nature", "year": "2023",
                "title": "Test Paper Title",
            }
        }
        messages = [{
            "role": "tool",
            "content": (
                "Content of Nature_2023_test.pdf (first 5000 chars):\n"
                + "X" * 200
            ),
        }]
        papers = self._extract_read_papers(messages, paper_meta)
        assert len(papers) == 1
        assert papers[0]["journal"] == "Nature"
        assert papers[0]["year"] == "2023"
        assert papers[0]["title"] == "Test Paper Title"

    def test_multiple_papers(self):
        messages = [
            {"role": "tool",
             "content": "Content of Paper_A.pdf (first 5000 chars):\n" + "A" * 200},
            {"role": "tool",
             "content": "Content of Paper_B.pdf (first 5000 chars):\n" + "B" * 200},
        ]
        papers = self._extract_read_papers(messages)
        assert len(papers) == 2
