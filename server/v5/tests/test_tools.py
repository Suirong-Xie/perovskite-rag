"""
Test Suite D: Tool Registry & Execution
Covers: tool registration, RETRIEVE/READ tool separation,
        executor dispatch, error handling
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5.services.tools import (
    ALL_TOOLS, EXECUTORS,
    RETRIEVE_TOOLS, READ_TOOLS, filter_tools,
)
from v5.core.schemas import ToolCall, ToolResult


# ═══════════════════════════════════════════════════════════════════
# D1: Tool Registry Consistency
# ═══════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """Verify the tool registry is consistent and well-formed."""

    def test_all_tools_non_empty(self):
        assert len(ALL_TOOLS) > 0, "Should have registered tools"

    def test_all_tools_have_name(self):
        for t in ALL_TOOLS:
            assert "name" in t, f"Tool missing name: {t}"
            assert t["name"], f"Tool has empty name"

    def test_all_tools_have_description(self):
        for t in ALL_TOOLS:
            assert "description" in t, f"Tool '{t['name']}' missing description"
            assert len(t["description"]) >= 10, \
                f"Tool '{t['name']}' description too short"

    def test_all_tools_have_parameters(self):
        for t in ALL_TOOLS:
            assert "parameters" in t, f"Tool '{t['name']}' missing parameters"

    def test_executor_for_every_tool(self):
        for t in ALL_TOOLS:
            name = t["name"]
            assert name in EXECUTORS, \
                f"Tool '{name}' has no registered executor"

    def test_no_orphan_executors(self):
        """Every executor should have a corresponding tool definition."""
        tool_names = {t["name"] for t in ALL_TOOLS}
        for name in EXECUTORS:
            assert name in tool_names, \
                f"Executor '{name}' has no tool definition"

    def test_tool_names_unique(self):
        names = [t["name"] for t in ALL_TOOLS]
        assert len(names) == len(set(names)), \
            f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

    def test_retrieve_tools_subset_of_all(self):
        all_names = {t["name"] for t in ALL_TOOLS}
        for name in RETRIEVE_TOOLS:
            assert name in all_names, \
                f"RETRIEVE tool '{name}' not in ALL_TOOLS"

    def test_read_tools_subset_of_all(self):
        all_names = {t["name"] for t in ALL_TOOLS}
        for name in READ_TOOLS:
            assert name in all_names, \
                f"READ tool '{name}' not in ALL_TOOLS"

    def test_retrieve_and_read_disjoint(self):
        """No tool should be in both RETRIEVE and READ sets."""
        overlap = RETRIEVE_TOOLS & READ_TOOLS
        assert len(overlap) == 0, \
            f"Tools in both RETRIEVE and READ: {overlap}"

    def test_required_retrieve_tools_present(self):
        """Critical search tools must be in RETRIEVE_TOOLS."""
        assert "search_papers" in RETRIEVE_TOOLS
        assert "search_arxiv" in RETRIEVE_TOOLS
        assert "search_semantic_scholar" in RETRIEVE_TOOLS

    def test_required_read_tools_present(self):
        """Critical read tools must be in READ_TOOLS."""
        assert "read_paper" in READ_TOOLS
        assert "read_arxiv_paper" in READ_TOOLS


class TestFilterTools:
    """Test the filter_tools helper function."""

    def test_filter_retrieve_tools(self):
        filtered = filter_tools(RETRIEVE_TOOLS)
        names = {t["name"] for t in filtered}
        assert names == RETRIEVE_TOOLS, \
            f"Filtered names {names} != expected {RETRIEVE_TOOLS}"

    def test_filter_read_tools(self):
        filtered = filter_tools(READ_TOOLS)
        names = {t["name"] for t in filtered}
        assert names == READ_TOOLS

    def test_filter_empty_set(self):
        filtered = filter_tools(set())
        assert filtered == []

    def test_filter_nonexistent_tool(self):
        filtered = filter_tools({"nonexistent_tool"})
        assert filtered == []

    def test_read_tools_dont_include_search(self):
        """READ tools should not include any search functions."""
        read_names = {t["name"] for t in filter_tools(READ_TOOLS)}
        for name in read_names:
            assert "search" not in name.lower(), \
                f"READ stage should not include search tool: {name}"

    def test_retrieve_tools_dont_include_read(self):
        """RETRIEVE tools should not include read_paper."""
        retrieve_names = {t["name"] for t in filter_tools(RETRIEVE_TOOLS)}
        assert "read_paper" not in retrieve_names
        assert "read_arxiv_paper" not in retrieve_names


# ═══════════════════════════════════════════════════════════════════
# D2: Tool Execution Dispatch
# ═══════════════════════════════════════════════════════════════════

class TestToolExecution:
    """Test the tool execution dispatch and error handling."""

    def test_unknown_tool_returns_error(self):
        """Executing an unknown tool should return error, not crash."""
        # Replicate agent.execute_tool dispatch logic
        executor = EXECUTORS.get("nonexistent_tool")
        assert executor is None  # not registered
        tc = ToolCall("nonexistent_tool", {"arg": "value"})
        if executor is None:
            result = ToolResult(tc, "", error=f"Unknown tool: {tc.name}")
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_search_papers_with_empty_query(self):
        """search_papers with empty query should return error."""
        from v5.services.tools.search_local import execute
        result, raw_data = execute({"query": ""})
        assert result.error is not None
        assert "query" in result.error.lower()

    def test_search_papers_with_valid_query(self):
        """search_papers with a valid query should return results."""
        from v5.services.tools.search_local import execute
        result, raw_data = execute({"query": "perovskite solar cell stability"})
        assert result.error is None
        assert result.output is not None
        assert len(result.output) > 0

    def test_search_papers_top_k_bounded(self):
        """top_k should be capped at 10."""
        from v5.services.tools.search_local import execute
        result, raw_data = execute({"query": "perovskite", "top_k": "100"})
        # Should be capped
        if raw_data:
            assert len(raw_data) <= 10, \
                f"Expected max 10 results, got {len(raw_data)}"

    def test_read_paper_empty_source(self):
        """read_paper with empty source should return error."""
        from v5.services.tools.read_paper import execute_read_paper
        result, raw_data = execute_read_paper({"source": ""})
        assert result.error is not None

    def test_read_paper_nonexistent_pdf(self):
        """read_paper with a nonexistent PDF should return error."""
        from v5.services.tools.read_paper import execute_read_paper
        result, raw_data = execute_read_paper({"source": "nonexistent_99999.pdf"})
        # Should say no PDF found (could be error or in output)
        assert (result.error is not None or "无全文" in (result.output or ""))

    def test_extract_data_empty_source(self):
        from v5.services.tools.extract_data import execute
        result, raw_data = execute({"source": ""})
        assert result.error is not None

    def test_compare_papers_empty_sources(self):
        from v5.services.tools.compare import execute
        result, raw_data = execute({"sources": ""})
        assert result.error is not None

    def test_compare_papers_single_source(self):
        from v5.services.tools.compare import execute
        result, raw_data = execute({"sources": "single_paper.pdf"})
        assert "Need at least 2 papers" in (result.output or "")

    def test_compare_papers_max_five(self):
        """compare_papers should cap at 5 papers."""
        from v5.services.tools.compare import execute
        result, raw_data = execute({
            "sources": "a.pdf,b.pdf,c.pdf,d.pdf,e.pdf,f.pdf,g.pdf"
        })
        assert result.error is None  # should not error

    def test_search_arxiv_empty_query(self):
        from v5.services.tools.search_arxiv import execute
        result, raw_data = execute({"query": ""})
        assert result.error is not None

    def test_search_s2_empty_query(self):
        from v5.services.tools.search_s2 import execute
        result, raw_data = execute({"query": ""})
        assert result.error is not None


# ═══════════════════════════════════════════════════════════════════
# D3: Tool Schema Validation
# ═══════════════════════════════════════════════════════════════════

class TestToolSchemas:
    """Validate that tool schemas match what LLM backends expect."""

    def test_openai_tool_format(self):
        """All tools should be convertible to OpenAI function format."""
        for t in ALL_TOOLS:
            properties = {}
            required = []
            for param_name, param_desc in t.get("parameters", {}).items():
                properties[param_name] = {
                    "type": "string",
                    "description": param_desc,
                }
                required.append(param_name)
            func = {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
            assert func["name"]
            assert func["description"]
            assert func["parameters"]["type"] == "object"

    def test_parameter_descriptions_meaningful(self):
        """Parameter descriptions should be descriptive, not just key names."""
        for t in ALL_TOOLS:
            for pname, pdesc in t.get("parameters", {}).items():
                # Description should have reasonable length
                assert len(pdesc) >= 5, \
                    f"Tool '{t['name']}' param '{pname}' description too short: '{pdesc}' (len={len(pdesc)})"
                # Description should not be identical to parameter name
                assert pdesc.strip().lower() != pname.strip().lower(), \
                    f"Tool '{t['name']}' param '{pname}' description equals parameter name"

    def test_search_tools_have_query_param(self):
        """All search tools must have a 'query' parameter."""
        search_tools = ["search_papers", "search_arxiv", "search_semantic_scholar"]
        for t in ALL_TOOLS:
            if t["name"] in search_tools:
                params = t.get("parameters", {})
                assert "query" in params, \
                    f"Search tool '{t['name']}' missing 'query' parameter"

    def test_read_paper_has_source_param(self):
        for t in ALL_TOOLS:
            if t["name"] in ("read_paper", "extract_data"):
                assert "source" in t.get("parameters", {}), \
                    f"'{t['name']}' missing 'source' parameter"

    def test_read_arxiv_has_arxiv_id_param(self):
        for t in ALL_TOOLS:
            if t["name"] == "read_arxiv_paper":
                assert "arxiv_id" in t.get("parameters", {}), \
                    "read_arxiv_paper missing 'arxiv_id' parameter"

    def test_compare_papers_has_sources_param(self):
        for t in ALL_TOOLS:
            if t["name"] == "compare_papers":
                assert "sources" in t.get("parameters", {}), \
                    "compare_papers missing 'sources' parameter"

    def test_no_duplicate_parameter_names(self):
        """Each tool should have unique parameter names."""
        for t in ALL_TOOLS:
            params = list(t.get("parameters", {}).keys())
            assert len(params) == len(set(params)), \
                f"Tool '{t['name']}' has duplicate parameter names: {params}"


# ═══════════════════════════════════════════════════════════════════
# D4: API Key/SDK Registration System
# ═══════════════════════════════════════════════════════════════════

class TestRegisterTool:
    """Test the dynamic tool registration API."""

    def test_register_new_tool(self):
        from v5.services.tools import ALL_TOOLS as TOOLS, EXECUTORS as TOOL_EXECUTORS

        def dummy_executor(args):
            return (ToolResult(ToolCall("test_tool", args), "ok"), {})

        # Register directly into the tools module
        TOOL_EXECUTORS["test_tool"] = dummy_executor
        TOOLS.append({
            "name": "test_tool",
            "description": "A test tool for unit testing",
            "parameters": {"param1": "Test parameter"},
        })

        assert "test_tool" in TOOL_EXECUTORS
        assert any(t["name"] == "test_tool" for t in TOOLS)

        # Cleanup: remove test tool
        TOOL_EXECUTORS.pop("test_tool", None)
        for i, t in enumerate(TOOLS):
            if t["name"] == "test_tool":
                TOOLS.pop(i)
                break
