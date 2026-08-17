"""
Test fixtures for PerovskiteGPT v1.5 agent test suite.
Mock external dependencies that may not be available in test environment.
"""
import sys
import os
import types
from unittest.mock import MagicMock


# ── Auto-mocking import hook for missing dependencies ──
# Creates mock modules on the fly for any import that starts with
# known third-party package names not available in test env.

_MOCK_PACKAGES = {"pymatgen", "mp_api", "pymupdf4llm"}


def _make_mock_module(fullname):
    """Create a proper mock module that supports submodule traversal."""
    if fullname in sys.modules:
        return sys.modules[fullname]

    mod = types.ModuleType(fullname)
    mod.__file__ = f"<mock:{fullname}>"
    mod.__package__ = fullname.rsplit(".", 1)[0] if "." in fullname else fullname
    mod.__path__ = []  # required for package traversal
    mod.__loader__ = None
    mod.__spec__ = None

    # Add common pymatgen exports as MagicMock attributes
    # so that `from pymatgen.core import Composition` returns a mock
    mod.Composition = MagicMock(name=f"{fullname}.Composition")
    mod.Element = MagicMock(name=f"{fullname}.Element")
    mod.Structure = MagicMock(name=f"{fullname}.Structure")
    mod.BVAnalyzer = MagicMock(name=f"{fullname}.BVAnalyzer")

    sys.modules[fullname] = mod
    return mod


class _AutoMockFinder:
    """A meta path finder that creates mock modules for missing packages."""

    def find_spec(self, fullname, path, target=None):
        # Check if it's a submodule of a known mock package
        for pkg in _MOCK_PACKAGES:
            if fullname == pkg or fullname.startswith(pkg + "."):
                # Return a ModuleSpec for the mock
                from importlib.machinery import ModuleSpec
                _make_mock_module(fullname)
                return ModuleSpec(fullname, None)
        return None


# Install the auto-mock finder
sys.meta_path.insert(0, _AutoMockFinder())

# Pre-populate top-level mock packages
for pkg in _MOCK_PACKAGES:
    _make_mock_module(pkg)


# Now safe to import pytest and app modules
import pytest

# Ensure app package is importable
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


@pytest.fixture
def mock_messages():
    """Minimal message list with user question and pre-search."""
    return [
        {"role": "system", "content": "You are a perovskite research assistant."},
        {"role": "user", "content": (
            "钙钛矿太阳能电池的稳定性研究进展如何？\n\n"
            "系统已为你预检索了以下文献：\n\n"
            "📄 **可阅读全文的论文**：\n"
            "[P1] Nature | File ID: `Nature_2023_s41586-023-06121-1` | "
            "PDF: /api/pdf/Nature_2023_s41586-023-06121-1\n"
            "    内容: This study investigates the long-term stability of perovskite solar cells...\n\n"
            "[P2] NatEnergy | File ID: `NatEnergy_2024_s41560-024-01234-5` | "
            "PDF: /api/pdf/NatEnergy_2024_s41560-024-01234-5\n"
            "    内容: We report a comprehensive review of degradation mechanisms...\n\n"
            "🔗 **仅有摘要/元数据的论文**：\n"
            "[S1] Science | DOI: https://doi.org/10.1126/science.abc1234\n"
            "    摘要: A brief overview of recent advances in perovskite stability..."
        )},
    ]


@pytest.fixture
def state_context():
    """Fresh StateContext for testing."""
    from app.services.agent_sm import StateContext
    return StateContext()


@pytest.fixture
def preset_context(state_context):
    """StateContext with pre-populated paper data."""
    state_context.fulltext_sources = {
        "Nature_2023_s41586-023-06121-1.pdf",
        "NatEnergy_2024_s41560-024-01234-5.pdf",
    }
    state_context.unknown_sources = {
        "Science_2022_science.abc1234.pdf",
        "ACS_Energy_Lett_2023_acs.5678.pdf",
        "AdvMater_2024_adma.9012.pdf",
    }
    state_context.nofulltext_sources = {
        "JACS_2021_jacs.3456.pdf",
    }
    state_context.paper_meta = {
        "Nature_2023_s41586-023-06121-1.pdf": {
            "title": "Long-term stability of perovskite solar cells",
            "journal": "Nature",
            "year": "2023",
            "content_preview": "We investigate the long-term operational stability...",
        },
        "NatEnergy_2024_s41560-024-01234-5.pdf": {
            "title": "A comprehensive review of degradation mechanisms",
            "journal": "NatEnergy",
            "year": "2024",
            "content_preview": "This review summarizes recent progress...",
        },
        "Science_2022_science.abc1234.pdf": {
            "title": "Recent advances in perovskite stability",
            "journal": "Science",
            "year": "2022",
            "content_preview": "Perovskite solar cells have achieved remarkable progress...",
        },
        "ACS_Energy_Lett_2023_acs.5678.pdf": {
            "title": "Ion migration in mixed halide perovskites",
            "journal": "ACS Energy Letters",
            "year": "2023",
            "content_preview": "Ion migration is a key degradation pathway...",
        },
        "AdvMater_2024_adma.9012.pdf": {
            "title": "Interface engineering for stable perovskite solar cells review",
            "journal": "Advanced Materials",
            "year": "2024",
            "content_preview": "Interface engineering has emerged as a promising strategy...",
        },
        "JACS_2021_jacs.3456.pdf": {
            "title": "Moisture-induced degradation of MAPbI3",
            "journal": "JACS",
            "year": "2021",
            "content_preview": "We study the effect of humidity on MAPbI3 films...",
        },
    }
    return state_context
