"""Tests for the dead code detection phase (Phase 10)."""

from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.dead_code import process_dead_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_file_node(graph: KnowledgeGraph, path: str) -> str:
    """Add a File node and return its ID."""
    node_id = generate_id(NodeLabel.FILE, path)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=NodeLabel.FILE,
            name=path.rsplit("/", 1)[-1],
            file_path=path,
        )
    )
    return node_id


def _add_symbol_node(
    graph: KnowledgeGraph,
    label: NodeLabel,
    file_path: str,
    name: str,
    *,
    is_entry_point: bool = False,
    is_exported: bool = False,
    class_name: str = "",
) -> str:
    """Add a symbol node and return its ID."""
    symbol_name = (
        f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
    )
    node_id = generate_id(label, file_path, symbol_name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=label,
            name=name,
            file_path=file_path,
            class_name=class_name,
            is_entry_point=is_entry_point,
            is_exported=is_exported,
        )
    )
    return node_id


def _add_calls_relationship(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> None:
    """Add a CALLS relationship from *source_id* to *target_id*."""
    rel_id = f"calls:{source_id}->{target_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source_id,
            target=target_id,
        )
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the test fixture specification.

    - Function:src/main.py:main         (entry point, no incoming calls)
    - Function:src/auth.py:validate     (has incoming calls from main)
    - Function:src/auth.py:unused_helper (no calls, not entry point) -> DEAD
    - Method:src/models.py:User.__init__ (no calls, constructor)    -> NOT dead
    - Function:src/tests/test_auth.py:test_validate (test function) -> NOT dead
    - Function:src/utils.py:orphan_function (no calls, not entry)   -> DEAD
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/main.py")
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/models.py")
    _add_file_node(g, "src/tests/test_auth.py")
    _add_file_node(g, "src/utils.py")

    # Symbols
    main_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/main.py", "main", is_entry_point=True
    )
    validate_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/auth.py", "validate"
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/auth.py", "unused_helper"
    )
    _add_symbol_node(
        g,
        NodeLabel.METHOD,
        "src/models.py",
        "__init__",
        class_name="User",
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/tests/test_auth.py", "test_validate"
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/utils.py", "orphan_function"
    )

    # CALLS: main -> validate
    _add_calls_relationship(g, main_id, validate_id)

    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectsUnusedFunction:
    """Unused helper functions with no incoming calls are flagged as dead."""

    def test_detects_unused_function(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        unused_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "unused_helper"
        )
        node = graph.get_node(unused_id)
        assert node is not None
        assert node.is_dead is True


class TestSkipsEntryPoints:
    """Entry points are never flagged as dead, even without incoming calls."""

    def test_skips_entry_points(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        main_id = generate_id(NodeLabel.FUNCTION, "src/main.py", "main")
        node = graph.get_node(main_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsCalledFunctions:
    """Functions with incoming CALLS relationships are not flagged."""

    def test_skips_called_functions(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        validate_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        node = graph.get_node(validate_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsConstructors:
    """__init__ and __new__ methods are never flagged as dead."""

    def test_skips_constructors(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        init_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__init__"
        )
        node = graph.get_node(init_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsTestFunctions:
    """Test functions (test_*) are never flagged as dead."""

    def test_skips_test_functions(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        test_id = generate_id(
            NodeLabel.FUNCTION, "src/tests/test_auth.py", "test_validate"
        )
        node = graph.get_node(test_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsDunderMethods:
    """Dunder methods (__str__, __repr__, etc.) are never flagged as dead."""

    def test_skips_dunder_methods(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "__str__",
            class_name="User",
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "__repr__",
            class_name="User",
        )

        process_dead_code(g)

        str_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__str__"
        )
        repr_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__repr__"
        )

        str_node = g.get_node(str_id)
        repr_node = g.get_node(repr_id)

        assert str_node is not None
        assert str_node.is_dead is False

        assert repr_node is not None
        assert repr_node.is_dead is False


class TestReturnsCount:
    """process_dead_code returns the correct count of dead symbols."""

    def test_returns_count(self, graph: KnowledgeGraph) -> None:
        count = process_dead_code(graph)

        # unused_helper and orphan_function are the two dead symbols.
        assert count == 2


class TestEmptyGraph:
    """An empty graph produces zero dead symbols."""

    def test_empty_graph(self) -> None:
        g = KnowledgeGraph()
        count = process_dead_code(g)
        assert count == 0
