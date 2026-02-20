"""Phase 10: Dead code detection for Axon.

Scans the knowledge graph to find unreachable symbols (functions, methods,
classes) that have zero incoming CALLS relationships and are not entry points,
exported, constructors, test functions, or dunder methods.  Flags them by
setting ``is_dead = True`` on the corresponding graph node.
"""

from __future__ import annotations

import logging

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel, RelType

logger = logging.getLogger(__name__)

# Labels that represent callable symbols eligible for dead-code analysis.
_SYMBOL_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)

# Constructor method names that should never be flagged as dead because they
# are invoked implicitly when a class is instantiated.
_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"__init__", "__new__"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_constructor(name: str) -> bool:
    """Return ``True`` if *name* is a Python constructor method."""
    return name in _CONSTRUCTOR_NAMES


def _is_test_function(name: str) -> bool:
    """Return ``True`` if *name* looks like a test function (``test_*``)."""
    return name.startswith("test_")


def _is_test_class(name: str) -> bool:
    """Return ``True`` if *name* follows pytest class convention (``Test*``).

    Matches names starting with ``Test`` where the next character is uppercase,
    e.g. ``TestHandleQuery``, ``TestBulkLoad``.
    """
    return len(name) > 4 and name.startswith("Test") and name[4].isupper()


def _is_test_file(file_path: str) -> bool:
    """Return ``True`` if the file is in a test directory or is a test file.

    Matches paths containing ``/tests/`` or files named ``test_*.py``.
    """
    return "/tests/" in file_path or "/test_" in file_path


def _is_dunder(name: str) -> bool:
    """Return ``True`` if *name* is a dunder (double-underscore) method.

    Dunders start and end with ``__`` and have at least one character in
    between (e.g. ``__str__``, ``__repr__``).
    """
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _has_incoming_calls(graph: KnowledgeGraph, node_id: str) -> bool:
    """Return ``True`` if *node_id* has at least one incoming CALLS edge."""
    return graph.has_incoming(node_id, RelType.CALLS)


def _is_python_public_api(name: str, file_path: str) -> bool:
    """Return ``True`` if *name* looks like a public Python API symbol.

    In Python, symbols whose names don't start with ``_`` are public by
    convention.  We only apply this to top-level functions and classes
    in ``__init__.py`` files, since those are most likely to be used
    externally without an explicit call relationship in the codebase.
    """
    if not file_path.endswith(".py"):
        return False
    if file_path.endswith("__init__.py") and not name.startswith("_"):
        return True
    return False


def _is_exempt(
    name: str, is_entry_point: bool, is_exported: bool, file_path: str = ""
) -> bool:
    """Return ``True`` if the symbol is exempt from dead-code flagging.

    A symbol is exempt when ANY of the following hold:

    - It is marked as an entry point.
    - It is marked as exported (may be used externally).
    - It is a constructor (``__init__`` / ``__new__``).
    - It is a test function (name starts with ``test_``).
    - It is a test class (name starts with ``Test``).
    - It lives in a test file (fixtures, helpers are not dead code).
    - It is a dunder method (``__str__``, ``__repr__``, etc.).
    - It is a public symbol in a Python ``__init__.py`` file.
    """
    return (
        is_entry_point
        or is_exported
        or _is_constructor(name)
        or _is_test_function(name)
        or _is_test_class(name)
        or _is_test_file(file_path)
        or _is_dunder(name)
        or _is_python_public_api(name, file_path)
    )


def _clear_override_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods that override a non-dead base class method.

    When ``A extends B`` and ``B.method`` is called, ``A.method`` (the
    override) has zero incoming CALLS and gets flagged dead.  This pass
    detects that situation and clears ``is_dead`` on the override.

    Returns the number of overrides un-flagged.
    """
    # Build a mapping: class_name -> set of method names that are NOT dead.
    alive_methods_by_class: dict[str, set[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead and method.class_name:
            alive_methods_by_class.setdefault(method.class_name, set()).add(method.name)

    # Build child -> parent class mapping from EXTENDS relationships.
    child_to_parents: dict[str, list[str]] = {}
    for rel in graph.get_relationships_by_type(RelType.EXTENDS):
        # source is the child class node id, target is the parent class node id.
        child_node = graph.get_node(rel.source)
        parent_node = graph.get_node(rel.target)
        if child_node and parent_node:
            child_to_parents.setdefault(child_node.name, []).append(parent_node.name)

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue

        # Check if any parent class has an alive method with the same name.
        parent_classes = child_to_parents.get(method.class_name, [])
        for parent_name in parent_classes:
            alive_in_parent = alive_methods_by_class.get(parent_name, set())
            if method.name in alive_in_parent:
                method.is_dead = False
                cleared += 1
                logger.debug("Un-flagged override: %s.%s", method.class_name, method.name)
                break

    return cleared


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_dead_code(graph: KnowledgeGraph) -> int:
    """Detect dead (unreachable) symbols and flag them in the graph.

    A symbol is considered dead when **all** of the following are true:

    1. It has zero incoming ``CALLS`` relationships.
    2. It is not an entry point (``is_entry_point == False``).
    3. It is not exported (``is_exported == False``).
    4. It is not a class constructor (``__init__`` / ``__new__``).
    5. It is not a test function (name starts with ``test_``).
    6. It is not a test class (name starts with ``Test``).
    7. It is not in a test file (fixtures/helpers are exempt).
    8. It is not a dunder method (name starts and ends with ``__``).

    After the initial pass, a second pass un-flags method overrides whose
    base class method is called (resolves dynamic dispatch false positives).

    For each dead symbol the function sets ``node.is_dead = True``.

    Args:
        graph: The knowledge graph to scan and mutate.

    Returns:
        The total number of symbols flagged as dead.
    """
    dead_count = 0

    for label in _SYMBOL_LABELS:
        for node in graph.get_nodes_by_label(label):
            # Skip symbols that are exempt from dead-code analysis.
            if _is_exempt(node.name, node.is_entry_point, node.is_exported, node.file_path):
                continue

            # Skip symbols that have incoming CALLS edges.
            if _has_incoming_calls(graph, node.id):
                continue

            # Flag the symbol as dead.
            node.is_dead = True
            dead_count += 1
            logger.debug("Dead symbol: %s (%s)", node.name, node.id)

    # Second pass: un-flag overrides of called base-class methods.
    cleared = _clear_override_false_positives(graph)
    dead_count -= cleared

    return dead_count
