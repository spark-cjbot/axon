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


def _is_dunder(name: str) -> bool:
    """Return ``True`` if *name* is a dunder (double-underscore) method.

    Dunders start and end with ``__`` and have at least one character in
    between (e.g. ``__str__``, ``__repr__``).
    """
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _has_incoming_calls(graph: KnowledgeGraph, node_id: str) -> bool:
    """Return ``True`` if *node_id* has at least one incoming CALLS edge."""
    return len(graph.get_incoming(node_id, RelType.CALLS)) > 0


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
    - It is a dunder method (``__str__``, ``__repr__``, etc.).
    - It is a public symbol in a Python ``__init__.py`` file.
    """
    return (
        is_entry_point
        or is_exported
        or _is_constructor(name)
        or _is_test_function(name)
        or _is_dunder(name)
        or _is_python_public_api(name, file_path)
    )


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
    6. It is not a dunder method (name starts and ends with ``__``).

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

    return dead_count
