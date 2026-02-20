"""Phase 6: Heritage extraction for Axon.

Takes FileParseData from the parser phase and creates EXTENDS / IMPLEMENTS
relationships between Class and Interface nodes in the knowledge graph.

Heritage tuples have the shape ``(class_name, kind, parent_name)`` where
*kind* is either ``"extends"`` or ``"implements"``.
"""

from __future__ import annotations

import logging

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
)
from axon.core.ingestion.parser_phase import FileParseData

logger = logging.getLogger(__name__)

# Labels that participate in heritage relationships.
_HERITAGE_LABELS: tuple[NodeLabel, ...] = (NodeLabel.CLASS, NodeLabel.INTERFACE)

_KIND_TO_REL: dict[str, RelType] = {
    "extends": RelType.EXTENDS,
    "implements": RelType.IMPLEMENTS,
}


# ---------------------------------------------------------------------------
# Symbol index
# ---------------------------------------------------------------------------


def build_symbol_index(graph: KnowledgeGraph) -> dict[str, list[str]]:
    """Build a mapping from symbol names to their node IDs.

    Only Class and Interface nodes are included.  Multiple symbols can
    share the same name across different files, so the value is a list.

    Args:
        graph: The knowledge graph containing parsed symbol nodes.

    Returns:
        A dict mapping symbol name to a list of node IDs.
    """
    index: dict[str, list[str]] = {}
    for label in _HERITAGE_LABELS:
        for node in graph.get_nodes_by_label(label):
            index.setdefault(node.name, []).append(node.id)
    return index


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_node(
    name: str,
    file_path: str,
    symbol_index: dict[str, list[str]],
    graph: KnowledgeGraph,
) -> str | None:
    """Resolve a symbol *name* to a node ID, preferring same-file matches.

    1. Check whether the global index contains *name*.
    2. Prefer any candidate defined in the same *file_path*.
    3. Fall back to the first candidate (cross-file reference).

    Returns:
        The node ID if resolved, otherwise ``None``.
    """
    candidate_ids = symbol_index.get(name)
    if not candidate_ids:
        return None

    # Prefer a same-file match.
    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path == file_path:
            return nid

    # Fall back to the first candidate (cross-file reference).
    return candidate_ids[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_heritage(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> None:
    """Create EXTENDS and IMPLEMENTS relationships from heritage tuples.

    For each ``(class_name, kind, parent_name)`` tuple in the parse results:

    * Resolve *class_name* and *parent_name* to existing graph nodes,
      preferring nodes defined in the same file.
    * If both nodes are found, add a relationship of the appropriate type.
    * If either node cannot be resolved (e.g. an external parent class),
      the tuple is silently skipped.

    Args:
        parse_data: File parse results produced by the parser phase.
        graph: The knowledge graph to populate with heritage relationships.
    """
    symbol_index = build_symbol_index(graph)

    for fpd in parse_data:
        for class_name, kind, parent_name in fpd.parse_result.heritage:
            rel_type = _KIND_TO_REL.get(kind)
            if rel_type is None:
                logger.warning(
                    "Unknown heritage kind %r for %s in %s, skipping",
                    kind,
                    class_name,
                    fpd.file_path,
                )
                continue

            child_id = _resolve_node(
                class_name, fpd.file_path, symbol_index, graph
            )
            parent_id = _resolve_node(
                parent_name, fpd.file_path, symbol_index, graph
            )

            if child_id is None or parent_id is None:
                logger.debug(
                    "Skipping heritage %s %s %s in %s: unresolved node(s)",
                    class_name,
                    kind,
                    parent_name,
                    fpd.file_path,
                )
                continue

            rel_id = f"{kind}:{child_id}->{parent_id}"
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=rel_type,
                    source=child_id,
                    target=parent_id,
                )
            )
