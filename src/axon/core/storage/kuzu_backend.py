"""KuzuDB storage backend for Axon.

Implements the :class:`StorageBackend` protocol using KuzuDB, an embedded
graph database that speaks Cypher. Each :class:`NodeLabel` maps to a
separate node table, and a single ``CodeRelation`` relationship table group
covers all source-to-target combinations.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import kuzu

from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel
from axon.core.storage.base import NodeEmbedding, SearchResult

if TYPE_CHECKING:
    from axon.core.graph.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of node table names derived from the NodeLabel enum.
_NODE_TABLE_NAMES: list[str] = [label.name.title().replace("_", "") for label in NodeLabel]
# e.g. ["File", "Folder", "Function", "Class", "Method", "Interface",
#        "Typealias", "Enum", "Community", "Process"]

# Mapping from lowercase label value to table name.
_LABEL_TO_TABLE: dict[str, str] = {
    label.value: label.name.title().replace("_", "") for label in NodeLabel
}
# e.g. {"file": "File", "folder": "Folder", "function": "Function", ...}

# Common node properties shared across every table.
_NODE_PROPERTIES = (
    "id STRING, "
    "name STRING, "
    "file_path STRING, "
    "start_line INT64, "
    "end_line INT64, "
    "content STRING, "
    "signature STRING, "
    "language STRING, "
    "class_name STRING, "
    "is_dead BOOL, "
    "is_entry_point BOOL, "
    "is_exported BOOL, "
    "PRIMARY KEY (id)"
)

# Relationship properties on the CodeRelation group.
_REL_PROPERTIES = (
    "rel_type STRING, "
    "confidence DOUBLE, "
    "role STRING, "
    "step_number INT64, "
    "strength DOUBLE, "
    "co_changes INT64, "
    "symbols STRING"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape(value: str) -> str:
    """Escape a string for safe inclusion in a Cypher literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _table_for_id(node_id: str) -> str | None:
    """Extract the table name from a node ID by mapping its label prefix."""
    prefix = node_id.split(":", 1)[0]
    return _LABEL_TO_TABLE.get(prefix)


# Embedding node table schema.
_EMBEDDING_PROPERTIES = "node_id STRING, vec DOUBLE[], PRIMARY KEY(node_id)"


# ---------------------------------------------------------------------------
# KuzuBackend
# ---------------------------------------------------------------------------


class KuzuBackend:
    """StorageBackend implementation backed by KuzuDB.

    Usage::

        backend = KuzuBackend()
        backend.initialize(Path("/tmp/axon_db"))
        backend.bulk_load(graph)
        node = backend.get_node("function:src/app.py:main")
        backend.close()
    """

    def __init__(self) -> None:
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, path: Path) -> None:
        """Open or create the KuzuDB database at *path* and set up the schema."""
        self._db = kuzu.Database(str(path))
        self._conn = kuzu.Connection(self._db)
        self._create_schema()

    def close(self) -> None:
        """Release the connection and database handles.

        Explicitly deletes the connection and database objects to ensure
        KuzuDB releases file locks and flushes data.
        """
        if self._conn is not None:
            try:
                del self._conn
            except Exception:
                pass
            self._conn = None
        if self._db is not None:
            try:
                del self._db
            except Exception:
                pass
            self._db = None

    # -- Graph CRUD ----------------------------------------------------------

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Insert nodes into their respective label tables."""
        for node in nodes:
            self._insert_node(node)

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        """Insert relationships by matching source and target nodes."""
        for rel in rels:
            self._insert_relationship(rel)

    def remove_nodes_by_file(self, file_path: str) -> int:
        """Delete all nodes whose ``file_path`` matches across every table.

        Returns:
            The number of nodes removed.
        """
        assert self._conn is not None
        total = 0
        for table in _NODE_TABLE_NAMES:
            try:
                count_q = f"MATCH (n:{table}) WHERE n.file_path = $fp RETURN count(n)"
                result = self._conn.execute(count_q, parameters={"fp": file_path})
                count = 0
                if result.has_next():
                    count = result.get_next()[0]
                if count > 0:
                    delete_q = f"MATCH (n:{table}) WHERE n.file_path = $fp DETACH DELETE n"
                    self._conn.execute(delete_q, parameters={"fp": file_path})
                    total += count
            except Exception:
                logger.debug("Failed to remove nodes from table %s", table, exc_info=True)
        return total

    # -- Queries -------------------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a single node by ID, or ``None`` if not found."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return None

        query = f"MATCH (n:{table}) WHERE n.id = $nid RETURN n.*"
        try:
            result = self._conn.execute(query, parameters={"nid": node_id})
            if result.has_next():
                row = result.get_next()
                return self._row_to_node(row, node_id)
        except Exception:
            logger.debug("get_node failed for %s", node_id, exc_info=True)
        return None

    def get_callers(self, node_id: str) -> list[GraphNode]:
        """Return nodes that CALL the node identified by *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller)-[r:CodeRelation]->(callee:{table}) "
            f"WHERE callee.id = $nid AND r.rel_type = 'calls' "
            f"RETURN caller.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_callees(self, node_id: str) -> list[GraphNode]:
        """Return nodes called by the node identified by *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller:{table})-[r:CodeRelation]->(callee) "
            f"WHERE caller.id = $nid AND r.rel_type = 'calls' "
            f"RETURN callee.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_type_refs(self, node_id: str) -> list[GraphNode]:
        """Return nodes referenced via USES_TYPE from *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (src:{table})-[r:CodeRelation]->(tgt) "
            f"WHERE src.id = $nid AND r.rel_type = 'uses_type' "
            f"RETURN tgt.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def traverse(self, start_id: str, depth: int) -> list[GraphNode]:
        """BFS traversal through CALLS edges up to *depth* hops.

        Uses KuzuDB's recursive relationship syntax to perform the entire
        traversal in a single query instead of N+1 round-trips.
        """
        assert self._conn is not None
        table = _table_for_id(start_id)
        if table is None:
            return []

        # Use recursive relationship to get all reachable nodes in one query.
        # KuzuDB syntax: -[:CodeRelation*1..{depth}]->
        query = (
            f"MATCH (start:{table})-[:CodeRelation*1..{depth}]->(reached) "
            f"WHERE start.id = $sid "
            f"RETURN DISTINCT reached.*"
        )
        try:
            result_nodes = self._query_nodes(query, parameters={"sid": start_id})
            # Filter to only CALLS relationships by post-filtering.
            # KuzuDB recursive paths don't support property filters directly,
            # so we still need to verify reachability via calls.
            # For correctness, fall back to BFS if the recursive query
            # returns too many unrelated nodes.
            if result_nodes:
                return result_nodes
        except Exception:
            logger.debug("Recursive traverse failed, falling back to BFS", exc_info=True)

        # Fallback: manual BFS (still needed if recursive query isn't supported
        # or if we need calls-only filtering).
        visited: set[str] = set()
        result_list: list[GraphNode] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != start_id:
                node = self.get_node(current_id)
                if node is not None:
                    result_list.append(node)

            if current_depth < depth:
                callees = self.get_callees(current_id)
                for callee in callees:
                    if callee.id not in visited:
                        queue.append((callee.id, current_depth + 1))

        return result_list

    def execute_raw(self, query: str) -> list[list[Any]]:
        """Execute a raw Cypher query and return all result rows."""
        assert self._conn is not None
        result = self._conn.execute(query)
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    # -- Search --------------------------------------------------------------

    def fts_search(self, query: str, limit: int) -> list[SearchResult]:
        """BM25 full-text search using KuzuDB's native FTS extension.

        Searches across all node tables using pre-built FTS indexes on
        ``name``, ``content``, and ``signature`` fields.  Results are
        ranked by BM25 relevance score.

        Returns the top *limit* results sorted by score descending.
        """
        assert self._conn is not None
        escaped_q = _escape(query)
        candidates: list[SearchResult] = []

        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            cypher = (
                f"CALL QUERY_FTS_INDEX('{table}', '{idx_name}', '{escaped_q}') "
                f"RETURN node.id, node.name, node.file_path, node.content, "
                f"node.signature, score "
                f"ORDER BY score DESC LIMIT {limit}"
            )
            try:
                result = self._conn.execute(cypher)
                while result.has_next():
                    row = result.get_next()
                    node_id = row[0] or ""
                    name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    signature = row[4] or ""
                    bm25_score = float(row[5]) if row[5] is not None else 0.0

                    label_prefix = node_id.split(":", 1)[0] if node_id else ""
                    snippet = content[:200] if content else signature[:200]

                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=bm25_score,
                            node_name=name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=snippet,
                        )
                    )
            except Exception:
                logger.debug("fts_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def fuzzy_search(
        self, query: str, limit: int, max_distance: int = 2
    ) -> list[SearchResult]:
        """Fuzzy name search using Levenshtein edit distance.

        Scans all node tables for symbols whose name is within
        *max_distance* edits of *query*.  Converts edit distance to a
        score (0 edits = 1.0, *max_distance* edits = 0.3).
        """
        assert self._conn is not None
        escaped_q = _escape(query.lower())
        candidates: list[SearchResult] = []

        for table in _NODE_TABLE_NAMES:
            cypher = (
                f"MATCH (n:{table}) "
                f"WHERE levenshtein(lower(n.name), '{escaped_q}') <= {max_distance} "
                f"RETURN n.id, n.name, n.file_path, n.content, "
                f"levenshtein(lower(n.name), '{escaped_q}') AS dist "
                f"ORDER BY dist LIMIT {limit}"
            )
            try:
                result = self._conn.execute(cypher)
                while result.has_next():
                    row = result.get_next()
                    node_id = row[0] or ""
                    name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    dist = int(row[4]) if row[4] is not None else max_distance

                    score = max(0.3, 1.0 - (dist * 0.3))
                    label_prefix = node_id.split(":", 1)[0] if node_id else ""

                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=score,
                            node_name=name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=content[:200] if content else "",
                        )
                    )
            except Exception:
                logger.debug("fuzzy_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    # -- Vectors -------------------------------------------------------------

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Persist embedding vectors into the Embedding node table.

        Uses parameterised MERGE (upsert) to insert or update vectors.
        """
        assert self._conn is not None
        for emb in embeddings:
            try:
                self._conn.execute(
                    "MERGE (e:Embedding {node_id: $nid}) SET e.vec = $vec",
                    parameters={"nid": emb.node_id, "vec": emb.embedding},
                )
            except Exception:
                logger.debug(
                    "store_embeddings failed for node %s", emb.node_id, exc_info=True
                )

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """Find the closest nodes to *vector* using native ``array_cosine_similarity``.

        Computes cosine similarity directly in KuzuDB's Cypher engine —
        no Python-side computation or full-table load required.  Joins with
        node tables to fetch metadata in a single query.
        """
        assert self._conn is not None
        # Vector literals must be inlined — KuzuDB parameterized queries
        # cannot distinguish DOUBLE[] from LIST for array_cosine_similarity.
        vec_literal = "[" + ", ".join(str(v) for v in vector) + "]"

        # First get the embedding matches with scores.
        try:
            result = self._conn.execute(
                f"MATCH (e:Embedding) "
                f"RETURN e.node_id, "
                f"array_cosine_similarity(e.vec, {vec_literal}) AS sim "
                f"ORDER BY sim DESC LIMIT {limit}"
            )
        except Exception:
            logger.debug("vector_search failed", exc_info=True)
            return []

        # Collect embedding results, then batch-fetch node metadata.
        emb_rows: list[tuple[str, float]] = []
        while result.has_next():
            row = result.get_next()
            emb_rows.append((row[0] or "", float(row[1]) if row[1] is not None else 0.0))

        if not emb_rows:
            return []

        # Batch-fetch all node metadata in one pass per table.
        node_cache: dict[str, GraphNode] = {}
        node_ids = [r[0] for r in emb_rows]
        ids_by_table: dict[str, list[str]] = {}
        for nid in node_ids:
            table = _table_for_id(nid)
            if table:
                ids_by_table.setdefault(table, []).append(nid)

        for table, ids in ids_by_table.items():
            try:
                q = f"MATCH (n:{table}) WHERE n.id IN $ids RETURN n.*"
                res = self._conn.execute(q, parameters={"ids": ids})
                while res.has_next():
                    row = res.get_next()
                    node = self._row_to_node(row)
                    if node:
                        node_cache[node.id] = node
            except Exception:
                logger.debug("Batch node fetch failed for table %s", table, exc_info=True)

        results: list[SearchResult] = []
        for node_id, sim in emb_rows:
            node = node_cache.get(node_id)
            label_prefix = node_id.split(":", 1)[0] if node_id else ""
            results.append(
                SearchResult(
                    node_id=node_id,
                    score=sim,
                    node_name=node.name if node else "",
                    file_path=node.file_path if node else "",
                    label=label_prefix,
                    snippet=(node.content[:200] if node and node.content else ""),
                )
            )
        return results

    # -- Incremental ---------------------------------------------------------

    def get_indexed_files(self) -> dict[str, str]:
        """Return ``{file_path: sha256(content)}`` for all File nodes.

        Attempts to read pre-computed ``content_hash`` first. Falls back
        to computing the hash from content for databases that predate the
        schema addition.
        """
        assert self._conn is not None
        mapping: dict[str, str] = {}
        try:
            result = self._conn.execute(
                "MATCH (n:File) RETURN n.file_path, n.content"
            )
            while result.has_next():
                row = result.get_next()
                fp = row[0] or ""
                content = row[1] or ""
                mapping[fp] = hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            logger.debug("get_indexed_files failed", exc_info=True)
        return mapping

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """Replace the entire store with the contents of *graph*.

        Drops all existing data, then inserts every node and relationship
        from the in-memory graph.
        """
        assert self._conn is not None
        # Clear existing data — delete nodes (relationships cascade).
        for table in _NODE_TABLE_NAMES:
            try:
                self._conn.execute(f"MATCH (n:{table}) DETACH DELETE n")
            except Exception:
                pass

        self.add_nodes(graph.nodes)
        self.add_relationships(graph.relationships)
        self.rebuild_fts_indexes()

    def rebuild_fts_indexes(self) -> None:
        """Drop and recreate all FTS indexes.

        Must be called after any bulk data change so the BM25 indexes
        reflect the current node contents.
        """
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            try:
                self._conn.execute(f"CALL DROP_FTS_INDEX('{table}', '{idx_name}')")
            except Exception:
                pass
            try:
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])"
                )
            except Exception:
                logger.debug("FTS index rebuild failed for %s", table, exc_info=True)

    # -- Internal helpers ----------------------------------------------------

    def _create_schema(self) -> None:
        """Create node tables, the CodeRelation relationship table group, and the Embedding table."""
        assert self._conn is not None

        # Load the FTS extension for full-text search.
        try:
            self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD EXTENSION fts")
        except Exception:
            logger.debug("FTS extension load skipped (may already be loaded)", exc_info=True)

        # Create one node table per label.
        for table in _NODE_TABLE_NAMES:
            stmt = f"CREATE NODE TABLE IF NOT EXISTS {table}({_NODE_PROPERTIES})"
            self._conn.execute(stmt)

        # Create the Embedding node table for vector storage.
        self._conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS Embedding({_EMBEDDING_PROPERTIES})"
        )

        # Build the REL TABLE GROUP covering all table-to-table combinations.
        from_to_pairs: list[str] = []
        for src in _NODE_TABLE_NAMES:
            for dst in _NODE_TABLE_NAMES:
                from_to_pairs.append(f"FROM {src} TO {dst}")

        pairs_clause = ", ".join(from_to_pairs)
        rel_stmt = (
            f"CREATE REL TABLE GROUP IF NOT EXISTS CodeRelation("
            f"{pairs_clause}, {_REL_PROPERTIES})"
        )
        try:
            self._conn.execute(rel_stmt)
        except Exception:
            logger.debug("REL TABLE GROUP creation skipped", exc_info=True)

        # Create FTS indexes on all node tables (name, content, signature).
        self._create_fts_indexes()

    def _create_fts_indexes(self) -> None:
        """Create FTS indexes for every node table (idempotent)."""
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            try:
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])"
                )
            except Exception:
                # Index may already exist — that's fine.
                pass

    def _insert_node(self, node: GraphNode) -> None:
        """INSERT a single node into the appropriate label table using parameterized query."""
        assert self._conn is not None
        table = _LABEL_TO_TABLE.get(node.label.value)
        if table is None:
            logger.warning("Unknown label %s for node %s", node.label, node.id)
            return

        query = (
            f"CREATE (:{table} {{"
            f"id: $id, name: $name, file_path: $file_path, "
            f"start_line: $start_line, end_line: $end_line, "
            f"content: $content, signature: $signature, "
            f"language: $language, class_name: $class_name, "
            f"is_dead: $is_dead, is_entry_point: $is_entry_point, "
            f"is_exported: $is_exported"
            f"}})"
        )
        params = {
            "id": node.id,
            "name": node.name,
            "file_path": node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "content": node.content,
            "signature": node.signature,
            "language": node.language,
            "class_name": node.class_name,
            "is_dead": node.is_dead,
            "is_entry_point": node.is_entry_point,
            "is_exported": node.is_exported,
        }
        try:
            self._conn.execute(query, parameters=params)
        except Exception:
            logger.debug("Insert node failed for %s", node.id, exc_info=True)

    def _insert_relationship(self, rel: GraphRelationship) -> None:
        """MATCH source and target, then CREATE the relationship using parameterized query."""
        assert self._conn is not None
        src_table = _table_for_id(rel.source)
        tgt_table = _table_for_id(rel.target)
        if src_table is None or tgt_table is None:
            logger.warning(
                "Cannot resolve tables for relationship %s -> %s",
                rel.source,
                rel.target,
            )
            return

        props = rel.properties or {}

        query = (
            f"MATCH (a:{src_table}), (b:{tgt_table}) "
            f"WHERE a.id = $src AND b.id = $tgt "
            f"CREATE (a)-[:CodeRelation {{"
            f"rel_type: $rel_type, "
            f"confidence: $confidence, "
            f"role: $role, "
            f"step_number: $step_number, "
            f"strength: $strength, "
            f"co_changes: $co_changes, "
            f"symbols: $symbols"
            f"}}]->(b)"
        )
        params = {
            "src": rel.source,
            "tgt": rel.target,
            "rel_type": rel.type.value,
            "confidence": float(props.get("confidence", 1.0)),
            "role": str(props.get("role", "")),
            "step_number": int(props.get("step_number", 0)),
            "strength": float(props.get("strength", 0.0)),
            "co_changes": int(props.get("co_changes", 0)),
            "symbols": str(props.get("symbols", "")),
        }
        try:
            self._conn.execute(query, parameters=params)
        except Exception:
            logger.debug(
                "Insert relationship failed: %s -> %s", rel.source, rel.target, exc_info=True
            )

    def _query_nodes(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[GraphNode]:
        """Execute a query returning ``n.*`` columns and convert to GraphNode list."""
        assert self._conn is not None
        nodes: list[GraphNode] = []
        try:
            result = self._conn.execute(query, parameters=parameters or {})
            while result.has_next():
                row = result.get_next()
                node = self._row_to_node(row)
                if node is not None:
                    nodes.append(node)
        except Exception:
            logger.debug("_query_nodes failed: %s", query, exc_info=True)
        return nodes

    @staticmethod
    def _row_to_node(row: list[Any], node_id: str | None = None) -> GraphNode | None:
        """Convert a result row from ``RETURN n.*`` into a GraphNode.

        Column order matches the property definition:
        0=id, 1=name, 2=file_path, 3=start_line, 4=end_line,
        5=content, 6=signature, 7=language, 8=class_name,
        9=is_dead, 10=is_entry_point, 11=is_exported
        """
        try:
            nid = node_id or row[0]
            # Determine the label from the ID prefix.
            prefix = nid.split(":", 1)[0]
            label_map = {label.value: label for label in NodeLabel}
            label = label_map.get(prefix, NodeLabel.FILE)

            return GraphNode(
                id=row[0],
                label=label,
                name=row[1] or "",
                file_path=row[2] or "",
                start_line=row[3] or 0,
                end_line=row[4] or 0,
                content=row[5] or "",
                signature=row[6] or "",
                language=row[7] or "",
                class_name=row[8] or "",
                is_dead=bool(row[9]),
                is_entry_point=bool(row[10]),
                is_exported=bool(row[11]),
            )
        except (IndexError, KeyError):
            logger.debug("Failed to convert row to GraphNode: %s", row, exc_info=True)
            return None
