"""MCP tool handler implementations for Axon.

Each function accepts a storage backend and the tool-specific arguments,
performs the appropriate query, and returns a human-readable string suitable
for inclusion in an MCP ``TextContent`` response.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axon.core.storage.base import StorageBackend


# ---------------------------------------------------------------------------
# 1. axon_list_repos
# ---------------------------------------------------------------------------


def handle_list_repos(registry_dir: Path | None = None) -> str:
    """List indexed repositories by scanning for .axon directories.

    Scans the global registry directory (defaults to ``~/.axon/repos``) for
    project metadata files and returns a formatted summary.

    Args:
        registry_dir: Directory containing repo metadata. If ``None``,
            defaults to ``~/.axon/repos``.

    Returns:
        Formatted list of indexed repositories with stats, or a message
        indicating none were found.
    """
    use_cwd_fallback = registry_dir is None
    if registry_dir is None:
        registry_dir = Path.home() / ".axon" / "repos"

    repos: list[dict[str, Any]] = []

    if registry_dir.exists():
        for meta_file in registry_dir.glob("*/meta.json"):
            try:
                data = json.loads(meta_file.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    if not repos and use_cwd_fallback:
        # Fall back: scan current directory for .axon
        cwd_axon = Path.cwd() / ".axon" / "meta.json"
        if cwd_axon.exists():
            try:
                data = json.loads(cwd_axon.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                pass

    if not repos:
        return "No indexed repositories found. Run `axon index` on a project first."

    lines = [f"Indexed repositories ({len(repos)}):"]
    lines.append("")
    for i, repo in enumerate(repos, 1):
        name = repo.get("name", "unknown")
        path = repo.get("path", "")
        nodes = repo.get("node_count", "?")
        edges = repo.get("edge_count", "?")
        files = repo.get("file_count", "?")
        lines.append(f"  {i}. {name}")
        lines.append(f"     Path: {path}")
        lines.append(f"     Nodes: {nodes}  Edges: {edges}  Files: {files}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. axon_query — Hybrid search
# ---------------------------------------------------------------------------


def handle_query(storage: StorageBackend, query: str, limit: int = 20) -> str:
    """Execute hybrid search and format results.

    Args:
        storage: The storage backend to search against.
        query: Text search query.
        limit: Maximum number of results (default 20).

    Returns:
        Formatted search results with file, name, label, and snippet.
    """
    from axon.core.search.hybrid import hybrid_search

    results = hybrid_search(query, storage, limit=limit)
    if not results:
        return f"No results found for '{query}'."

    lines = []
    for i, r in enumerate(results, 1):
        label = r.label.title() if r.label else "Unknown"
        lines.append(f"{i}. {r.node_name} ({label}) -- {r.file_path}")
        if r.snippet:
            snippet = r.snippet[:200].replace("\n", " ").strip()
            lines.append(f"   {snippet}")
    lines.append("")
    lines.append("Next: Use context() on a specific symbol for the full picture.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. axon_context — 360-degree symbol view
# ---------------------------------------------------------------------------


def handle_context(storage: StorageBackend, symbol: str) -> str:
    """Provide a 360-degree view of a symbol.

    Looks up the symbol by name via full-text search, then retrieves its
    callers, callees, and type references.

    Args:
        storage: The storage backend.
        symbol: The symbol name to look up.

    Returns:
        Formatted view including callers, callees, type refs, and guidance.
    """
    results = storage.fts_search(symbol, limit=1)
    if not results:
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        return f"Symbol '{symbol}' not found."

    label_display = node.label.value.title() if node.label else "Unknown"
    lines = [f"Symbol: {node.name} ({label_display})"]
    lines.append(f"File: {node.file_path}:{node.start_line}-{node.end_line}")

    if node.signature:
        lines.append(f"Signature: {node.signature}")

    if node.is_dead:
        lines.append("Status: DEAD CODE (unreachable)")

    # Callers
    callers = storage.get_callers(node.id)
    if callers:
        lines.append(f"\nCallers ({len(callers)}):")
        for c in callers:
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}")

    # Callees
    callees = storage.get_callees(node.id)
    if callees:
        lines.append(f"\nCallees ({len(callees)}):")
        for c in callees:
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}")

    # Type references
    type_refs = storage.get_type_refs(node.id)
    if type_refs:
        lines.append(f"\nType references ({len(type_refs)}):")
        for t in type_refs:
            lines.append(f"  -> {t.name}  {t.file_path}")

    lines.append("")
    lines.append("Next: Use impact() if planning changes to this symbol.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. axon_impact — Blast radius analysis
# ---------------------------------------------------------------------------


def handle_impact(storage: StorageBackend, symbol: str, depth: int = 3) -> str:
    """Analyse the blast radius of changing a symbol.

    Uses BFS traversal through CALLS edges to find all affected symbols
    up to the specified depth.

    Args:
        storage: The storage backend.
        symbol: The symbol name to analyse.
        depth: Maximum traversal depth (default 3).

    Returns:
        Formatted impact analysis showing affected symbols at each depth level.
    """
    results = storage.fts_search(symbol, limit=1)
    if not results:
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        return f"Symbol '{symbol}' not found."

    affected = storage.traverse(start_node.id, depth)
    if not affected:
        return f"No downstream dependencies found for '{symbol}'."

    lines = [f"Impact analysis for: {start_node.name} ({start_node.label.value.title()})"]
    lines.append(f"Depth: {depth}")
    lines.append(f"Total affected symbols: {len(affected)}")
    lines.append("")

    for i, node in enumerate(affected, 1):
        label = node.label.value.title() if node.label else "Unknown"
        lines.append(f"  {i}. {node.name} ({label}) -- {node.file_path}:{node.start_line}")

    lines.append("")
    lines.append("Tip: Review each affected symbol before making changes.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. axon_dead_code — List unreachable code
# ---------------------------------------------------------------------------


def handle_dead_code(storage: StorageBackend) -> str:
    """List all symbols marked as dead code.

    Queries the storage backend for nodes where ``is_dead`` is ``True``.

    Args:
        storage: The storage backend.

    Returns:
        Formatted list of dead code symbols with file and label.
    """
    try:
        rows = storage.execute_raw(
            "MATCH (n) WHERE n.is_dead = true RETURN n.id, n.name, n.file_path"
        )
    except Exception:
        rows = []

    if not rows:
        return "No dead code detected."

    lines = [f"Dead code ({len(rows)} symbols):"]
    lines.append("")
    for i, row in enumerate(rows, 1):
        node_id = row[0] if row else ""
        name = row[1] if len(row) > 1 else "?"
        file_path = row[2] if len(row) > 2 else "?"
        label = node_id.split(":", 1)[0].title() if node_id else "Unknown"
        lines.append(f"  {i}. {name} ({label}) -- {file_path}")

    lines.append("")
    lines.append("Tip: Consider removing or refactoring these symbols.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. axon_detect_changes — Git diff -> affected symbols
# ---------------------------------------------------------------------------


_DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_DIFF_HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def handle_detect_changes(storage: StorageBackend, diff: str) -> str:
    """Map git diff output to affected symbols.

    Parses the diff to find changed files and line ranges, then queries
    the storage backend to identify which symbols those lines belong to.

    Args:
        storage: The storage backend.
        diff: Raw git diff output string.

    Returns:
        Formatted list of affected symbols per changed file.
    """
    if not diff.strip():
        return "Empty diff provided."

    # Parse changed files and their modified line ranges
    changed_files: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split("\n"):
        file_match = _DIFF_FILE_PATTERN.match(line)
        if file_match:
            current_file = file_match.group(2)
            if current_file not in changed_files:
                changed_files[current_file] = []
            continue

        hunk_match = _DIFF_HUNK_PATTERN.match(line)
        if hunk_match and current_file is not None:
            start = int(hunk_match.group(1))
            count = int(hunk_match.group(2) or "1")
            changed_files[current_file].append((start, start + count - 1))

    if not changed_files:
        return "Could not parse any changed files from the diff."

    lines = [f"Changed files: {len(changed_files)}"]
    lines.append("")
    total_affected = 0

    for file_path, ranges in changed_files.items():
        # Find symbols in this file by direct Cypher query on file_path.
        affected_symbols = []
        try:
            rows = storage.execute_raw(
                f"MATCH (n) WHERE n.file_path = '{file_path.replace(chr(39), '')}' "
                f"AND n.start_line > 0 "
                f"RETURN n.id, n.name, n.file_path, n.start_line, n.end_line"
            )
            for row in rows or []:
                node_id = row[0] or ""
                name = row[1] or ""
                start_line = row[3] or 0
                end_line = row[4] or 0
                label_prefix = node_id.split(":", 1)[0] if node_id else ""
                # Check if any changed range overlaps with the symbol's line range
                for start, end in ranges:
                    if start_line <= end and end_line >= start:
                        # Create a lightweight proxy object for display.
                        from types import SimpleNamespace
                        sym = SimpleNamespace(
                            name=name, label=SimpleNamespace(value=label_prefix),
                            start_line=start_line, end_line=end_line,
                        )
                        affected_symbols.append(sym)
                        break
        except Exception:
            pass

        lines.append(f"  {file_path}:")
        if affected_symbols:
            for sym in affected_symbols:
                label = sym.label.value.title() if sym.label else "Unknown"
                lines.append(
                    f"    - {sym.name} ({label}) lines {sym.start_line}-{sym.end_line}"
                )
                total_affected += 1
        else:
            lines.append("    (no indexed symbols in changed lines)")
        lines.append("")

    lines.append(f"Total affected symbols: {total_affected}")
    lines.append("")
    lines.append("Next: Use impact() on affected symbols to see downstream effects.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. axon_cypher — Raw Cypher query
# ---------------------------------------------------------------------------


# Keywords that indicate a write/destructive Cypher query.
_WRITE_KEYWORDS = re.compile(
    r"\b(DELETE|DROP|CREATE|SET|REMOVE|MERGE|DETACH|INSTALL|LOAD|COPY|CALL)\b",
    re.IGNORECASE,
)


def handle_cypher(storage: StorageBackend, query: str) -> str:
    """Execute a raw Cypher query and return formatted results.

    Only read-only queries are allowed.  Queries containing write keywords
    (DELETE, DROP, CREATE, SET, etc.) are rejected.

    Args:
        storage: The storage backend.
        query: The Cypher query string.

    Returns:
        Formatted query results, or an error message if execution fails.
    """
    if _WRITE_KEYWORDS.search(query):
        return (
            "Query rejected: only read-only queries (MATCH/RETURN) are allowed. "
            "Write operations (DELETE, DROP, CREATE, SET, MERGE) are not permitted."
        )

    try:
        rows = storage.execute_raw(query)
    except Exception as exc:
        return f"Cypher query failed: {exc}"

    if not rows:
        return "Query returned no results."

    lines = [f"Results ({len(rows)} rows):"]
    lines.append("")
    for i, row in enumerate(rows, 1):
        formatted_values = [str(v) for v in row]
        lines.append(f"  {i}. {' | '.join(formatted_values)}")

    return "\n".join(lines)
