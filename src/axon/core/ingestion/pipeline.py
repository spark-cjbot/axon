"""Pipeline orchestrator for Axon.

Runs all ingestion phases in sequence, populates an in-memory knowledge graph,
bulk-loads it into a storage backend, and returns a summary of the results.

Phases executed:
    0. Incremental diff (reserved -- not yet implemented)
    1. File walking
    2. Structure processing (File/Folder nodes + CONTAINS edges)
    3. Code parsing (symbol nodes + DEFINES edges)
    4. Import resolution (IMPORTS edges)
    5. Call tracing (CALLS edges)
    6. Heritage extraction (EXTENDS / IMPLEMENTS edges)
    7. Type analysis (USES_TYPE edges)
    8. Community detection (COMMUNITY nodes + MEMBER_OF edges)
    9. Process detection (PROCESS nodes + STEP_IN_PROCESS edges)
    10. Dead code detection (flags unreachable symbols)
    11. Change coupling (COUPLED_WITH edges from git history)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from axon.config.ignore import load_gitignore
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.ingestion.calls import process_calls
from axon.core.ingestion.community import process_communities
from axon.core.ingestion.coupling import process_coupling
from axon.core.ingestion.dead_code import process_dead_code
from axon.core.ingestion.heritage import process_heritage
from axon.core.ingestion.imports import process_imports
from axon.core.ingestion.parser_phase import process_parsing
from axon.core.ingestion.processes import process_processes
from axon.core.ingestion.structure import FileInfo, process_structure
from axon.core.ingestion.types import process_types
from axon.core.ingestion.walker import FileEntry, walk_repo
from axon.core.storage.base import StorageBackend


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    files: int = 0
    symbols: int = 0
    relationships: int = 0
    clusters: int = 0
    processes: int = 0
    dead_code: int = 0
    coupled_pairs: int = 0
    duration_seconds: float = 0.0
    incremental: bool = False
    changed_files: int = 0


# Labels that count as "symbols" (everything except structural nodes).
_SYMBOL_LABELS: frozenset[NodeLabel] = frozenset(NodeLabel) - {
    NodeLabel.FILE,
    NodeLabel.FOLDER,
    NodeLabel.COMMUNITY,
    NodeLabel.PROCESS,
}


def run_pipeline(
    repo_path: Path,
    storage: StorageBackend,
    full: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PipelineResult:
    """Run phases 1-11 of the ingestion pipeline and load results into storage.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to analyse.
    storage:
        An already-initialised :class:`StorageBackend` to persist the graph.
    full:
        When ``True``, skip incremental-diff logic (Phase 0) and force a full
        re-index.  Currently Phase 0 is a no-op regardless of this flag.
    progress_callback:
        Optional ``(phase_name, progress)`` callback where *progress* is a
        float in ``[0.0, 1.0]``.

    Returns
    -------
    PipelineResult
        A summary dataclass with counts and timings.
    """
    start = time.monotonic()
    result = PipelineResult()

    def report(phase: str, pct: float) -> None:
        if progress_callback is not None:
            progress_callback(phase, pct)

    # ------------------------------------------------------------------
    # Phase 1: Walk files
    # ------------------------------------------------------------------
    report("Walking files", 0.0)
    gitignore = load_gitignore(repo_path)
    files = walk_repo(repo_path, gitignore)
    result.files = len(files)
    report("Walking files", 1.0)

    # ------------------------------------------------------------------
    # Build in-memory graph
    # ------------------------------------------------------------------
    graph = KnowledgeGraph()

    # ------------------------------------------------------------------
    # Phase 2: Structure (File / Folder nodes + CONTAINS)
    # ------------------------------------------------------------------
    report("Processing structure", 0.0)
    file_infos = [
        FileInfo(path=f.path, content=f.content, language=f.language)
        for f in files
    ]
    process_structure(file_infos, graph)
    report("Processing structure", 1.0)

    # ------------------------------------------------------------------
    # Phase 3: Code parsing (symbol nodes + DEFINES)
    # ------------------------------------------------------------------
    report("Parsing code", 0.0)
    parse_data = process_parsing(files, graph)
    report("Parsing code", 1.0)

    # ------------------------------------------------------------------
    # Phase 4: Import resolution (IMPORTS edges)
    # ------------------------------------------------------------------
    report("Resolving imports", 0.0)
    process_imports(parse_data, graph)
    report("Resolving imports", 1.0)

    # ------------------------------------------------------------------
    # Phase 5: Call tracing (CALLS edges)
    # ------------------------------------------------------------------
    report("Tracing calls", 0.0)
    process_calls(parse_data, graph)
    report("Tracing calls", 1.0)

    # ------------------------------------------------------------------
    # Phase 6: Heritage extraction (EXTENDS / IMPLEMENTS)
    # ------------------------------------------------------------------
    report("Extracting heritage", 0.0)
    process_heritage(parse_data, graph)
    report("Extracting heritage", 1.0)

    # ------------------------------------------------------------------
    # Phase 7: Type analysis (USES_TYPE edges)
    # ------------------------------------------------------------------
    report("Analyzing types", 0.0)
    process_types(parse_data, graph)
    report("Analyzing types", 1.0)

    # ------------------------------------------------------------------
    # Phase 8: Community detection (COMMUNITY nodes + MEMBER_OF)
    # ------------------------------------------------------------------
    report("Detecting communities", 0.0)
    result.clusters = process_communities(graph)
    report("Detecting communities", 1.0)

    # ------------------------------------------------------------------
    # Phase 9: Process detection (PROCESS nodes + STEP_IN_PROCESS)
    # ------------------------------------------------------------------
    report("Detecting execution flows", 0.0)
    result.processes = process_processes(graph)
    report("Detecting execution flows", 1.0)

    # ------------------------------------------------------------------
    # Phase 10: Dead code detection
    # ------------------------------------------------------------------
    report("Finding dead code", 0.0)
    result.dead_code = process_dead_code(graph)
    report("Finding dead code", 1.0)

    # ------------------------------------------------------------------
    # Phase 11: Change coupling (git history analysis)
    # ------------------------------------------------------------------
    report("Analyzing git history", 0.0)
    result.coupled_pairs = process_coupling(graph, repo_path)
    report("Analyzing git history", 1.0)

    # ------------------------------------------------------------------
    # Load into storage
    # ------------------------------------------------------------------
    report("Loading to storage", 0.0)
    storage.bulk_load(graph)
    report("Loading to storage", 1.0)

    # ------------------------------------------------------------------
    # Compute summary stats
    # ------------------------------------------------------------------
    stats = graph.stats()
    result.symbols = len(
        [n for n in graph.nodes if n.label in _SYMBOL_LABELS]
    )
    result.relationships = stats["relationships"]
    result.duration_seconds = time.monotonic() - start

    return result


def reindex_files(
    file_entries: list[FileEntry],
    repo_path: Path,
    storage: StorageBackend,
) -> KnowledgeGraph:
    """Re-index specific files through phases 2-7 (file-local phases).

    Removes old nodes for these files from storage, re-parses them,
    and inserts updated nodes/relationships. Returns the partial graph
    for further processing (global phases, embeddings).

    Parameters
    ----------
    file_entries:
        The files to re-index (already read from disk).
    repo_path:
        Root directory of the repository.
    storage:
        An already-initialised storage backend.

    Returns
    -------
    KnowledgeGraph
        The partial in-memory graph containing only the reindexed files.
    """
    # Remove old data for these files from storage.
    for entry in file_entries:
        storage.remove_nodes_by_file(entry.path)

    # Build a mini graph for just these files.
    graph = KnowledgeGraph()

    file_infos = [
        FileInfo(path=f.path, content=f.content, language=f.language)
        for f in file_entries
    ]
    process_structure(file_infos, graph)
    parse_data = process_parsing(file_entries, graph)
    process_imports(parse_data, graph)
    process_calls(parse_data, graph)
    process_heritage(parse_data, graph)
    process_types(parse_data, graph)

    # Load partial graph into storage.
    storage.add_nodes(graph.nodes)
    storage.add_relationships(graph.relationships)
    storage.rebuild_fts_indexes()

    return graph


def build_graph(repo_path: Path) -> KnowledgeGraph:
    """Run phases 1-11 and return the in-memory graph (no storage load).

    This is used by branch comparison to build a graph snapshot without
    needing a storage backend.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to analyse.

    Returns
    -------
    KnowledgeGraph
        The fully populated in-memory graph.
    """
    gitignore = load_gitignore(repo_path)
    files = walk_repo(repo_path, gitignore)

    graph = KnowledgeGraph()

    file_infos = [
        FileInfo(path=f.path, content=f.content, language=f.language)
        for f in files
    ]
    process_structure(file_infos, graph)
    parse_data = process_parsing(files, graph)
    process_imports(parse_data, graph)
    process_calls(parse_data, graph)
    process_heritage(parse_data, graph)
    process_types(parse_data, graph)
    process_communities(graph)
    process_processes(graph)
    process_dead_code(graph)
    process_coupling(graph, repo_path)

    return graph
