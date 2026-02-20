"""Phase 2: Structure processing for Axon.

Takes a list of file entries (path, content, language) and populates the
knowledge graph with File and Folder nodes connected by CONTAINS relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)


@dataclass
class FileInfo:
    """Minimal file info needed by the structure processor."""

    path: str  # relative path
    content: str
    language: str


def process_structure(files: list[FileInfo], graph: KnowledgeGraph) -> None:
    """Build File/Folder nodes and CONTAINS relationships from a list of files.

    For every file entry a :pyclass:`NodeLabel.FILE` node is created.  Every
    unique directory that appears in any file path produces a
    :pyclass:`NodeLabel.FOLDER` node.  Parent-child folder relationships and
    folder-to-file relationships are expressed as :pyclass:`RelType.CONTAINS`
    edges.

    Args:
        files: File entries to process.  Each entry carries the relative path,
            raw content, and detected language.
        graph: The knowledge graph to populate.  Nodes and relationships are
            **added** (existing content is not removed).
    """
    # Collect all unique directory paths across every file.
    folder_paths: set[str] = set()

    for file_info in files:
        pure = PurePosixPath(file_info.path)

        # Gather every ancestor directory (excluding ".").
        for parent in pure.parents:
            parent_str = str(parent)
            if parent_str == ".":
                continue
            folder_paths.add(parent_str)

    # --- Create Folder nodes ---------------------------------------------------
    for dir_path in folder_paths:
        folder_id = generate_id(NodeLabel.FOLDER, dir_path)
        if graph.get_node(folder_id) is None:
            graph.add_node(
                GraphNode(
                    id=folder_id,
                    label=NodeLabel.FOLDER,
                    name=PurePosixPath(dir_path).name,
                    file_path=dir_path,
                )
            )

    # --- Create File nodes -----------------------------------------------------
    for file_info in files:
        file_id = generate_id(NodeLabel.FILE, file_info.path)
        graph.add_node(
            GraphNode(
                id=file_id,
                label=NodeLabel.FILE,
                name=PurePosixPath(file_info.path).name,
                file_path=file_info.path,
                content=file_info.content,
                language=file_info.language,
            )
        )

    # --- Create CONTAINS relationships -----------------------------------------

    # Folder -> Folder (parent contains child)
    for dir_path in folder_paths:
        pure = PurePosixPath(dir_path)
        parent_str = str(pure.parent)
        if parent_str == ".":
            # Top-level folder has no parent — skip.
            continue
        parent_id = generate_id(NodeLabel.FOLDER, parent_str)
        child_id = generate_id(NodeLabel.FOLDER, dir_path)
        rel_id = f"contains:{parent_id}->{child_id}"
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.CONTAINS,
                source=parent_id,
                target=child_id,
            )
        )

    # Folder -> File (immediate parent folder contains file)
    for file_info in files:
        pure = PurePosixPath(file_info.path)
        parent_str = str(pure.parent)
        if parent_str == ".":
            # Root-level file — no containing folder.
            continue
        parent_id = generate_id(NodeLabel.FOLDER, parent_str)
        file_id = generate_id(NodeLabel.FILE, file_info.path)
        rel_id = f"contains:{parent_id}->{file_id}"
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.CONTAINS,
                source=parent_id,
                target=file_id,
            )
        )
