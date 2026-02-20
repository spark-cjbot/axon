"""MCP server for Axon — exposes code intelligence tools over stdio transport.

Registers seven tools and three resources that give AI agents and MCP clients
access to the Axon knowledge graph.  The server lazily initialises a
:class:`KuzuBackend` from the ``.axon/kuzu`` directory in the current
working directory.

Usage::

    # Run directly
    python -m axon.mcp.server

    # Or via the MCP protocol (stdio transport)
    axon serve  # (once CLI is wired)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.resources import get_dead_code_list, get_overview, get_schema
from axon.mcp.tools import (
    handle_context,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_impact,
    handle_list_repos,
    handle_query,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = Server("axon")


class _StorageHolder:
    """Encapsulates lazy-initialised storage to avoid module-level globals."""

    def __init__(self) -> None:
        self._storage: KuzuBackend | None = None

    def get(self) -> KuzuBackend:
        """Lazily initialise the KuzuDB storage backend.

        Looks for a ``.axon/kuzu`` directory in the current working directory.
        If it exists, the backend is initialised from that path.  Otherwise a
        bare (uninitialised) backend is returned so that tools can still be
        called without crashing.
        """
        if self._storage is None:
            self._storage = KuzuBackend()
            db_path = Path.cwd() / ".axon" / "kuzu"
            if db_path.exists():
                self._storage.initialize(db_path, read_only=True)
                logger.info("Initialised storage (read-only) from %s", db_path)
            else:
                logger.warning("No .axon/kuzu directory found in %s", Path.cwd())
        return self._storage

    def close(self) -> None:
        """Close the storage backend if open."""
        if self._storage is not None:
            self._storage.close()
            self._storage = None


_holder = _StorageHolder()


def _get_storage() -> KuzuBackend:
    """Return the lazily-initialised storage backend."""
    return _holder.get()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="axon_list_repos",
        description="List all indexed repositories with their stats.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_query",
        description=(
            "Search the knowledge graph using hybrid (keyword + vector) search. "
            "Returns ranked symbols matching the query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="axon_context",
        description=(
            "Get a 360-degree view of a symbol: callers, callees, type references, "
            "and community membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to look up.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_impact",
        description=(
            "Blast radius analysis: find all symbols affected by changing a given symbol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to analyse.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth (default 3).",
                    "default": 3,
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_dead_code",
        description="List all symbols detected as dead (unreachable) code.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_detect_changes",
        description=(
            "Parse a git diff and map changed files/lines to affected symbols "
            "in the knowledge graph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
            },
            "required": ["diff"],
        },
    ),
    Tool(
        name="axon_cypher",
        description="Execute a raw Cypher query against the knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query string.",
                },
            },
            "required": ["query"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available Axon tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    storage = _get_storage()

    if name == "axon_list_repos":
        result = handle_list_repos()

    elif name == "axon_query":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 20)
        result = handle_query(storage, query, limit=limit)

    elif name == "axon_context":
        symbol = arguments.get("symbol", "")
        result = handle_context(storage, symbol)

    elif name == "axon_impact":
        symbol = arguments.get("symbol", "")
        depth = arguments.get("depth", 3)
        result = handle_impact(storage, symbol, depth=depth)

    elif name == "axon_dead_code":
        result = handle_dead_code(storage)

    elif name == "axon_detect_changes":
        diff = arguments.get("diff", "")
        result = handle_detect_changes(storage, diff)

    elif name == "axon_cypher":
        query = arguments.get("query", "")
        result = handle_cypher(storage, query)

    else:
        result = f"Unknown tool: {name}"

    return [TextContent(type="text", text=result)]


# ---------------------------------------------------------------------------
# Resource registration
# ---------------------------------------------------------------------------


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return the list of available Axon resources."""
    return [
        Resource(
            uri="axon://overview",
            name="Codebase Overview",
            description="High-level statistics about the indexed codebase.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://dead-code",
            name="Dead Code Report",
            description="List of all symbols flagged as unreachable.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://schema",
            name="Graph Schema",
            description="Description of the Axon knowledge graph schema.",
            mimeType="text/plain",
        ),
    ]


@server.read_resource()
async def read_resource(uri) -> str:
    """Read the contents of an Axon resource."""
    uri_str = str(uri)

    if uri_str == "axon://overview":
        storage = _get_storage()
        return get_overview(storage)

    if uri_str == "axon://dead-code":
        storage = _get_storage()
        return get_dead_code_list(storage)

    if uri_str == "axon://schema":
        return get_schema()

    return f"Unknown resource: {uri_str}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the Axon MCP server over stdio transport."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
