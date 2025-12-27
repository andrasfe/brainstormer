"""Custom tools for the research agents."""

import os
from collections.abc import Callable
from typing import Any, Literal

from tavily import TavilyClient

from ..backends.memory import MemoryManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_search_tool(api_key: str | None = None) -> Callable[..., dict[str, Any]]:
    """Create a Tavily web search tool."""
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY is required for web search")

    client = TavilyClient(api_key=key)

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
        search_depth: Literal["basic", "advanced"] = "basic",
    ) -> dict[str, Any]:
        """
        Search the internet for information.

        Args:
            query: The search query
            max_results: Maximum number of results to return (1-10)
            topic: The topic category for the search
            include_raw_content: Whether to include raw page content
            search_depth: Search depth - 'basic' for quick, 'advanced' for thorough

        Returns:
            Search results with titles, URLs, and content snippets
        """
        try:
            results: dict[str, Any] = client.search(
                query=query,
                max_results=min(max_results, 10),
                topic=topic,
                include_raw_content=include_raw_content,
                search_depth=search_depth,
            )
            logger.debug(f"Search completed: {query} ({len(results.get('results', []))} results)")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e), "results": []}

    return internet_search


def create_memory_tools(memory_manager: MemoryManager) -> dict[str, Callable[..., Any]]:
    """Create tools for interacting with long-term memory."""

    def remember(
        content: str,
        memory_type: Literal["research", "insight", "note"] = "note",
        tags: list[str] | None = None,
    ) -> str:
        """
        Store information in long-term memory for future recall.

        Args:
            content: The information to remember
            memory_type: Type of memory (research, insight, or note)
            tags: Optional tags for categorization

        Returns:
            Memory ID for the stored content
        """
        if memory_type == "research":
            memory_id = memory_manager.remember_research(
                session_id="",  # Will be set by context
                agent_name="user",
                content=content,
                focus_area="general",
                tags=tags,
            )
        else:
            memory_id = memory_manager.remember_insight(
                content=content,
                source=memory_type,
            )
        logger.debug(f"Stored memory: {memory_id}")
        return f"Stored with ID: {memory_id}"

    def recall(
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Recall relevant memories based on a query.

        Args:
            query: What to search for in memory
            n_results: Maximum number of memories to return

        Returns:
            List of relevant memories with content and metadata
        """
        memories = memory_manager.recall_relevant(query, n_results=n_results)
        logger.debug(f"Recalled {len(memories)} memories for: {query}")
        return memories

    return {"remember": remember, "recall": recall}


def create_file_context_tool(input_files: list[dict[str, Any]]) -> Callable[..., str | list[dict[str, Any]]]:
    """Create a tool for accessing input file context."""

    def get_input_context(file_name: str | None = None) -> str | list[dict[str, Any]]:
        """
        Get context from the input files provided for this research session.

        Args:
            file_name: Specific file to retrieve, or None for all files

        Returns:
            Content of the specified file or list of all input files
        """
        if file_name:
            for f in input_files:
                if f["name"] == file_name or file_name in f["path"]:
                    return str(f["content"])
            return f"File not found: {file_name}"

        # Return summary of all files
        return [
            {"name": f["name"], "type": f["type"], "size": f["size"]}
            for f in input_files
        ]

    return get_input_context
