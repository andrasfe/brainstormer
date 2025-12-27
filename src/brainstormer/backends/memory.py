"""Long-term memory using ChromaDB for vector storage."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ChromaMemoryStore:
    """ChromaDB-based vector memory store for semantic search."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "brainstormer_memory",
        embedding_function: Any = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"Initialized ChromaDB memory store at {persist_directory}")

    def add_memory(
        self,
        content: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> str:
        """Add a memory to the store."""
        if memory_id is None:
            memory_id = hashlib.sha256(
                f"{content}{datetime.now(tz=UTC).isoformat()}".encode()
            ).hexdigest()[:16]

        full_metadata = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "content_length": len(content),
            **(metadata or {}),
        }

        # Convert any non-string metadata values to strings for ChromaDB
        full_metadata = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in full_metadata.items()
        }

        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[full_metadata],
        )

        logger.debug(f"Added memory: {memory_id}")
        return memory_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Search memories by semantic similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                memories.append({
                    "id": memory_id,
                    "content": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return memories

    def get_memory(self, memory_id: str) -> dict | None:
        """Get a specific memory by ID."""
        results = self.collection.get(ids=[memory_id])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else None,
            }
        return None

    def delete_memory(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self.collection.delete(ids=[memory_id])
        logger.debug(f"Deleted memory: {memory_id}")

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Update an existing memory."""
        if content and metadata:
            clean_metadata = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in metadata.items()
            }
            self.collection.update(
                ids=[memory_id],
                documents=[content],
                metadatas=[clean_metadata],
            )
        elif content:
            self.collection.update(ids=[memory_id], documents=[content])
        elif metadata:
            clean_metadata = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in metadata.items()
            }
            self.collection.update(ids=[memory_id], metadatas=[clean_metadata])

    def count(self) -> int:
        """Get the total number of memories."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all memories from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all memories")


class MemoryManager:
    """High-level memory manager integrating with the agent system."""

    def __init__(self, chroma_store: ChromaMemoryStore):
        self.store = chroma_store

    def remember_research(
        self,
        session_id: str,
        agent_name: str,
        content: str,
        focus_area: str,
        tags: list[str] | None = None,
    ) -> str:
        """Store a research finding in long-term memory."""
        return self.store.add_memory(
            content=content,
            metadata={
                "type": "research",
                "session_id": session_id,
                "agent_name": agent_name,
                "focus_area": focus_area,
                "tags": tags or [],
            },
        )

    def remember_insight(
        self,
        content: str,
        session_id: str | None = None,
        source: str | None = None,
    ) -> str:
        """Store a general insight or learning."""
        return self.store.add_memory(
            content=content,
            metadata={
                "type": "insight",
                "session_id": session_id,
                "source": source,
            },
        )

    def recall_relevant(
        self,
        query: str,
        n_results: int = 5,
        session_id: str | None = None,
    ) -> list[dict]:
        """Recall relevant memories for a query."""
        where = None
        if session_id:
            where = {"session_id": session_id}
        return self.store.search(query, n_results=n_results, where=where)

    def recall_by_type(
        self,
        memory_type: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Recall memories of a specific type."""
        return self.store.search(
            query,
            n_results=n_results,
            where={"type": memory_type},
        )

    def get_session_memories(self, session_id: str, limit: int = 50) -> list[dict]:
        """Get all memories for a session."""
        # Use a broad query to get session memories
        return self.store.search(
            query="research findings insights",
            n_results=limit,
            where={"session_id": session_id},
        )
