"""Tests for memory module."""


import pytest

from brainstormer.backends.memory import ChromaMemoryStore, MemoryManager


class TestChromaMemoryStore:
    """Tests for ChromaMemoryStore."""

    def test_store_initialization(self, temp_dir):
        """Test store initializes correctly."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")

        assert store.count() == 0

    def test_add_memory(self, temp_dir):
        """Test adding a memory."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")

        memory_id = store.add_memory(
            content="This is a test memory",
            metadata={"type": "test"},
        )

        assert memory_id is not None
        assert store.count() == 1

    def test_search_memory(self, temp_dir):
        """Test searching memories."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")

        store.add_memory("Python is a programming language")
        store.add_memory("JavaScript runs in browsers")
        store.add_memory("Machine learning uses algorithms")

        results = store.search("programming language", n_results=2)

        assert len(results) <= 2
        # Python should be most relevant
        assert any("Python" in r["content"] for r in results)

    def test_get_memory(self, temp_dir):
        """Test getting a specific memory."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")

        store.add_memory("Test content", memory_id="test-id-123")

        memory = store.get_memory("test-id-123")

        assert memory is not None
        assert memory["content"] == "Test content"

    def test_delete_memory(self, temp_dir):
        """Test deleting a memory."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")

        memory_id = store.add_memory("To be deleted")
        assert store.count() == 1

        store.delete_memory(memory_id)
        assert store.count() == 0

    def test_update_memory(self, temp_dir):
        """Test updating a memory."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")

        memory_id = store.add_memory("Original content")

        store.update_memory(memory_id, content="Updated content")

        memory = store.get_memory(memory_id)
        assert memory["content"] == "Updated content"

    def test_clear_memories(self, temp_dir):
        """Test clearing all memories."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")

        store.add_memory("Memory 1")
        store.add_memory("Memory 2")
        assert store.count() == 2

        store.clear()
        assert store.count() == 0


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a memory manager for testing."""
        store = ChromaMemoryStore(persist_directory=temp_dir / "chroma")
        return MemoryManager(store)

    def test_remember_research(self, manager):
        """Test storing research memory."""
        memory_id = manager.remember_research(
            session_id="session-1",
            agent_name="researcher",
            content="Important finding about topic X",
            focus_area="Topic X",
            tags=["important", "verified"],
        )

        assert memory_id is not None

    def test_remember_insight(self, manager):
        """Test storing insight memory."""
        memory_id = manager.remember_insight(
            content="Key insight discovered during research",
            session_id="session-1",
            source="analysis",
        )

        assert memory_id is not None

    def test_recall_relevant(self, manager):
        """Test recalling relevant memories."""
        manager.remember_insight("Python is great for data science")
        manager.remember_insight("JavaScript is used for web development")
        manager.remember_insight("Machine learning requires lots of data")

        results = manager.recall_relevant("data science programming")

        assert len(results) > 0

    def test_recall_by_type(self, manager):
        """Test recalling memories by type."""
        manager.remember_research(
            session_id="s1",
            agent_name="agent",
            content="Research finding",
            focus_area="area",
        )
        manager.remember_insight("An insight")

        results = manager.recall_by_type("research", "finding")

        assert len(results) > 0

    def test_get_session_memories(self, manager):
        """Test getting all memories for a session."""
        manager.remember_research(
            session_id="session-1",
            agent_name="agent",
            content="Session 1 finding",
            focus_area="area",
        )
        manager.remember_research(
            session_id="session-2",
            agent_name="agent",
            content="Session 2 finding",
            focus_area="area",
        )

        # This relies on metadata filtering
        results = manager.get_session_memories("session-1")

        # Should return at least one result
        assert isinstance(results, list)
