"""Tests for persistence module."""



from brainstormer.backends.persistence import PersistenceManager, SQLiteStore


class TestSQLiteStore:
    """Tests for SQLiteStore."""

    def test_store_initialization(self, temp_dir):
        """Test store creates database and tables."""
        db_path = temp_dir / "test.db"
        SQLiteStore(db_path)

        assert db_path.exists()

    def test_create_session(self, temp_dir):
        """Test creating a research session."""
        store = SQLiteStore(temp_dir / "test.db")

        session = store.create_session(
            session_id="test-session-1",
            problem="Test research problem",
            metadata={"key": "value"},
        )

        assert session["id"] == "test-session-1"
        assert session["problem"] == "Test research problem"
        assert session["status"] == "active"

    def test_get_session(self, temp_dir):
        """Test getting a session by ID."""
        store = SQLiteStore(temp_dir / "test.db")
        store.create_session("session-1", "Problem 1")

        session = store.get_session("session-1")

        assert session is not None
        assert session["id"] == "session-1"

    def test_get_nonexistent_session(self, temp_dir):
        """Test getting a non-existent session."""
        store = SQLiteStore(temp_dir / "test.db")

        session = store.get_session("nonexistent")

        assert session is None

    def test_update_session(self, temp_dir):
        """Test updating a session."""
        store = SQLiteStore(temp_dir / "test.db")
        store.create_session("session-1", "Original problem")

        store.update_session("session-1", status="completed", plan="The plan")

        session = store.get_session("session-1")
        assert session["status"] == "completed"
        assert session["plan"] == "The plan"

    def test_list_sessions(self, temp_dir):
        """Test listing all sessions."""
        store = SQLiteStore(temp_dir / "test.db")
        store.create_session("session-1", "Problem 1")
        store.create_session("session-2", "Problem 2")

        sessions = store.list_sessions()

        assert len(sessions) == 2

    def test_list_sessions_by_status(self, temp_dir):
        """Test filtering sessions by status."""
        store = SQLiteStore(temp_dir / "test.db")
        store.create_session("session-1", "Problem 1")
        store.create_session("session-2", "Problem 2")
        store.update_session("session-2", status="completed")

        active = store.list_sessions(status="active")
        completed = store.list_sessions(status="completed")

        assert len(active) == 1
        assert len(completed) == 1

    def test_create_agent_state(self, temp_dir):
        """Test creating an agent state."""
        store = SQLiteStore(temp_dir / "test.db")
        store.create_session("session-1", "Problem")

        agent = store.create_agent_state(
            agent_id="agent-1",
            session_id="session-1",
            agent_name="test-agent",
            focus_area="Testing",
        )

        assert agent["id"] == "agent-1"
        assert agent["agent_name"] == "test-agent"

    def test_get_session_agents(self, temp_dir):
        """Test getting all agents for a session."""
        store = SQLiteStore(temp_dir / "test.db")
        store.create_session("session-1", "Problem")
        store.create_agent_state("agent-1", "session-1", "Agent 1", "Focus 1")
        store.create_agent_state("agent-2", "session-1", "Agent 2", "Focus 2")

        agents = store.get_session_agents("session-1")

        assert len(agents) == 2

    def test_log_hook(self, temp_dir):
        """Test logging hook execution."""
        store = SQLiteStore(temp_dir / "test.db")

        store.log_hook(
            hook_name="test_hook",
            event_type="plan_creation",
            session_id="session-1",
            payload={"key": "value"},
        )

        # Verify log was created (no public method to read, but no error)


class TestPersistenceManager:
    """Tests for PersistenceManager."""

    def test_manager_initialization(self, temp_dir):
        """Test manager creates directories."""
        PersistenceManager(
            db_path=temp_dir / "test.db",
            base_output_dir=temp_dir / "output",
        )

        assert (temp_dir / "output").exists()

    def test_get_session_dir(self, temp_dir):
        """Test getting session directory."""
        manager = PersistenceManager(
            db_path=temp_dir / "test.db",
            base_output_dir=temp_dir / "output",
        )

        session_dir = manager.get_session_dir("test-session")

        assert session_dir.exists()
        assert session_dir.name == "test-session"

    def test_get_agent_dir(self, temp_dir):
        """Test getting agent directory."""
        manager = PersistenceManager(
            db_path=temp_dir / "test.db",
            base_output_dir=temp_dir / "output",
        )

        agent_dir = manager.get_agent_dir("session-1", "agent-1")

        assert agent_dir.exists()
        assert agent_dir.name == "agent-1"
        assert agent_dir.parent.name == "session-1"

    def test_write_plan(self, temp_dir):
        """Test writing research plan."""
        manager = PersistenceManager(
            db_path=temp_dir / "test.db",
            base_output_dir=temp_dir / "output",
        )
        manager.store.create_session("session-1", "Problem")

        plan_path = manager.write_plan("session-1", "# Research Plan\n\nContent here.")

        assert plan_path.exists()
        assert plan_path.name == "RESEARCH_PLAN.md"
        assert "Research Plan" in plan_path.read_text()

    def test_write_agent_result(self, temp_dir):
        """Test writing agent result file."""
        manager = PersistenceManager(
            db_path=temp_dir / "test.db",
            base_output_dir=temp_dir / "output",
        )

        result_path = manager.write_agent_result(
            session_id="session-1",
            agent_name="agent-1",
            filename="findings.md",
            content="# Findings\n\nContent here.",
        )

        assert result_path.exists()
        assert result_path.name == "findings.md"
        assert "Findings" in result_path.read_text()
