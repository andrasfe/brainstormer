"""SQLite-based persistence for agent state and research sessions."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteStore:
    """SQLite storage for research sessions and agent state."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id TEXT PRIMARY KEY,
                    problem TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    plan TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_states (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    focus_area TEXT,
                    status TEXT DEFAULT 'pending',
                    result_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    state_data TEXT,
                    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
                );

                CREATE TABLE IF NOT EXISTS research_artifacts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT,
                    artifact_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES research_sessions(id),
                    FOREIGN KEY (agent_id) REFERENCES agent_states(id)
                );

                CREATE TABLE IF NOT EXISTS hooks_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    hook_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    result TEXT,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_agent_states_session
                    ON agent_states(session_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_session
                    ON research_artifacts(session_id);
                CREATE INDEX IF NOT EXISTS idx_hooks_session
                    ON hooks_log(session_id);
            """)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Research Sessions
    def create_session(
        self, session_id: str, problem: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a new research session."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO research_sessions (id, problem, metadata)
                VALUES (?, ?, ?)
                """,
                (session_id, problem, json.dumps(metadata or {})),
            )
        logger.info(f"Created research session: {session_id}")
        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Failed to create session: {session_id}")
        return session

    def get_session(self, session_id: str) -> dict | None:
        """Get a research session by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM research_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def update_session(self, session_id: str, **kwargs: Any) -> None:
        """Update a research session."""
        allowed_fields = {"problem", "status", "plan", "metadata"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            with self._connection() as conn:
                conn.execute(
                    f"""
                    UPDATE research_sessions
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (*updates.values(), session_id),
                )

    def list_sessions(self, status: str | None = None) -> list[dict]:
        """List all research sessions."""
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM research_sessions WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM research_sessions ORDER BY created_at DESC"
                ).fetchall()
            return [dict(row) for row in rows]

    # Agent States
    def create_agent_state(
        self,
        agent_id: str,
        session_id: str,
        agent_name: str,
        focus_area: str,
        state_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an agent state record."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_states (id, session_id, agent_name, focus_area, state_data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agent_id, session_id, agent_name, focus_area, json.dumps(state_data or {})),
            )
        agent = self.get_agent_state(agent_id)
        if agent is None:
            raise RuntimeError(f"Failed to create agent state: {agent_id}")
        return agent

    def get_agent_state(self, agent_id: str) -> dict | None:
        """Get an agent state by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_states WHERE id = ?", (agent_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def update_agent_state(self, agent_id: str, **kwargs: Any) -> None:
        """Update an agent state."""
        allowed_fields = {"status", "result_path", "state_data"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if "state_data" in updates and isinstance(updates["state_data"], dict):
            updates["state_data"] = json.dumps(updates["state_data"])

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            with self._connection() as conn:
                conn.execute(
                    f"""
                    UPDATE agent_states
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (*updates.values(), agent_id),
                )

    def get_session_agents(self, session_id: str) -> list[dict]:
        """Get all agents for a session."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_states WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # Research Artifacts
    def create_artifact(
        self,
        artifact_id: str,
        session_id: str,
        artifact_type: str,
        file_path: str,
        agent_id: str | None = None,
        content_hash: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a research artifact."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO research_artifacts
                (id, session_id, agent_id, artifact_type, file_path, content_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    session_id,
                    agent_id,
                    artifact_type,
                    file_path,
                    content_hash,
                    json.dumps(metadata or {}),
                ),
            )

    def get_session_artifacts(self, session_id: str) -> list[dict]:
        """Get all artifacts for a session."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM research_artifacts WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # Hooks Log
    def log_hook(
        self,
        hook_name: str,
        event_type: str,
        session_id: str | None = None,
        payload: dict | None = None,
        result: Any = None,
    ) -> None:
        """Log a hook execution."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO hooks_log (session_id, hook_name, event_type, payload, result)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    hook_name,
                    event_type,
                    json.dumps(payload) if payload else None,
                    json.dumps(result) if result else None,
                ),
            )


class PersistenceManager:
    """High-level persistence manager combining SQLite with file system."""

    def __init__(self, db_path: Path, base_output_dir: Path):
        self.store = SQLiteStore(db_path)
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self, session_id: str) -> Path:
        """Get the output directory for a session."""
        session_dir = self.base_output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def get_agent_dir(self, session_id: str, agent_name: str) -> Path:
        """Get the output directory for an agent within a session."""
        agent_dir = self.get_session_dir(session_id) / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def write_plan(self, session_id: str, plan_content: str) -> Path:
        """Write the research plan to the session directory."""
        session_dir = self.get_session_dir(session_id)
        plan_path = session_dir / "RESEARCH_PLAN.md"
        plan_path.write_text(plan_content, encoding="utf-8")
        self.store.update_session(session_id, plan=plan_content)
        return plan_path

    def write_agent_result(
        self,
        session_id: str,
        agent_name: str,
        filename: str,
        content: str,
    ) -> Path:
        """Write an agent's result file."""
        agent_dir = self.get_agent_dir(session_id, agent_name)
        result_path = agent_dir / filename
        result_path.write_text(content, encoding="utf-8")
        return result_path
