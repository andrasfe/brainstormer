"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_settings(temp_dir):
    """Create sample settings for testing."""
    from brainstormer.config import Settings

    return Settings(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        tavily_api_key="test-tavily-key",
        sqlite_db_path=temp_dir / "test.db",
        chromadb_path=temp_dir / "chromadb",
        skills_dir=temp_dir / "skills",
    )


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is sample text content for testing.\nLine 2.\nLine 3.")
    return file_path


@pytest.fixture
def sample_skill_dir(temp_dir):
    """Create a sample skill directory."""
    skill_dir = temp_dir / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("""---
name: test-skill
description: A test skill for unit testing
---

# Test Skill

This is a test skill.

## Guidelines

- Guideline 1
- Guideline 2
""")
    return temp_dir / "skills"


@pytest.fixture
def sample_subagents_file(temp_dir):
    """Create a sample subagents.jsonl file."""
    file_path = temp_dir / "subagents.jsonl"
    file_path.write_text("""\
{"name": "test-agent-1", "description": "Test agent 1", "system_prompt": "You are test agent 1", "focus_areas": ["testing", "qa"]}
{"name": "test-agent-2", "description": "Test agent 2", "system_prompt": "You are test agent 2", "focus_areas": ["research"]}
""")
    return file_path
